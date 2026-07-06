# -*- coding: utf-8 -*-
r"""
GDNuSeg / PromptTexNet testing script.

This script is adapted to models whose forward function is:
    output = model(image, glcm, densitymap)

Important:
    During testing / inference, density maps are NOT used:
        output = model(image, glcm, None)

Expected data structure:
    test_root/
        images/
            xxx.png / xxx.jpg / xxx.tif ...
        masks/                 # optional, only needed when computing metrics
            xxx.png / xxx.jpg / xxx.tif ...
        glcm/                  # optional, if absent GLCM maps are computed online
            xxx.png / xxx.jpg / xxx.tif ...

For binary segmentation:
    --n_classes 1
    masks should be 0/255 or 0/1.

For multi-class segmentation:
    --n_classes K
    masks should contain integer labels in [0, K-1].

Example:
python test_gdnuseg.py \
    --test_root "D:/data/MoNuSeg/test" \
    --checkpoint "./checkpoints/GDNuSeg/MoNuSeg/best.pth" \
    --model_module "nets.PromptTexNet.PromptTexNet" \
    --model_name "DGCSNet" \
    --n_classes 1 \
    --glcm_distance 5 \
    --save_dir "./results/GDNuSeg/MoNuSeg"
"""

import argparse
import csv
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# =========================================================
# 1. Basic image I/O
# =========================================================

IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


def find_file_by_stem(folder: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def cv_imread_rgb(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cv_imread_gray(path: Path) -> np.ndarray:
    mask = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def cv_imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    buf.tofile(str(path))


# =========================================================
# 2. GLCM texture map
# =========================================================


def compute_global_glcm_features(
    image_rgb: np.ndarray,
    levels: int = 16,
    distance: int = 4,
    angles: Tuple[Tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (-1, 1)),
) -> np.ndarray:
    """
    Compute a compact 3-channel GLCM texture map from the input image.

    Output shape: [H, W, 3]
    Channel 0: contrast
    Channel 1: homogeneity
    Channel 2: energy

    The scalar GLCM statistics are broadcast to spatial maps because the
    uploaded model uses a 3-channel image-like texture input.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    q = np.floor(gray.astype(np.float32) / 256.0 * levels).astype(np.int32)
    q = np.clip(q, 0, levels - 1)

    glcm = np.zeros((levels, levels), dtype=np.float64)

    for dx, dy in angles:
        off_x = dx * distance
        off_y = dy * distance

        y0 = max(0, -off_y)
        y1 = min(h, h - off_y)
        x0 = max(0, -off_x)
        x1 = min(w, w - off_x)

        if y1 <= y0 or x1 <= x0:
            continue

        a = q[y0:y1, x0:x1].reshape(-1)
        b = q[y0 + off_y:y1 + off_y, x0 + off_x:x1 + off_x].reshape(-1)
        idx = a * levels + b
        counts = np.bincount(idx, minlength=levels * levels).reshape(levels, levels)
        glcm += counts
        glcm += counts.T

    total = glcm.sum()
    if total <= 0:
        return np.zeros((h, w, 3), dtype=np.float32)

    p = glcm / (total + 1e-12)
    ii, jj = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")

    contrast = np.sum(((ii - jj) ** 2) * p) / ((levels - 1) ** 2 + 1e-12)
    homogeneity = np.sum(p / (1.0 + np.abs(ii - jj)))
    energy = np.sqrt(np.sum(p ** 2))

    feat = np.array([contrast, homogeneity, energy], dtype=np.float32)
    feat = np.clip(feat, 0.0, 1.0)
    glcm_map = np.ones((h, w, 3), dtype=np.float32) * feat.reshape(1, 1, 3)
    return glcm_map


# =========================================================
# 3. Dataset
# =========================================================


class NucleiTestDataset(Dataset):
    def __init__(
        self,
        test_root: str,
        image_dir: str = "images",
        mask_dir: str = "masks",
        glcm_dir: str = "glcm",
        size: int = 256,
        n_classes: int = 1,
        glcm_distance: int = 4,
        glcm_levels: int = 16,
        use_precomputed_glcm: bool = False,
    ) -> None:
        self.root = Path(test_root)
        self.image_root = self.root / image_dir
        self.mask_root = self.root / mask_dir
        self.glcm_root = self.root / glcm_dir
        self.size = size
        self.n_classes = n_classes
        self.glcm_distance = glcm_distance
        self.glcm_levels = glcm_levels
        self.use_precomputed_glcm = use_precomputed_glcm and self.glcm_root.exists()
        self.has_masks = self.mask_root.exists()

        if not self.image_root.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_root}")

        image_paths: List[Path] = []
        for ext in IMG_EXTS:
            image_paths.extend(sorted(self.image_root.glob(f"*{ext}")))
        if len(image_paths) == 0:
            raise RuntimeError(f"No images found in {self.image_root}")

        self.samples: List[Tuple[Path, Optional[Path], Optional[Path]]] = []
        for img_path in image_paths:
            mask_path = find_file_by_stem(self.mask_root, img_path.stem) if self.has_masks else None
            glcm_path = find_file_by_stem(self.glcm_root, img_path.stem) if self.use_precomputed_glcm else None
            self.samples.append((img_path, mask_path, glcm_path))

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path, glcm_path = self.samples[idx]

        image = cv_imread_rgb(img_path)
        ori_h, ori_w = image.shape[:2]
        image_resized = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        # GLCM texture map: [3, H, W]
        if glcm_path is not None and glcm_path.exists():
            glcm = cv_imread_rgb(glcm_path)
            glcm = cv2.resize(glcm, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            glcm = glcm.astype(np.float32) / 255.0
        else:
            glcm = compute_global_glcm_features(
                image_resized,
                levels=self.glcm_levels,
                distance=self.glcm_distance,
            )

        image_f = image_resized.astype(np.float32) / 255.0
        image_f = (image_f - self.mean) / self.std
        image_t = torch.from_numpy(image_f.transpose(2, 0, 1)).float()
        glcm_t = torch.from_numpy(glcm.transpose(2, 0, 1)).float()

        item = {
            "image": image_t,
            "glcm": glcm_t,
            "name": img_path.name,
            "ori_h": torch.tensor(ori_h, dtype=torch.long),
            "ori_w": torch.tensor(ori_w, dtype=torch.long),
        }

        if mask_path is not None and mask_path.exists():
            mask = cv_imread_gray(mask_path)
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            if self.n_classes == 1:
                mask = (mask > 0).astype(np.float32)
                item["mask"] = torch.from_numpy(mask[None, ...]).float()
            else:
                mask = np.clip(mask.astype(np.int64), 0, self.n_classes - 1)
                item["mask"] = torch.from_numpy(mask).long()
        else:
            if self.n_classes == 1:
                item["mask"] = torch.zeros((1, self.size, self.size), dtype=torch.float32)
            else:
                item["mask"] = torch.zeros((self.size, self.size), dtype=torch.long)

        return item


# =========================================================
# 4. Metrics
# =========================================================


@torch.no_grad()
def binary_iou_dice(pred_prob: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    pred = (pred_prob > 0.5).float()
    target = (target > 0.5).float()
    inter = torch.sum(pred * target, dim=(1, 2, 3))
    pred_sum = torch.sum(pred, dim=(1, 2, 3))
    target_sum = torch.sum(target, dim=(1, 2, 3))
    union = pred_sum + target_sum - inter
    iou = ((inter + 1e-6) / (union + 1e-6)).mean().item()
    dice = ((2.0 * inter + 1e-6) / (pred_sum + target_sum + 1e-6)).mean().item()
    return iou, dice


@torch.no_grad()
def multiclass_iou_dice(logits: torch.Tensor, target: torch.Tensor, n_classes: int) -> Tuple[float, float]:
    pred = torch.argmax(logits, dim=1)
    ious, dices = [], []
    for cls in range(1, n_classes):
        p = (pred == cls).float()
        t = (target == cls).float()
        inter = torch.sum(p * t)
        p_sum = torch.sum(p)
        t_sum = torch.sum(t)
        union = p_sum + t_sum - inter
        if t_sum.item() == 0 and p_sum.item() == 0:
            continue
        ious.append(((inter + 1e-6) / (union + 1e-6)).item())
        dices.append(((2.0 * inter + 1e-6) / (p_sum + t_sum + 1e-6)).item())
    if len(ious) == 0:
        return 0.0, 0.0
    return float(np.mean(ious)), float(np.mean(dices))


# =========================================================
# 5. Visualization helpers
# =========================================================


def get_default_palette(n_classes: int) -> np.ndarray:
    base = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 0],
            [128, 0, 255],
            [128, 255, 0],
            [255, 0, 128],
            [0, 128, 255],
        ],
        dtype=np.uint8,
    )
    if n_classes <= len(base):
        return base[:n_classes]
    rng = np.random.default_rng(123)
    extra = rng.integers(0, 255, size=(n_classes - len(base), 3), dtype=np.uint8)
    return np.concatenate([base, extra], axis=0)


def mask_to_color(mask: np.ndarray, n_classes: int) -> np.ndarray:
    palette = get_default_palette(max(n_classes, 2))
    mask = np.clip(mask.astype(np.int64), 0, len(palette) - 1)
    color = palette[mask]
    return color


def make_overlay(image_rgb: np.ndarray, mask_color_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image_rgb = image_rgb.astype(np.float32)
    mask_color_rgb = mask_color_rgb.astype(np.float32)
    overlay = image_rgb * (1.0 - alpha) + mask_color_rgb * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


# =========================================================
# 6. Model builder and checkpoint loading
# =========================================================


def build_default_config() -> SimpleNamespace:
    return SimpleNamespace(
        decoder_channels=[64, 128, 256, 512],
        transformer=SimpleNamespace(embedding_channels=512),
        patch_sizes=[16, 8, 4, 2],
    )


def build_model(model_module: str, model_name: str, n_classes: int, device: torch.device) -> nn.Module:
    module = importlib.import_module(model_module)
    model_cls = getattr(module, model_name)
    config = build_default_config()
    model = model_cls(config=config, n_channels=3, n_classes=n_classes)
    return model.to(device)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    state = strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    print(f"[INFO] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("[WARN] First missing keys:", missing[:5])
    if len(unexpected) > 0:
        print("[WARN] First unexpected keys:", unexpected[:5])


# =========================================================
# 7. Testing
# =========================================================


@torch.no_grad()
def run_test(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    save_dir = Path(args.save_dir)
    pred_dir = save_dir / "pred_masks"
    color_dir = save_dir / "color_masks"
    overlay_dir = save_dir / "overlays"
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    if args.save_color:
        color_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    dataset = NucleiTestDataset(
        test_root=args.test_root,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        glcm_dir=args.glcm_dir,
        size=args.img_size,
        n_classes=args.n_classes,
        glcm_distance=args.glcm_distance,
        glcm_levels=args.glcm_levels,
        use_precomputed_glcm=args.use_precomputed_glcm,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(args.model_module, args.model_name, args.n_classes, device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    csv_path = save_dir / "test_results.csv"
    has_gt = dataset.has_masks
    total_iou, total_dice, total_n = 0.0, 0.0, 0

    print("[INFO] Device:", device)
    print("[INFO] Test root:", args.test_root)
    print("[INFO] n_classes:", args.n_classes)
    print("[INFO] glcm_distance:", args.glcm_distance)
    print("[INFO] Density map is NOT used during testing: output = model(image, glcm, None).")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "iou", "dice"] if has_gt else ["name"])

        pbar = tqdm(loader, desc="Test", leave=True)
        for batch in pbar:
            image = batch["image"].to(device, non_blocking=True)
            glcm = batch["glcm"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            names = batch["name"]

            # Core inference rule of GDNuSeg: no density map during testing.
            output = model(image, glcm, None)

            if args.n_classes == 1:
                # The uploaded model already applies Sigmoid in binary mode.
                # This fallback makes the script robust if a logits-based binary model is used.
                if output.min().item() < 0.0 or output.max().item() > 1.0:
                    prob = torch.sigmoid(output)
                else:
                    prob = output.clamp(0.0, 1.0)
                pred = (prob > args.threshold).long().squeeze(1)  # [B, H, W]

                if has_gt:
                    iou, dice = binary_iou_dice(prob, mask)
                else:
                    iou, dice = 0.0, 0.0
            else:
                pred = torch.argmax(output, dim=1).long()  # [B, H, W]
                if has_gt:
                    iou, dice = multiclass_iou_dice(output, mask, args.n_classes)
                else:
                    iou, dice = 0.0, 0.0

            if has_gt:
                total_iou += iou
                total_dice += dice
                total_n += 1
                pbar.set_postfix(iou=f"{iou:.4f}", dice=f"{dice:.4f}")

            pred_np = pred.cpu().numpy().astype(np.uint8)

            for bi, name in enumerate(names):
                stem = Path(name).stem
                out_mask = pred_np[bi]

                # Save prediction mask at network resolution.
                if args.n_classes == 1:
                    save_mask = (out_mask * 255).astype(np.uint8)
                else:
                    save_mask = out_mask.astype(np.uint8)
                cv_imwrite(pred_dir / f"{stem}.png", save_mask)

                if args.save_color:
                    color_rgb = mask_to_color(out_mask, max(args.n_classes, 2))
                    color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
                    cv_imwrite(color_dir / f"{stem}.png", color_bgr)

                if args.save_overlay:
                    # Reload original image and resize to network resolution for overlay.
                    img_path = find_file_by_stem(dataset.image_root, stem)
                    if img_path is not None:
                        image_rgb = cv_imread_rgb(img_path)
                        image_rgb = cv2.resize(image_rgb, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
                        color_rgb = mask_to_color(out_mask, max(args.n_classes, 2))
                        overlay_rgb = make_overlay(image_rgb, color_rgb, alpha=args.overlay_alpha)
                        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                        cv_imwrite(overlay_dir / f"{stem}.png", overlay_bgr)

                if has_gt:
                    writer.writerow([name, f"{iou:.6f}", f"{dice:.6f}"])
                else:
                    writer.writerow([name])

    if has_gt and total_n > 0:
        mean_iou = total_iou / total_n
        mean_dice = total_dice / total_n
        with open(save_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write(f"Mean IoU:  {mean_iou:.6f}\n")
            f.write(f"Mean Dice: {mean_dice:.6f}\n")
        print(f"[DONE] Mean IoU:  {mean_iou:.6f}")
        print(f"[DONE] Mean Dice: {mean_dice:.6f}")
    else:
        print("[DONE] Testing finished. No ground-truth masks were found, so metrics were not computed.")

    print("[DONE] Results saved to:", save_dir)


# =========================================================
# 8. Args
# =========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--test_root", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default="images")
    parser.add_argument("--mask_dir", type=str, default="masks")
    parser.add_argument("--glcm_dir", type=str, default="glcm")
    parser.add_argument("--use_precomputed_glcm", action="store_true")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--n_classes", type=int, default=1)
    parser.add_argument("--glcm_distance", type=int, default=4)
    parser.add_argument("--glcm_levels", type=int, default=16)

    # Model
    parser.add_argument("--model_module", type=str, default="")
    parser.add_argument("--model_name", type=str, default="", choices=["GDNuSeg"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    # Inference
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)

    # Save
    parser.add_argument("--save_dir", type=str, default="./results/GDNuSeg")
    parser.add_argument("--save_color", action="store_true")
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--overlay_alpha", type=float, default=0.45)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_test(args)


if __name__ == "__main__":
    main()
