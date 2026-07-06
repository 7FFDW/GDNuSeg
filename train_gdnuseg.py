# -*- coding: utf-8 -*-
r"""
GDNuSeg / PromptTexNet training script.

This script is adapted to models whose forward function is:
    output = model(image, glcm, densitymap)

Important:
    Density maps are used only during training.
    During validation / testing, the model is called as:
        output = model(image, glcm, None)

Expected data structure:
    data_root/
        images/
            xxx.png / xxx.jpg / xxx.tif ...
        masks/
            xxx.png / xxx.jpg / xxx.tif ...
        glcm/                 # optional; if absent, GLCM texture maps are computed online
            xxx.png / xxx.jpg / xxx.tif ...

For binary segmentation:
    --n_classes 1
    masks should be 0/255 or 0/1.

For multi-class segmentation:
    --n_classes K
    masks should contain integer labels in [0, K-1].

Example:
python train_gdnuseg.py \
    --train_root "D:/data/MoNuSeg/train" \
    --val_root   "D:/data/MoNuSeg/val" \
    --model_module "nets.PromptTexNet.PromptTexNet" \
    --model_name "DGCSNet" \
    --n_classes 1 \
    --density_radius 2 \
    --epochs 300 \
    --batch_size 8 \
    --lr 1e-3 \
    --save_dir "./checkpoints/GDNuSeg/MoNuSeg"
"""

import argparse
import importlib
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# =========================================================
# 1. Reproducibility
# =========================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# =========================================================
# 2. Basic image I/O
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def cv_imread_gray(path: Path) -> np.ndarray:
    mask = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


# =========================================================
# 3. Data augmentation
# =========================================================

def resize_pair(image: np.ndarray, mask: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return image, mask


def random_augment(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Horizontal flip
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])

    # Vertical flip
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[::-1, :])
        mask = np.ascontiguousarray(mask[::-1, :])

    # 0, 90, 180, 270 rotation
    k = random.randint(0, 3)
    if k > 0:
        image = np.ascontiguousarray(np.rot90(image, k))
        mask = np.ascontiguousarray(np.rot90(mask, k))

    return image, mask


# =========================================================
# 4. Online GLCM texture map
# =========================================================

def compute_global_glcm_features(
    image_rgb: np.ndarray,
    levels: int = 16,
    distance: int = 4,
    angles: Tuple[Tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (-1, 1)),
) -> np.ndarray:
    """
    Compute a compact 3-channel GLCM texture map from the augmented image.

    Output shape: [H, W, 3]
    Channel 0: contrast
    Channel 1: homogeneity
    Channel 2: energy

    The three scalar GLCM statistics are broadcast to spatial maps because the
    model's texture encoder expects a 3-channel image-like tensor.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Quantize to [0, levels - 1]
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
        glcm += counts.T  # symmetric GLCM

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
# 5. Density map generation from mask
# =========================================================

def gaussian_kernel(radius: int) -> np.ndarray:
    radius = int(max(1, radius))
    sigma = radius / 3.0
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2 + 1e-12))
    kernel = kernel / (kernel.sum() + 1e-12)
    return kernel.astype(np.float32)


def build_density_map_from_mask(mask: np.ndarray, radius: int = 4) -> np.ndarray:
    """
    Build a density map using centroids of connected nuclei regions.
    For multi-class masks, all foreground classes are merged for centroid extraction.
    """
    fg = (mask > 0).astype(np.uint8)
    h, w = fg.shape
    density = np.zeros((h, w), dtype=np.float32)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_labels <= 1:
        return density

    kernel = gaussian_kernel(radius)
    r = radius

    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area <= 0:
            continue
        cx, cy = centroids[lab]
        cx, cy = int(round(cx)), int(round(cy))

        x1, x2 = max(0, cx - r), min(w, cx + r + 1)
        y1, y2 = max(0, cy - r), min(h, cy + r + 1)

        kx1, kx2 = x1 - (cx - r), kernel.shape[1] - ((cx + r + 1) - x2)
        ky1, ky2 = y1 - (cy - r), kernel.shape[0] - ((cy + r + 1) - y2)

        density[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]

    # Keep a stable range for the prompt encoder / PointGenerator.
    if density.max() > 0:
        density = density / density.max()
    return density.astype(np.float32)


# =========================================================
# 6. Dataset
# =========================================================

class NucleiSegDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        image_dir: str = "images",
        mask_dir: str = "masks",
        glcm_dir: str = "glcm",
        size: int = 256,
        n_classes: int = 1,
        train: bool = True,
        density_radius: int = 4,
        glcm_distance: int = 4,
        use_precomputed_glcm: bool = False,
    ) -> None:
        self.root = Path(data_root)
        self.image_root = self.root / image_dir
        self.mask_root = self.root / mask_dir
        self.glcm_root = self.root / glcm_dir
        self.size = size
        self.n_classes = n_classes
        self.train = train
        self.density_radius = density_radius
        self.glcm_distance = glcm_distance
        self.use_precomputed_glcm = use_precomputed_glcm and self.glcm_root.exists()

        if not self.image_root.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_root}")
        if not self.mask_root.exists():
            raise FileNotFoundError(f"Mask folder not found: {self.mask_root}")

        self.image_paths: List[Path] = []
        for ext in IMG_EXTS:
            self.image_paths.extend(sorted(self.image_root.glob(f"*{ext}")))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.image_root}")

        self.samples: List[Tuple[Path, Path, Optional[Path]]] = []
        for img_path in self.image_paths:
            mask_path = find_file_by_stem(self.mask_root, img_path.stem)
            if mask_path is None:
                continue
            glcm_path = find_file_by_stem(self.glcm_root, img_path.stem) if self.use_precomputed_glcm else None
            self.samples.append((img_path, mask_path, glcm_path))

        if len(self.samples) == 0:
            raise RuntimeError("No matched image-mask pairs found.")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path, glcm_path = self.samples[idx]

        image = cv_imread_rgb(img_path)
        mask = cv_imread_gray(mask_path)

        image, mask = resize_pair(image, mask, self.size)

        if self.train:
            image, mask = random_augment(image, mask)

        # Mask processing
        if self.n_classes == 1:
            mask_bin = (mask > 0).astype(np.float32)
            target = torch.from_numpy(mask_bin[None, ...]).float()
            density_mask = mask_bin.astype(np.uint8)
        else:
            mask_cls = mask.astype(np.int64)
            mask_cls = np.clip(mask_cls, 0, self.n_classes - 1)
            target = torch.from_numpy(mask_cls).long()
            density_mask = (mask_cls > 0).astype(np.uint8)

        # Density map should be [H, W], because PointGenerator in the model receives densitymap[idx].
        density = build_density_map_from_mask(density_mask, radius=self.density_radius)
        density = torch.from_numpy(density).float()

        # GLCM texture map: [3, H, W]
        if glcm_path is not None and glcm_path.exists():
            glcm = cv_imread_rgb(glcm_path)
            glcm = cv2.resize(glcm, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            glcm = glcm.astype(np.float32) / 255.0
        else:
            glcm = compute_global_glcm_features(image, distance=self.glcm_distance)

        # Image normalization for ResNet encoder
        image_f = image.astype(np.float32) / 255.0
        image_f = (image_f - self.mean) / self.std

        image_t = torch.from_numpy(image_f.transpose(2, 0, 1)).float()
        glcm_t = torch.from_numpy(glcm.transpose(2, 0, 1)).float()

        return {
            "image": image_t,
            "mask": target,
            "glcm": glcm_t,
            "density": density,
            "name": img_path.name,
        }


# =========================================================
# 7. Losses and metrics
# =========================================================

class BinaryDiceBCELoss(nn.Module):
    """For n_classes=1. The uploaded model already applies Sigmoid in binary mode."""
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()

    def forward(self, pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = pred_prob.clamp(1e-6, 1.0 - 1e-6)
        bce = self.bce(pred_prob, target)
        dims = (1, 2, 3)
        inter = torch.sum(pred_prob * target, dim=dims)
        union = torch.sum(pred_prob, dim=dims) + torch.sum(target, dim=dims)
        dice_loss = 1.0 - torch.mean((2.0 * inter + 1e-6) / (union + 1e-6))
        return self.bce_weight * bce + self.dice_weight * dice_loss


class MultiClassDiceCELoss(nn.Module):
    """For n_classes>1. The uploaded model returns logits in multi-class mode."""
    def __init__(self, n_classes: int, dice_weight: float = 1.0, ce_weight: float = 1.0, class_weights=None) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        prob = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target, num_classes=self.n_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        inter = torch.sum(prob * onehot, dim=dims)
        union = torch.sum(prob, dim=dims) + torch.sum(onehot, dim=dims)
        # Exclude background from Dice if possible.
        dice_per_class = (2.0 * inter + 1e-6) / (union + 1e-6)
        if self.n_classes > 1:
            dice_per_class = dice_per_class[1:]
        dice_loss = 1.0 - dice_per_class.mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss


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
# 8. Model builder
# =========================================================

def build_default_config() -> SimpleNamespace:
    # decoder_channels[0] must match the last decoder feature channels, which is 64 in the uploaded model.
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


# =========================================================
# 9. Train / validate
# =========================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_classes: int,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    model.train()
    total_loss, total_iou, total_dice, n_batches = 0.0, 0.0, 0.0, 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        glcm = batch["glcm"].to(device, non_blocking=True)
        density = batch["density"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                output = model(image, glcm, density)
                loss = criterion(output, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(image, glcm, density)
            loss = criterion(output, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if n_classes == 1:
            iou, dice = binary_iou_dice(output.detach(), mask.detach())
        else:
            iou, dice = multiclass_iou_dice(output.detach(), mask.detach(), n_classes)

        total_loss += loss.item()
        total_iou += iou
        total_dice += dice
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}", dice=f"{dice:.4f}")

    return {
        "loss": total_loss / max(1, n_batches),
        "iou": total_iou / max(1, n_batches),
        "dice": total_dice / max(1, n_batches),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int,
) -> Dict[str, float]:
    model.eval()
    total_loss, total_iou, total_dice, n_batches = 0.0, 0.0, 0.0, 0

    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        glcm = batch["glcm"].to(device, non_blocking=True)
        # During validation / testing, density maps are not used.
        # This is consistent with the inference setting of GDNuSeg, where PH-Encoder
        # provides training-time positional guidance only.
        output = model(image, glcm, None)
        loss = criterion(output, mask)

        if n_classes == 1:
            iou, dice = binary_iou_dice(output, mask)
        else:
            iou, dice = multiclass_iou_dice(output, mask, n_classes)

        total_loss += loss.item()
        total_iou += iou
        total_dice += dice
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}", dice=f"{dice:.4f}")

    return {
        "loss": total_loss / max(1, n_batches),
        "iou": total_iou / max(1, n_batches),
        "dice": total_dice / max(1, n_batches),
    }


def save_checkpoint(
    save_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_dice: float,
    args: argparse.Namespace,
) -> None:
    state = {
        "epoch": epoch,
        "best_dice": best_dice,
        "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
    }
    torch.save(state, save_path)


# =========================================================
# 10. Main
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--val_root", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default="images")
    parser.add_argument("--mask_dir", type=str, default="masks")
    parser.add_argument("--glcm_dir", type=str, default="glcm")
    parser.add_argument("--use_precomputed_glcm", action="store_true")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--n_classes", type=int, default=1)

    # Dataset-specific PH-Encoder and GLCM settings
    parser.add_argument("--density_radius", type=int, default=4, help="Dataset-specific Gaussian radius r_D.")
    parser.add_argument("--glcm_distance", type=int, default=4, help="Distance d used for online GLCM computation.")
    parser.add_argument("--glcm_levels", type=int, default=16)

    # Model
    parser.add_argument("--model_module", type=str, default="nets.PromptTexNet.PromptTexNet")
    parser.add_argument("--model_name", type=str, default="DGCSNet", choices=["DGCSNet", "PromptTexNet"])
    parser.add_argument("--pretrained", type=str, default="", help="Optional checkpoint path to resume/load.")
    parser.add_argument("--mgpu", action="store_true")

    # Optimization
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/GDNuSeg")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Device:", device)
    print("[INFO] Train root:", args.train_root)
    print("[INFO] Val root:", args.val_root)
    print("[INFO] n_classes:", args.n_classes)
    print("[INFO] density_radius:", args.density_radius)
    print("[INFO] glcm_distance:", args.glcm_distance)
    print("[INFO] Density map is used during training only; validation/testing uses densitymap=None.")

    train_set = NucleiSegDataset(
        data_root=args.train_root,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        glcm_dir=args.glcm_dir,
        size=args.img_size,
        n_classes=args.n_classes,
        train=True,
        density_radius=args.density_radius,
        glcm_distance=args.glcm_distance,
        use_precomputed_glcm=args.use_precomputed_glcm,
    )
    val_set = NucleiSegDataset(
        data_root=args.val_root,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        glcm_dir=args.glcm_dir,
        size=args.img_size,
        n_classes=args.n_classes,
        train=False,
        density_radius=args.density_radius,
        glcm_distance=args.glcm_distance,
        use_precomputed_glcm=args.use_precomputed_glcm,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(args.model_module, args.model_name, args.n_classes, device)

    if args.mgpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("[INFO] Loaded checkpoint:", args.pretrained)
        print("[INFO] Missing keys:", len(missing), "Unexpected keys:", len(unexpected))

    if args.n_classes == 1:
        criterion = BinaryDiceBCELoss(dice_weight=1.0, bce_weight=1.0)
    else:
        criterion = MultiClassDiceCELoss(n_classes=args.n_classes, dice_weight=1.0, ce_weight=1.0)
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = GradScaler() if args.amp and device.type == "cuda" else None

    best_dice = -1.0
    log_path = save_dir / "train_log.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,lr,train_loss,train_iou,train_dice,val_loss,val_iou,val_dice,best_dice\n")

    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch}/{args.epochs}] lr={lr:.6g}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            n_classes=args.n_classes,
            scaler=scaler,
        )
        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            n_classes=args.n_classes,
        )
        scheduler.step()

        print(
            f"Train: loss={train_metrics['loss']:.4f}, IoU={train_metrics['iou']:.4f}, Dice={train_metrics['dice']:.4f} | "
            f"Val: loss={val_metrics['loss']:.4f}, IoU={val_metrics['iou']:.4f}, Dice={val_metrics['dice']:.4f}"
        )

        is_best = val_metrics["dice"] > best_dice
        if is_best:
            best_dice = val_metrics["dice"]
            save_checkpoint(save_dir / "best.pth", model, optimizer, scheduler, epoch, best_dice, args)
            print(f"[INFO] Saved best checkpoint. best_dice={best_dice:.4f}")

        if epoch % 20 == 0 or epoch == args.epochs:
            save_checkpoint(save_dir / f"epoch_{epoch:03d}.pth", model, optimizer, scheduler, epoch, best_dice, args)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{lr:.8f},{train_metrics['loss']:.6f},{train_metrics['iou']:.6f},{train_metrics['dice']:.6f},"
                f"{val_metrics['loss']:.6f},{val_metrics['iou']:.6f},{val_metrics['dice']:.6f},{best_dice:.6f}\n"
            )

    print("[DONE] Training finished.")
    print("[DONE] Best Dice:", best_dice)
    print("[DONE] Checkpoints saved to:", save_dir)


if __name__ == "__main__":
    main()
