import torch
from torch import nn

from .PromptEncoder import PromptEncoder, PointGenerator
from .UncertFuse import DUU
from .transformer import TwoWayTransformer
import torch.nn.functional as F

class conv_block(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
class ResCNN_block(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(ResCNN_block, self).__init__()
        self.Conv = conv_block(ch_in, ch_out)
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv_1x1(x)
        x = self.Conv(x)
        return x + x1
class GDNuSeg(nn.Module):

    def __init__(self, img_ch=3, output_ch=1):
        super(GDNuSeg, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.ResCNN1 = ResCNN_block(ch_in=img_ch, ch_out=64)

        self.ResCNN2 = ResCNN_block(ch_in=64, ch_out=128)

        self.ResCNN3 = ResCNN_block(ch_in=128, ch_out=256)

        self.ResCNN4 = ResCNN_block(ch_in=256, ch_out=512)

        self.ResCNN5 = ResCNN_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_ResCNN5 = ResCNN_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_ResCNN4 = ResCNN_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_ResCNN3 = ResCNN_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_ResCNN2 = ResCNN_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.textureconv = nn.Conv2d(3, 1024, 1, 16, 0)
        self.fm = DUU(1024)

        self.prompt_encoder = PromptEncoder(
            embed_dim=1024,
            image_embedding_size=(256 // 16, 256 // 16),
            input_image_size=(256, 256),
            mask_in_chans=32,
        )
        self.num_multimask_outputs = 3
        self.transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=1024,
            mlp_dim=2048,
            num_heads=32,
        )

        self.iou_token = nn.Embedding(1, 1024)
        self.num_mask_tokens = self.num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, 1024)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.mlp_head = MLP(1024, 1024 // 8, 1024, 3)
        if output_ch == 1:
            self.last_activation = nn.Sigmoid()


    def forward(self, x,glcm=None,densitymap=None):

        x1 = self.ResCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.ResCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.ResCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.ResCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.ResCNN5(x5)

        b = x.shape[0]
        e5_out = []
        for idx in range(b):
            if densitymap is None:
                break

            e5_i = x5[idx]
            point_coord, point_class, mask_shape, box_coord = PointGenerator(densitymap[idx].cpu().numpy())

            density_map = densitymap[idx].unsqueeze(0).float()
            if density_map.dim() == 2:
                density_map = density_map.unsqueeze(0).unsqueeze(0)  # ??[H, W]±???[1, 1, H, W]

            elif density_map.dim() == 3:
                density_map = density_map.unsqueeze(1)

            # print("density_map shape:", density_map.shape)
            point_coord = torch.tensor(point_coord, device=self.device).float().unsqueeze(0)
            point_class = torch.tensor(point_class, device=self.device).long().unsqueeze(0)
            if point_coord.nelement() == 0:
                point_coord = torch.rand((1, 1, 2), device=self.device).float() * 256
                point_class = torch.tensor([[1]], device=self.device).long()

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point_coord, point_class),
                # points=None,
                boxes=None,
                masks=density_map,
            )

            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
            output_tokens = output_tokens.unsqueeze(0).expand(sparse_embeddings.size(0), -1, -1)
            tokens = torch.cat((output_tokens, sparse_embeddings), dim=1)

            if dense_embeddings.shape[2:] != e5_i.shape[1:]:
                dense_embeddings = F.interpolate(dense_embeddings, size=e5_i.shape[1:], mode='bilinear',
                                                 align_corners=False)
            e5_i = e5_i + dense_embeddings
            pos_src = torch.repeat_interleave(self.prompt_encoder.get_dense_pe(), sparse_embeddings.shape[0], dim=0)
            b, c, h, w = e5_i.shape

            e5_i = self.transformer(e5_i, pos_src, tokens)
            e5_i = self.mlp_head(e5_i)
            e5_i = e5_i.transpose(1, 2).view(b, c, h, w)

            e5_out.append(e5_i.squeeze(0))

        if densitymap is not None:
            x5 = torch.stack(e5_out, dim=0)

        x5 = self.fm(x5, self.textureconv(glcm))



        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_ResCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_ResCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_ResCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_ResCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x



