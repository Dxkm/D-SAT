import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from models.ResNet import ResBlock, ResXBlock


class Decoder(nn.Module):
    """  Decoder """

    def __init__(self, cfg):
        super().__init__()

        self.up3 = Up(cfg.embed_dim * 8, cfg.embed_dim * 4)  # 512 - 256
        self.up2 = Up(cfg.embed_dim * 4, cfg.embed_dim * 2)  # 256 - 128
        self.up1 = Up(cfg.embed_dim * 2, cfg.embed_dim)      # 128 - 64

    def forward(self, x):
        # x = [(B, 64, 56, 56) (B, 128, 28, 28) (B, 256, 14, 14) (B, 512, 7, 7)]
        stage1, stage2, stage3, stage4 = x
        de3 = self.up3(stage4, stage3)  # (B, embed_dim * 4 , H // 16, W // 16)   256 14 14
        de2 = self.up2(de3, stage2)     # (B, embed_dim * 2 , H // 8 , W // 8)    128 28 28
        de1 = self.up1(de2, stage1)     # (B, embed_dim     , H // 4 , W // 4)    64 56 56

        return de1


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # (B, C, H, W)  --->  (B, C/2, 2H, 2W)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            ResXBlock(out_channels),
        )

        self.CCA = CA(out_channels)
        self.conv = ResBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x, skip = self.CCA(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class CA(nn.Module):
    """                        F D G A                    """

    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Linear(dim, 3 * dim)
        self.linear = nn.Linear(dim, dim)
        self.r1 = Rearrange('b c h w -> b h w c')
        self.r2 = Rearrange('b h w c -> b c h w')
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x, skip):
        b, c, h, w = x.shape
        x = self.r1(x)
        skip = self.r1(skip)

        gate = self.gate(x).reshape(b, h, w, 3, c).permute(3, 0, 1, 2, 4)
        g1, g2, g3 = gate[0], gate[1], gate[2]
        skip = torch.sigmoid(self.linear(g1 + skip)) * skip + torch.sigmoid(g2) * torch.tanh(g3)
        skip = self.relu(skip)

        x = self.r2(x)
        skip = self.r2(skip)
        return x, skip
