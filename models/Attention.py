import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from models.ResNet import ResXBlock


class SSA(nn.Module):
    """                        S S A                    """
    def __init__(self, dim, h, w, bias=False):
        super().__init__()

        self.re3_4 = Rearrange('b (h w) c -> b c h w', h=h, w=w)

        self.blk1 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            ResXBlock(dim // 4),
        )

        self.strip = Strip(dim // 4, bias=bias)

        self.blk2 = nn.Sequential(
            nn.Conv2d(dim // 4, dim, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(inplace=True),
        )

        self.re4_3 = Rearrange('b c h w -> b (h w) c', h=h, w=w)

    def forward(self, x):
        x = self.re3_4(x)

        x = self.blk1(x)
        x = self.strip(x)
        x = self.blk2(x)

        x = self.re4_3(x)
        return x


class Strip(nn.Module):
    def __init__(self, dim, bias=False, mode='bilinear'):
        super().__init__()

        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        self.mode = mode

        # 1D conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 3), 1, (0, 1), bias=bias),
            nn.InstanceNorm2d(dim),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, dim, (3, 1), 1, (1, 0), bias=bias),
            nn.InstanceNorm2d(dim),
        )
        # Fusion
        self.conv4 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=bias),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(inplace=True),
        )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = F.interpolate(self.conv2(self.pool_w(x)), (h, w), mode=self.mode, align_corners=True)
        x2 = F.interpolate(self.conv3(self.pool_h(x)), (h, w), mode=self.mode, align_corners=True)
        out = self.conv4(self.relu(x1 + x2))                       # Fusion
        return self.relu(torch.matmul(x, out))                     # V2  best
