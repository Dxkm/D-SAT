import torch.nn as nn
from einops.layers.torch import Rearrange


class ResNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.resnet_layers = nn.ModuleList(cfg.cnn_backbone.children())[:8]

    def forward(self, x):
        # ResNet
        for i in range(4):
            x = self.resnet_layers[i](x)
        # Stage1
        stage1_x_r = self.resnet_layers[4](x)               # (B, C, H//4, W//4)
        # Stage2
        stage2_x_r = self.resnet_layers[5](stage1_x_r)      # (B, 2C, H//8, W//8)
        # Stage3
        stage3_x_r = self.resnet_layers[6](stage2_x_r)      # (B, 4C, H//16, W//16)
        # stage4
        stage4_x_r = self.resnet_layers[7](stage3_x_r)      # (B, 8C, H//32, W//32)

        r_feature = [stage1_x_r, stage2_x_r, stage3_x_r, stage4_x_r]

        return r_feature


class ResXBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # depth_wise conv
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, groups=dim, padding_mode='reflect')
        # point_wise conv
        self.pw_conv1 = nn.Linear(dim, 4 * dim)
        self.pw_conv2 = nn.Linear(4 * dim, dim)

        self.norm = nn.InstanceNorm2d(dim)
        self.act = nn.LeakyReLU(inplace=True)

        self.r1 = Rearrange('b c h w -> b h w c')
        self.r2 = Rearrange('b h w c -> b c h w')

    def forward(self, x):
        residual = x

        x = self.dw_conv(x)
        x = self.r1(x)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = self.r2(x)
        x = self.norm(x)

        x = residual + x
        x = self.act(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.norm2 = nn.InstanceNorm2d(out_channels)

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x):
        skip = x.clone()
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)

        skip = self.skip(skip)

        out = x + skip
        out = self.act(out)
        return out
