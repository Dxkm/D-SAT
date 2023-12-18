import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from models.CSWin import CSWinTransformer
from models.ResNet import ResBlock, ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """ Mutil-scale feature from Transformer and CNN, respectively """

    def __init__(self, cfg, image_size):
        super().__init__()

        # CSWin Transformer Feature
        self.CSWin = CSWinTransformer(img_size=image_size, in_channels=cfg.in_channels, embed_dim=cfg.embed_dim,
                                      depth=cfg.depth, split_size=cfg.split_size, num_heads=cfg.num_head)

        # Load pretrained model and delete unexpected parameters
        checkpoint = torch.load(cfg.CSWin_pretrained_path, map_location=device)['state_dict_ema']
        unexpected = ['norm.weight', 'norm.bias', 'head.weight', 'head.bias']
        for key in list(checkpoint.keys()):
            if key in unexpected:
                del checkpoint[key]
        self.CSWin.load_state_dict(checkpoint, strict=False)

        # ResNet
        self.resnet = ResNet(cfg)

        # Feature Fusion   DFA
        self.f1 = MFA(cfg.cnn_pyramid_fm[0], cfg.embed_dim, image_size // 4)
        self.f2 = MFA(cfg.cnn_pyramid_fm[1], cfg.embed_dim * 2, image_size // 8)
        self.f3 = MFA(cfg.cnn_pyramid_fm[2], cfg.embed_dim * 4, image_size // 16)
        self.f4 = MFA(cfg.cnn_pyramid_fm[3], cfg.embed_dim * 8, image_size // 32)

    def forward(self, x):
        # The feature of CNN
        stage1_x_r, stage2_x_r, stage3_x_r, stage4_x_r = self.resnet(x)
        # The feature of Transformer
        stage1_x_t, stage2_x_t, stage3_x_t, stage4_x_t = self.CSWin(x)

        # Feature Fusion
        stage1 = self.f1(stage1_x_r, stage1_x_t)  # (B,  C,  H//4, W//4)
        stage2 = self.f2(stage2_x_r, stage2_x_t)  # (B, 2C,  H//8, W//8)
        stage3 = self.f3(stage3_x_r, stage3_x_t)  # (B, 4C, H//16, W//16)
        stage4 = self.f4(stage4_x_r, stage4_x_t)  # (B, 8C, H//32, W//32)

        # Feature
        feature = [stage1, stage2, stage3, stage4]

        return feature


class MFA(nn.Module):
    """                       D F A                    """
    def __init__(self, c_channels, t_channels, image_size):
        super().__init__()

        if c_channels == t_channels:
            self.r_ch = nn.Identity()
        else:
            self.r_ch = nn.Conv2d(c_channels, t_channels, kernel_size=1, bias=False)

        self.IN = nn.InstanceNorm2d(t_channels, affine=False)

        self.t_re = Rearrange('b (h w) c -> b c h w', h=image_size, w=image_size)
        self.res = ResBlock(t_channels, t_channels)

    def forward(self, x_c, x_t):

        # CNN
        x_c = self.IN(self.r_ch(x_c))

        # Transformer
        x_t = self.t_re(x_t)

        # Feature Fusion
        out = self.res(x_c * x_t + x_t)

        return out

