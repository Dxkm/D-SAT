import torch.nn as nn
from models.ResNet import ResBlock
from models.Encoder import Encoder
from models.Decoder import Decoder, Up


class TCUNet(nn.Module):
    """                     D-SAT                       """
    def __init__(self, cfg, image_size, num_classes):
        super().__init__()

        # Encoder
        self.encoder = Encoder(cfg, image_size)
        # Decoder
        self.decoder = Decoder(cfg)

        # Input layer
        self.conv_pred1 = ResBlock(cfg.in_channels, cfg.embed_dim // 2)                 # 3 - 32
        self.conv_pred2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(cfg.embed_dim // 2, cfg.embed_dim // 2),
        )

        # Output layer
        self.conv_out1 = Up(cfg.embed_dim // 2, cfg.embed_dim // 2)   # 32 - 32
        self.conv_out2 = Up(cfg.embed_dim, cfg.embed_dim // 2)        # 64 - 32

        # Segmentation Head
        self.out = nn.Conv2d(cfg.embed_dim // 2, num_classes, kernel_size=1)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Input layer
        pred = x
        pred1 = self.conv_pred1(pred)       # (B, 32, 224, 224)
        pred2 = self.conv_pred2(pred1)      # (B, 32, 112, 112)

        # Encoder
        x = self.encoder(x)
        # Decoder
        x = self.decoder(x)                 # (B, 64, 56, 56)

        # Output layer
        out2 = self.conv_out2(x, pred2)     # (B, 32, 112, 112)
        out1 = self.conv_out1(out2, pred1)  # (B, 32, 224, 224)

        out = self.out(out1)
        return out
