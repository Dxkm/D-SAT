import os
import wget
import ml_collections
import torchvision.models as models

os.makedirs('../Weights', exist_ok=True)


def TCU_T():
    """  CSWin-T + resnet34  """
    cfg = ml_collections.ConfigDict()
    cfg.in_channels = 3

    # CSWin Transformer Configs
    # H × W × 3 -- H/4 × W/4 × C  --  H/8 × W/8 × 2C --  H/16 × W/16 × 4C  -- H/32 × W/32 × 8C
    cfg.image_size = 224
    cfg.embed_dim = 64
    cfg.depth = [1, 2, 21, 1]
    cfg.split_size = [1, 2, 7, 7]
    cfg.num_head = [2, 4, 8, 16]

    if not os.path.isfile('../Weights/cswin_tiny_224.pth'):
        print('Downloading CSWin-transformer model ...')
        wget.download(
            "https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_base_224.pth",
            "../Weights/cswin_tiny_224.pth")

    cfg.CSWin_pretrained_path = '../Weights/cswin_tiny_224.pth'

    # CNN Configs
    cfg.cnn_pyramid_fm = [64, 128, 256, 512]
    # cfg.cnn_backbone = models.resnet34(pretrained=True)
    cfg.cnn_backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    return cfg


def TCU_S():
    """  CSWin-B-224 + resnet34  """
    cfg = ml_collections.ConfigDict()
    cfg.in_channels = 3

    # CSWin Transformer Configs
    cfg.image_size = 224
    cfg.embed_dim = 64
    cfg.depth = [2, 4, 32, 2]
    cfg.split_size = [1, 2, 7, 7]
    cfg.num_head = [2, 4, 8, 16]

    if not os.path.isfile('../Weights/cswin_small_224.pth'):
        print('Downloading CSwin-transformer model ...')
        wget.download(
            "https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_small_224.pth",
            "../Weights/cswin_small_224.pth")

    cfg.CSWin_pretrained_path = '../Weights/cswin_small_224.pth'

    # CNN Configs
    cfg.cnn_pyramid_fm = [64, 128, 256, 512]
    # cfg.cnn_backbone = models.resnet34(pretrained=True)
    cfg.cnn_backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    return cfg
