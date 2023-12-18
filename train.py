import os
import yaml
import torch
import argparse
from torch.optim import lr_scheduler as lr

from models.TCUNet import TCUNet
from utils.trainer import trainer
from utils.losses import BCEDiceLoss
from configs import cswin_config as cfg
from utils.trainer_npy import trainer_npy
from utils.trainer_acdc import trainer_acdc
from utils.trainer_Synapse import trainer_synapse
from utils.utils import count_params, str2bool, random_seed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--item_name', default='test_Synapse', help='The item name')

    # dataset
    parser.add_argument('--dataset', type=str, default='Synapse',
                        choices=['ISIC2018', 'Kvasir-SEG', 'Synapse', 'ACDC'], help='dataset')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--output_dir', type=str, default='./results', help='root dir for output log')

    # Synapse dataset
    parser.add_argument('--root_path', type=str, default='../Data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--test_path', type=str, default='../Data/Synapse/test_vol_h5', help='root dir for data')
    parser.add_argument('--list_dir', type=str, default='../Data/Synapse/lists/lists_Synapse', help='list dir')

    # ACDC dataset
    parser.add_argument('--ac_list_dir', type=str, default='../Data/ACDC/lists_ACDC')
    parser.add_argument('--ac_test_dir', type=str, default='../Data/ACDC/test')

    # others dataset
    parser.add_argument('--root_dir', default='../Data', help='Dataset root dir')
    parser.add_argument('--img_ext', default='.jpg', help='image file extension')
    parser.add_argument('--mask_ext', default='.jpg', help='mask file extension')

    # Network
    parser.add_argument('--arch', type=str, default='tcu-t', choices=['tcu-t', 'tcu-s'], help='Net')
    parser.add_argument('--in_channels', type=int, default=3, help='Input channels')
    parser.add_argument('--input_w', type=int, default=224, help='image width')
    parser.add_argument('--input_h', type=int, default=224, help='image height')
    parser.add_argument('--epochs', type=int, default=200, help='Number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=8, help='Mini-batch size (default: 8)')
    parser.add_argument('--save_interval', type=int, default=5, help='evaluation epoch')

    # optimizer and scheduler
    parser.add_argument('--lr_scheduler', type=str, default='Cosine', choices=['Cosine', 'Reduce'], help='lr_scheduler')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')

    # Others
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--deep_supervision', type=str2bool, default=False, help='Deep supervision')
    parser.add_argument('--early_stopping', type=int, default=50, help='Early stopping (default: 50)')

    config = parser.parse_args()
    return config


def main():
    config = vars(parse_args())

    # Dataset choose
    config['root_dir'] = config['root_dir'] + '/' + config['dataset']
    if config['dataset'] == 'Kvasir-SEG':
        config['img_ext'] = '.jpg'
        config['mask_ext'] = '.jpg'
        config['lr_scheduler'] = 'Reduce'
    elif config['dataset'] == 'CVC-ClinicDB':
        config['img_ext'] = '.tif'
        config['mask_ext'] = '.tif'
    elif config['dataset'] == 'ISIC2018':
        config['img_ext'] = '.jpg'
        config['mask_ext'] = '.png'
        config['num_classes'] = 2
        config['lr_scheduler'] = 'Reduce'
    elif config['dataset'] == 'Synapse':
        config['num_classes'] = 9
        config['epochs'] = 250
    elif config['dataset'] == 'ACDC':
        config['num_classes'] = 4
        config['epochs'] = 100
        config['lr_scheduler'] = 'Reduce'

    # Sava config as config.yaml
    os.makedirs('results/%s' % config['item_name'], exist_ok=True)
    with open('results/%s/config.yml' % config['item_name'], 'w') as f:
        yaml.dump(config, f)

    # Set Random Seed
    random_seed(config['seed'], state=True)

    # Set Network  D-SAT
    models_name = {
        'tcu-t': cfg.TCU_T(),
        'tcu-s': cfg.TCU_S(),
    }
    model = TCUNet(cfg=models_name[config['arch']], num_classes=config['num_classes'],
                   image_size=config['input_w']).cuda()

    count_params(model)

    # Set loss function (criterion)
    criterion = BCEDiceLoss()

    # Optimization Function and lr_scheduler
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params, lr=config['lr'])

    if config['lr_scheduler'] == 'Cosine':
        scheduler = lr.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
        # Start training
        if config['dataset'] == 'Synapse':
            trainer_synapse(config, model, optimizer, scheduler)
        elif config['dataset'] == 'ACDC':
            trainer_acdc(config, model, optimizer, scheduler)
        else:
            trainer(config, model, optimizer, scheduler, criterion)
    elif config['lr_scheduler'] == 'Reduce':
        scheduler = lr.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, min_lr=config['min_lr'], verbose=True)
        # Start training
        if config['dataset'] == 'ISIC2018':
            trainer_npy(config, model, optimizer, scheduler, criterion)
        elif config['dataset'] == 'ACDC':
            trainer_acdc(config, model, optimizer, scheduler)
        else:
            trainer(config, model, optimizer, scheduler, criterion)


if __name__ == '__main__':
    main()
