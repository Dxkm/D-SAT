import os
import sys
import yaml
import torch
import logging
import argparse
from thop import profile
from torch.utils.data import DataLoader

from models.TCUNet import TCUNet
from utils.trainer_npy import inf
from configs import cswin_config as cfg
from data.dataset_ACDC import ACDCDataset
from data.dataset_synapse import SynapseDataset
from utils.utils import accuracy, inference, random_seed, count_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--item_name', default='ACDC', help='The item name')
    args = parser.parse_args()
    return args


def Synapse_test_loader(config):
    db_test = SynapseDataset(base_dir=config['test_path'], split="test_vol", list_dir=config['list_dir'])
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=config['num_workers'])
    return test_loader


def acdc_test_loader(config):
    db_test = ACDCDataset(config['ac_test_dir'], config['ac_list_dir'], split="test")
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=config['num_workers'])
    return test_loader


def main():
    args = parse_args()
    with open('results/%s/config.yml' % args.item_name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    random_seed(config['seed'], state=True)

    models_name = {
        'tcu-t': cfg.TCU_T(),
        'tcu-s': cfg.TCU_S(),
    }
    model = TCUNet(cfg=models_name[config['arch']], num_classes=config['num_classes'],
                   image_size=config['input_w']).cuda()

    # G FLOPs and parameter
    put = torch.zeros((1, 3, 224, 224)).cuda()
    flops, _ = profile(model, inputs=(put,))
    print("G FLOPs：", flops / 1e9)
    count_params(model)

    if config['dataset'] == 'Synapse' or config['dataset'] == 'ACDC':
        os.makedirs(os.path.join(config['output_dir'], config['item_name'], 'test'), exist_ok=True)
        test_save_path = os.path.join(config['output_dir'], config['item_name'], 'test')
        logging.basicConfig(
            filename=config['output_dir'] + '/' + config['item_name'] + '/' + "text.txt",
            level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        if config['dataset'] == 'Synapse':
            model.load_state_dict(torch.load('results/%s/epoch_249.pth' % config['item_name']), strict=False)
            inference(config, model, test_loader=Synapse_test_loader(config), test_save_path=test_save_path)
        elif config['dataset'] == 'ACDC':
            model.load_state_dict(torch.load('results/%s/Best.pth' % config['item_name']), strict=False)
            inference(config, model, test_loader=acdc_test_loader(config), test_save_path=test_save_path)
    elif config['dataset'] == 'ISIC2018':
        model.load_state_dict(torch.load('results/%s/Best.pth' % config['item_name']), strict=False)
        inf(config, model)
    else:
        model.load_state_dict(torch.load('results/%s/Best.pth' % config['item_name']), strict=False)
        accuracy(config, model)


if __name__ == '__main__':
    main()
