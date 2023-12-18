import os
import sys
import logging
import pandas as pd
from tqdm import tqdm
from medpy.metric import dc
from collections import OrderedDict

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss


from utils.losses import DiceLoss
from data.dataset_ACDC import ACDCDataset
from utils.utils import inference, AverageMeter
from data.dataset_synapse import RandomGenerator


def trainer_acdc(config, model, optimizer, scheduler):

    train_dataset = ACDCDataset(config['root_dir'], config['ac_list_dir'], split="train", transform=transforms.Compose(
        [RandomGenerator(output_size=[config['input_h'], config['input_w']])]))
    val_dataset = ACDCDataset(config['root_dir'], config['ac_list_dir'], split="valid")
    test_dataset = ACDCDataset(config['ac_test_dir'], config['ac_list_dir'], split="test")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=config['num_workers'], shuffle=False)

    # loss Function
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(config['num_classes'])

    log = OrderedDict([
        ('epoch', []),
        ('loss', []),
        ('val', []),
    ])

    best_dsc = 0
    iterator = tqdm(range(0, config['epochs']), ncols=70)
    for epoch_num in iterator:
        model.train()
        epoch_loss = AverageMeter()

        # Train one epoch
        for i, sampled in enumerate(train_loader):

            image, label = sampled['image'], sampled['label']
            image, label = image.type(torch.FloatTensor), label.type(torch.FloatTensor)
            image = image.cuda()
            label = label.cuda()
            outputs = model(image)

            loss_dice = dice_loss(outputs, label, softmax=True)
            loss_ce = ce_loss(outputs, label[:].long())
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.update(loss.item(), n=config['batch_size'])

        val_dsc = val(model, val_loader)

        # print('epoch %d  :   loss: %.4f' % (epoch_num, epoch_loss.avg))
        log['epoch'].append(epoch_num)
        log['loss'].append(epoch_loss.avg)
        log['val'].append(val_dsc)
        pd.DataFrame(log).to_csv('results/%s/log.csv' % config['item_name'], index=False)

        if config['lr_scheduler'] == 'Cosine':
            scheduler.step()
        elif config['lr_scheduler'] == 'Reduce':
            scheduler.step(val_dsc)

        if val_dsc > best_dsc:
            torch.save(model.state_dict(), 'results/%s/Best.pth' % config['item_name'])
            best_dsc = val_dsc
            print("=> saved best model   => Best DSC: %.4f <=" % best_dsc)

        if epoch_num >= config['epochs'] - 1:
            iterator.close()
            os.makedirs(os.path.join(config['output_dir'], config['item_name'], 'acdc_test'), exist_ok=True)
            test_save_path = os.path.join(config['output_dir'], config['item_name'], 'acdc_test')
            logging.basicConfig(
                filename=config['output_dir'] + '/' + config['item_name'] + '/' + "test.txt",
                level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
            torch.save(model.state_dict(), 'results/%s/Last.pth' % config['item_name'])
            model.load_state_dict(torch.load('results/%s/Best.pth' % config['item_name']))
            inference(config, model, test_loader=test_loader, test_save_path=test_save_path)        # Last.pth
            break

    return "Training Finished!"


def val(model, val_loader):
    dc_sum = 0
    model.eval()
    for i, val_sampled_batch in enumerate(val_loader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(
            torch.FloatTensor)
        val_image_batch, val_label_batch = val_image_batch.cuda().unsqueeze(1), val_label_batch.cuda().unsqueeze(1)
        val_outputs = model(val_image_batch)
        val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)

        dc_sum += dc(val_outputs.cpu().data.numpy(), val_label_batch[:].cpu().data.numpy())
    dsc = dc_sum / len(val_loader)
    return dsc
