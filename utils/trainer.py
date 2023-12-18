import torch
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

from data.dataset import dataloader
from utils.utils import iou_score, AverageMeter, accuracy


def trainer(config, model, optimizer, scheduler, criterion, random_state=42):
    train_loader, val_loader, _ = dataloader(config, random_state=random_state)

    log = OrderedDict([
        ('epoch', []),
        ('train_loss', []),
        ('train_iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    best_iou = 0
    trigger = 0

    for epoch in range(config['epochs']):
        # freeze(epoch, model, freeze_epoch=3)
        # print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train epoch
        train_log = train(config, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(val_loader, model, criterion)

        if config['lr_scheduler'] == 'Cosine':
            scheduler.step()
        elif config['lr_scheduler'] == 'Reduce':
            scheduler.step(val_log['iou'])

        # print('train_loss %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
        #       % (train_log['loss'], val_log['loss'], val_log['iou'], val_log['dice']))
        log['epoch'].append(epoch)
        log['train_loss'].append(train_log['loss'])
        log['train_iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        pd.DataFrame(log).to_csv('results/%s/log.csv' % config['item_name'], index=False)

        trigger += 1
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'results/%s/Best.pth' % config['item_name'])
            best_iou = val_log['iou']
            print("=> saved best model   => Best IoU: %.4f <=" % best_iou)
            trigger = 0

        # early stopping
        if 0 <= config['early_stopping'] < trigger and epoch >= 100:
            print("=> early stopping")
            torch.save(model.state_dict(), 'results/%s/Last.pth' % config['item_name'])
            model.load_state_dict(torch.load('results/%s/Best.pth' % config['item_name']))
            accuracy(config, model)
            break

        if epoch + 1 == config['epochs']:
            print("Training Finished!")
            torch.save(model.state_dict(), 'results/%s/Last.pth' % config['item_name'])
            model.load_state_dict(torch.load('results/%s/Best.pth' % config['item_name']))
            accuracy(config, model)
            break

        torch.cuda.empty_cache()


def train(config, train_loader, model, criterion, optimizer, epoch):
    model.train()
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    pbar = tqdm(total=len(train_loader))
    for net_input, target, _ in train_loader:
        net_input = net_input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(net_input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou, dice = iou_score(outputs[-1], target)
        else:
            output = model(net_input)
            loss = criterion(output, target)
            iou, dice, _, _ = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), net_input.size(0))
        avg_meters['iou'].update(iou, net_input.size(0))

        postfix = OrderedDict([
            ('Epoch', epoch),
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(val_loader, model, criterion):
    model.eval()
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for net_input, target, _ in val_loader:
            net_input, target = net_input.cuda(), target.cuda()

            output = model(net_input)
            loss = criterion(output, target)
            iou, dice, _, _ = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), net_input.size(0))
            avg_meters['iou'].update(iou, net_input.size(0))
            avg_meters['dice'].update(dice, net_input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])
