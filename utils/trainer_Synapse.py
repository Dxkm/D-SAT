import os
import sys
import random
import logging
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from tensorboardX import SummaryWriter

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from utils.losses import DiceLoss
from data.dataset_synapse import SynapseDataset, RandomGenerator
from utils.utils import inference, AverageMeter


def trainer_synapse(config, model, optimizer, scheduler):
    #  tensorboard --logdir=log_path(args.output_dir + '/log')
    writer = SummaryWriter(config['output_dir'] + '/' + config['item_name'] + '/log')

    def worker_init_fn(worker_id):
        random.seed(config['seed'] + worker_id)

    # Train Dataset
    db_train = SynapseDataset(base_dir=config['root_path'], list_dir=config['list_dir'], split="train",
                              transform=transforms.Compose(
                                  [RandomGenerator(output_size=[config['input_h'], config['input_w']])]))
    train_loader = DataLoader(db_train, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], worker_init_fn=worker_init_fn)
    # print("The length of train set is: {}".format(len(db_train)))

    db_test = SynapseDataset(base_dir=config['test_path'], split="test_vol", list_dir=config['list_dir'])
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=config['num_workers'])

    # loss Function
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(config['num_classes'])

    # max_iterations = args.max_epochs * len(train_loader)
    # print("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))

    log = OrderedDict([
        ('epoch', []),
        ('loss', []),
    ])

    iter_num = 0
    iterator = tqdm(range(config['epochs']), ncols=70)
    for epoch_num in iterator:

        epoch_loss = AverageMeter()

        # Train one epoch
        for i_batch, sampled_batch in enumerate(train_loader):
            model.train()

            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.update(loss.item(), n=config['batch_size'])

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            # print('epoch %d  iteration %d   :   loss: %.4f,   loss_ce: %.4f,   loss_dice: %.4f' % (
            #     epoch_num, iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        scheduler.step()

        # print('epoch %d  :   loss: %.4f' % (epoch_num, epoch_loss.avg))

        log['epoch'].append(epoch_num)
        log['loss'].append(epoch_loss.avg)
        # pd.DataFrame(log).to_csv(config['output_dir'] + '/' + config['item_name'] + '_log.csv', index=False)
        pd.DataFrame(log).to_csv('results/%s/log.csv' % config['item_name'], index=False)

        if epoch_num > int(config['epochs'] - 10) and (epoch_num + 1) % config['save_interval'] == 0:
            filename = f'epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(config['output_dir'], config['item_name'], filename)
            torch.save(model.state_dict(), save_mode_path)
            print("save model to {}".format(save_mode_path))

        if epoch_num >= config['epochs'] - 1:
            iterator.close()

            print(f"Running Inference after epoch {epoch_num} (Last Epoch)")
            os.makedirs(os.path.join(config['output_dir'], config['item_name'], 'test'), exist_ok=True)
            test_save_path = os.path.join(config['output_dir'], config['item_name'], 'test')
            logging.basicConfig(
                filename=config['output_dir'] + '/' + config['item_name'] + '/' + f'epoch_{epoch_num}' + ".txt",
                level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
            inference(config, model, test_loader=test_loader, test_save_path=test_save_path)
            break

    writer.close()
    return "Training Finished!"
