import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from data.isic import ISIC
from utils.utils import iou_score, AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainer_npy(config, model, optimizer, scheduler, criterion):
    tr_dataloader, vl_dataloader, _ = ISIC(config)
    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        tr_loss, tr_iou = train(config, model, tr_dataloader, optimizer, criterion)
        print("=> epoch: %d   => loss: %.4f  iou: %.4f" % (epoch, tr_loss, tr_iou))
        val_loss, val_iou, val_dice = validate(config, model, vl_dataloader, criterion)
        scheduler.step(val_iou)

        trigger += 1
        if val_iou > best_iou:
            torch.save(model.state_dict(), 'results/%s/Best.pth' % config['item_name'])
            best_iou = val_iou
            print("=> saved best model   => Best IoU: %.4f <=" % best_iou)
            trigger = 0

        # early stopping
        if 0 <= config['early_stopping'] < trigger and epoch >= 100:
            print("=> early stopping")
            torch.save(model.state_dict(), 'results/%s/Last.pth' % config['item_name'])
            model.load_state_dict(torch.load('results/%s/Best.pth' % config['item_name']))
            inf(config, model)
            break

        if epoch + 1 == config['epochs']:
            print("Training Finished!")
            torch.save(model.state_dict(), 'results/%s/Last.pth' % config['item_name'])
            model.load_state_dict(torch.load('results/%s/Best.pth' % config['item_name']))
            inf(config, model)
            break

        torch.cuda.empty_cache()


def train(config, model, tr_dataloader, optimizer, criterion):
    model.train()
    tr_iterator = tqdm(enumerate(tr_dataloader))
    tr_losses = AverageMeter()
    tr_iou = AverageMeter()
    for batch, batch_data in tr_iterator:
        img = batch_data['image'].to(device)
        mask = batch_data['mask'].to(device)

        optimizer.zero_grad()
        pred = model(img)
        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()

        # evaluate by metrics
        pred_ = torch.argmax(pred, 1, keepdim=False).float()
        mask_ = torch.argmax(mask, 1)

        iou, _, _, _ = iou_score(pred_, mask_)
        tr_losses.update(loss.item(), n=config['batch_size'])
        tr_iou.update(iou, n=config['batch_size'])
    return tr_losses.avg, tr_iou.avg


def validate(config, model, val_dataloader, criterion):
    model.eval()
    losses = AverageMeter()
    val_iou = AverageMeter()
    val_dice = AverageMeter()
    with torch.no_grad():
        for batch, batch_data in enumerate(val_dataloader):
            img = batch_data['image']
            mask = batch_data['mask']

            img = img.to(device)
            mask = mask.to(device)

            pred = model(img)
            loss = criterion(pred, mask)

            pred_ = torch.argmax(pred, 1, keepdim=False).float()
            mask_ = torch.argmax(mask, 1, keepdim=False)

            iou, dice, _, _ = iou_score(pred_, mask_)
            losses.update(loss.item(), n=config['batch_size'])
            val_iou.update(iou, n=config['batch_size'])
            val_dice.update(dice, n=config['batch_size'])

    return losses.avg, val_iou.avg, val_dice.avg


def inf(config, model):
    model.eval()
    _, _, te_dataloader = ISIC(config)
    inf_iou = AverageMeter()
    inf_dice = AverageMeter()
    inf_pre = AverageMeter()
    inf_rec = AverageMeter()

    save_img_dir = f"{config['output_dir'] + '/' + config['item_name']}/output"
    if not os.path.isdir(save_img_dir):
        os.mkdir(save_img_dir)

    with torch.no_grad():
        for batch_data in tqdm(te_dataloader):
            img = batch_data['image']
            mask = batch_data['mask']
            ids = batch_data['id']

            img = img.to(device)
            mask = mask.to(device)

            pred = model(img)

            # evaluate by metrics
            pred_ = torch.argmax(pred, 1, keepdim=False).float()
            mask_ = torch.argmax(mask, 1, keepdim=False)

            iou, dice, pre, rec = iou_score(pred_, mask_)
            inf_iou.update(iou, n=config['batch_size'])
            inf_dice.update(dice, n=config['batch_size'])
            inf_pre.update(pre, n=config['batch_size'])
            inf_rec.update(rec, n=config['batch_size'])

            # save results
            txm = img.cpu().numpy()
            tbm = torch.argmax(mask, 1).cpu().numpy()
            tpm = torch.argmax(pred, 1).cpu().numpy()
            tid = ids

            for idx in range(len(tbm)):
                img = np.moveaxis(txm[idx, :3], 0, -1)
                img = np.ascontiguousarray(img * 255., dtype=np.uint8)
                gt = np.uint8(tbm[idx] * 255.)
                pred = np.where(tpm[idx] > 0.5, 255, 0)
                pred = np.ascontiguousarray(pred, dtype=np.uint8)

                res_img = skin_plot(img, gt, pred)

                fid = tid[idx]
                Image.fromarray(img).save(f"{save_img_dir}/{fid}_img.png")
                Image.fromarray(gt).save(f"{save_img_dir}/{fid}_gt.png")
                Image.fromarray(pred).save(f"{save_img_dir}/{fid}_pred.png")
                Image.fromarray(res_img).save(f"{save_img_dir}/{fid}_img_gt_pred.png")

    log_path = os.path.join('results', config['item_name'], 'test.txt')
    log = open(log_path, mode="a", encoding="utf-8")
    print('IoU: %.4f - Dice: %.4f - Pre: %.4f - Rec: %.4f' % (
        inf_iou.avg, inf_dice.avg, inf_pre.avg, inf_rec.avg), file=log)
    log.close()
    print('IoU: %.4f - Dice: %.4f - Pre: %.4f - Rec: %.4f' % (inf_iou.avg, inf_dice.avg, inf_pre.avg, inf_rec.avg))
    torch.cuda.empty_cache()


def skin_plot(img, gt, pred):
    img = np.array(img)
    gt = np.array(gt)
    pred = np.array(pred)
    edged_test = cv2.Canny(pred, 100, 255)
    contours_test, _ = cv2.findContours(edged_test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    edged_gt = cv2.Canny(gt, 100, 255)
    contours_gt, _ = cv2.findContours(edged_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt_test in contours_test:
        cv2.drawContours(img, [cnt_test], -1, (0, 0, 255), 1)
    for cnt_gt in contours_gt:
        cv2.drawContours(img, [cnt_gt], -1, (0, 255, 0), 1)
    return img
