import os
import cv2
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from medpy import metric
import SimpleITK as SiTk
from scipy.ndimage import zoom
import torch.backends.cudnn as cudnn

from data.dataset import dataloader


def iou_score(pred, target):
    smooth = 1e-5
    te = 100

    if torch.is_tensor(pred):
        pred = torch.sigmoid(pred).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    pred_ = pred > 0.5
    target_ = target > 0.5

    intersection = (pred_ & target_).sum()
    union = (pred_ | target_).sum()

    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)
    pre = (intersection + smooth) / (pred_.sum() + smooth)
    rec = (intersection + smooth) / (target_.sum() + smooth)
    return te * iou, te * dice, te * pre, te * rec


def accuracy(config, model):
    _, _, test_loader = dataloader(config)

    model.eval()
    acc_iou = AverageMeter()
    acc_dice = AverageMeter()
    acc_pre = AverageMeter()
    acc_rec = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('results', config['item_name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for result, target, meta in tqdm(test_loader, total=len(test_loader)):
            result = result.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            pred = model(result)

            iou, dice, pre, rec = iou_score(pred, target)
            acc_iou.update(iou, result.size(0))
            acc_dice.update(dice, result.size(0))
            acc_pre.update(pre, result.size(0))
            acc_rec.update(rec, result.size(0))

            pred = torch.sigmoid(pred).cpu().numpy()
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0

            for i in range(len(pred)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('results', config['item_name'], str(c), meta['img_id'][i] + '.jpg'),
                                (pred[i, c] * 255).astype('uint8'))
    # log
    log_path = os.path.join('results', config['item_name'], config['item_name'] + '_test.txt')
    log = open(log_path, mode="a+", encoding="utf-8")
    print('IoU: %.4f - Dice: %.4f - Pre: %.4f - Rec: %.4f' % (acc_iou.avg, acc_dice.avg, acc_pre.avg, acc_rec.avg),
          file=log)
    log.close()
    print('IoU: %.4f - Dice: %.4f - Pre: %.4f - Rec: %.4f' % (acc_iou.avg, acc_dice.avg, acc_pre.avg, acc_rec.avg))
    print("Testing Finished! ! !")
    torch.cuda.empty_cache()


def inference(config, model, test_loader, test_save_path=None):

    model.eval()
    metric_list = 0.0

    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(test_loader)):
            # h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = val_single_volume(image, label, model, classes=config['num_classes'],
                                         patch_size=[config['input_h'], config['input_w']],
                                         test_save_path=test_save_path, case=case_name, z_spacing=1)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
                i_batch, case_name, np.mean(metric_i, axis=0)[0] * 100, np.mean(metric_i, axis=0)[1]))

        metric_list = metric_list / len(test_loader)

        for i in range(1, config['num_classes']):
            logging.info('Mean class %d mean_dice %f mean_hd95 %f'
                         % (i, metric_list[i - 1][0] * 100, metric_list[i - 1][1]))

        performance = np.mean(metric_list, axis=0)[0] * 100
        mean_hd95 = np.mean(metric_list, axis=0)[1]

        logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))

        return "Testing Finished!"


def val_single_volume(image, label, net, classes, z_spacing=1, patch_size=None, test_save_path=None, case=None):
    if patch_size is None:
        patch_size = [256, 256]

    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            data_slice = image[ind, :, :]
            x, y = data_slice.shape[0], data_slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                data_slice = zoom(data_slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            data_input = torch.from_numpy(data_slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(data_input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        data_input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(data_input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_pre_case(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = SiTk.GetImageFromArray(image.astype(np.float32))
        prd_itk = SiTk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = SiTk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        SiTk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        SiTk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        SiTk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    return metric_list


def calculate_metric_pre_case(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def count_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return print("Total parameters count: %.2fM" % (pytorch_total_params / 1e6))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def random_seed(seed, state=True):
    if state is True:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    elif state is False:
        cudnn.deterministic = False
        cudnn.benchmark = True


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)


def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)


def dice_coe(pred, target):
    smooth = 1e-5

    pred = torch.sigmoid(pred).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (pred * target).sum()

    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
