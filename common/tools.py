import os
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from common.transforms import direct_val
import pdb
import sys
import torch.nn as nn
from apex import amp
from common.utils import Normalize_3D, UnNormalize_3D, Progbar, remove_small, \
    caculate_f1iou, str2bool, calculate_img_score

debug = 0
transform_tns = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
transform_pil = transforms.Compose([
    transforms.ToPILImage(),
])
transform_norm = Normalize_3D((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
transform_unnorm = UnNormalize_3D((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
normalize = {"mean": [0.485, 0.456, 0.406],
             "std": [0.229, 0.224, 0.225]}


def run_model(model, inputs):
    output = model(inputs)
    return output

def inference_single(img, model, th=0.25, remove=True):
    model.eval()
    with torch.no_grad():
        img = img.reshape((-1, img.shape[-3], img.shape[-2], img.shape[-1]))
        img = direct_val(img)
        img = img.cuda()
        seg = run_model(model, img)
        seg = torch.sigmoid(seg).detach().cpu()
        if torch.isnan(seg).any() or torch.isinf(seg).any():
            max_score = 0.0
        else:
            max_score = torch.max(seg).numpy()
        seg = [np.array(transform_pil(seg[i])) for i in range(len(seg))]

        if len(seg) != 1:
            pdb.set_trace()
        else:
            fake_seg = seg[0]
        if th == 0:
            return fake_seg, max_score

        fake_seg = 255.0 * (fake_seg > 255 * th)
        fake_seg = fake_seg.astype(np.uint8)
        if remove:
            fake_seg = remove_small(fake_seg)
    return fake_seg, max_score


def run_validation(val_data_loader, model, config, epoch):
    model.eval()
    nclass = int(config["train"]["nclass"])

    with torch.no_grad():
        f1_all, lab_all = [], []
        gt_img_lab, pd_img_lab = [], []
        count = 0
        progbar = Progbar(len(val_data_loader), stateful_metrics=['epoch', 'config', 'lr'])

        for ix, data in enumerate(val_data_loader):
            img, mask, lab = data
            img = img.reshape((-1, img.shape[-3], img.shape[-2], img.shape[-1]))
            mask = mask.reshape((-1, mask.shape[-3], mask.shape[-2], mask.shape[-1])).detach().cpu()
            lab = lab.reshape(-1).float()
            lab_all.extend(lab.cpu().numpy().tolist())
            img = img.cuda()

            seg = run_model(model, img)
            seg = torch.sigmoid(seg).detach().cpu()
            seg = [np.array(transform_pil(seg[i])) for i in range(len(seg))]
            mask = [np.array(transform_pil(mask[i])) for i in range(len(mask))]

            f1_batch = 0.
            gt_lab = (1.0 *(lab > 0)).cpu().numpy()
            gt_img_lab.extend(gt_lab.tolist())

            for i, (seg_, mask_) in enumerate(zip(seg, mask)):
                if seg_.shape != mask_.shape:
                    pdb.set_trace()
                if not float(config["train"]["th"]) == 0.:
                    seg_ = 255.0 * (seg_ > 255 * float(config["train"]["th"]))
                    seg_ = seg_.astype(np.uint8)
                    if str2bool(config["train"]["remove"]):
                        seg_ = remove_small(seg_)

                pd_img_lab.append(np.max(seg_))
                f1, _ = caculate_f1iou(seg_, mask_)
                f1_batch += f1
                f1_all.append(f1)
                count += 1

            f1_batch /= len(lab)
            progbar.add(1, values=[('epoch', epoch),
                                   ('f1', f1_batch),])

        f1_all = np.array(f1_all)
        lab_all = np.array(lab_all)

        auc_imglevel, sen, spe, f1_imglevel, tp, tn, fp, fn = calculate_img_score(pd_img_lab, gt_img_lab)
        print("img level  sen: %.4f  spe: %.4f  f1: %.4f  auc: %.4f" % (sen, spe, f1_imglevel, auc_imglevel))

        f1s_pixlevel = []
        for c in range(1, nclass):
            type_ix = (lab_all == c)
            if (type_ix > 0).any():
                f1_avg = float(np.mean(f1_all[type_ix]))
                print("%d  f1_avg: %.4f\n" % (c, f1_avg))
            else:
                print("do not exists class %d" % c)
                f1_avg = 0.0
            f1s_pixlevel.append(f1_avg)

    return f1s_pixlevel, f1_imglevel



def run_iter(model, data_loader, epoch, loss_funcs, config, optimizer=None, scheduler=None,writer=None):
    count = 0
    nclass = int(config["train"]["nclass"])
    loss_sum_list = [0.0 for _ in range(nclass)]
    seg_loss_func, cls_loss_func = loss_funcs

    model.train()
    progbar = Progbar(len(data_loader), stateful_metrics=['epoch', 'config', 'lr'])
    for data in data_loader:
        if optimizer is not None:
            optimizer.zero_grad()

        img, mask, lab = data
        img = img.reshape((-1, img.shape[-3], img.shape[-2], img.shape[-1]))
        mask = mask.reshape((-1, mask.shape[-3], mask.shape[-2], mask.shape[-1]))
        lab = lab.reshape(-1).float()

        # 只考虑假图
        if float(config['train']['fake_weight']) == 1:
            img = img[lab != 0]
            mask = mask[lab != 0]
            lab = lab[lab != 0]

        #网络输入输出
        img = img.cuda()
        mask = mask.cuda()
        lab = lab.cuda()
        seg = run_model(model, img)
        seg = torch.sigmoid(seg)

        # 计算分类损失
        if config['train']['fake_weight'] != 1:
            cls_pred = nn.MaxPool2d((int(config['train']['resize']), int(config['train']['resize'])))(seg).view(-1)
            cls_lab = 1.0 * (lab > 0).view(-1)
            loss_cls = cls_loss_func(cls_pred, cls_lab, cls_lab == 0, cls_lab == 1)
        else:
            loss_cls = torch.tensor(0.0)

        # 计算篡改图像的分割损失
        loss_ix = (lab > 0)
        if (loss_ix > 0).any():
            loss_seg = seg_loss_func(seg[loss_ix], mask[loss_ix])
        else:
            loss_seg = 0

        # 计算总的损失
        loss = [loss_seg,loss_cls]
        loss_total = config['train']['fake_weight'] * loss_seg + (1 - config['train']['fake_weight']) * loss_cls

        if optimizer is not None:
            if str2bool(config["train"]["fp16"]):
                with amp.scale_loss(loss_total, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_total.backward()

        progbar.add(1, values=[('epoch', epoch),
                               ('loss_total', loss_total.item()),
                               ('seg_loss', loss[0].item()),
                               ('cls_loss', loss[1].item()),
                               ])

        if writer is not None:
            writer.add_scalars(config["model_name"], {"loss_total": loss_total.item(),
                                                      "seg_loss": loss[0].item(),
                                                      "cls_loss": loss[1].item(),
                                                      }, epoch * len(data_loader) + count)

        # 优化optimizer
        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            optimizer.step()
            torch.cuda.synchronize()
            scheduler.step()

        for i in range(nclass):
            loss_sum_list[i] += loss[i].item() if not isinstance(loss[i], int) else loss[i]
        count += 1

    for i in range(nclass):
        loss_sum_list[i] /= count

    return loss_sum_list
