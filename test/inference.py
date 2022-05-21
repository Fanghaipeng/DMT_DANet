import sys
sys.path.insert(0,'..')
import os
import re
import sys
import shutil
import time
import numpy as np
import pickle
from apex import amp
import torch.nn as nn
from model.unet import UNet
from common.utils import Progbar, str2bool, caculate_f1iou, read_annotations_3, calculate_img_score
import torch.backends.cudnn as cudnn
from model.danet import get_danet
import torch.utils.data
from common.tools import inference_single
from common.glob import opt
import cv2
import yaml

if __name__ == '__main__':
    print("in the head of inference:", opt)
    if opt.gpu_id != -1:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)

    f = open("../config/%s.yaml" % opt.config, 'r', encoding='utf-8')
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    print(config)
    with open("../config/%s.yaml" % opt.config, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    test_dir = os.path.join(config['path']['test_dir'],opt.test_name, 'annotations','%s.txt'%opt.test_name)
    model_path = config['path']['resume_path']
    dataset_name = os.path.split(test_dir)[-1].split('.')[0]
    save_path = os.path.join(config['path']['save_dir'], dataset_name, 'pred',config['describe'])
    os.makedirs(save_path,exist_ok=True)
    new_size = int(config["test"]["resize"])

    img_types = ['0','1']
    for img_type in img_types:
        path = os.path.join(save_path, img_type)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    test_data = read_annotations_3(test_dir)
    cudnn.benchmark = False
    if "danet" in config["model_name"] or "resfcn" in config["model_name"]:
        model = get_danet(backbone='resnet50', pretrained_base=True, nclass=1)
    elif "unet" in config["describe"]:
        model = UNet(in_channels=3, n_classes=1)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in checkpoint['model_dict'].items()}, strict=True)
        model.eval()
        print("load %s finish" % (os.path.basename(model_path)))
    else:
        print("%s not exist" % model_path)
        sys.exit()

    model.cuda()
    if str2bool(config["test"]["fp16"]):
        amp.register_float_function(torch, 'sigmoid')
        model = amp.initialize(models=model, opt_level='O1', loss_scale='dynamic')
    if opt.gpu_num > 1:
        model = nn.DataParallel(model)
    model.eval()

    with torch.no_grad():
        progbar = Progbar(len(test_data), stateful_metrics=['epoch', 'config', 'lr','path'])
        f1_all, iou_all = 0., 0.
        pd_img_lab = []
        lab_all = []
        f1_all = []
        iou_all = []
        scores = []
        st_time = time.time()
        for ix, (img_path,mask_path,lab) in enumerate(test_data):
            img = cv2.imread(img_path)
            ori_size = img.shape[:2]
            img = cv2.resize(img, (new_size, new_size))
            if opt.test_save:
                img_type = str(lab)
                seg, _ = inference_single(img=img, model=model, th=0, remove=False)

                save_seg_path = os.path.join(save_path, img_type, os.path.split(img_path)[-1].split('.')[0] + '.png')
                os.makedirs(os.path.split(save_seg_path)[0],exist_ok=True)
                seg = cv2.resize(seg,(ori_size[1],ori_size[0]))
                cv2.imwrite(save_seg_path, seg.astype(np.uint8))
                progbar.add(1, values=[('path', save_seg_path), ])
            else:
                seg,pd_score = inference_single(img=img, model=model, th=0.5, remove=False)
                scores.append(pd_score)
                seg = cv2.resize(seg, (ori_size[1], ori_size[0]))
                gt = cv2.imread(mask_path, 0)
                if gt is None:
                    gt = np.zeros_like(seg)
                pd_img_lab.append(int(np.max(seg)==255))
                lab_all.append(lab)
                f1, iou = caculate_f1iou(seg, gt)
                f1_all.append(f1)
                iou_all.append(iou)
                progbar.add(1, values=[('f1', f1),])
        ed_time = time.time()


    if not opt.test_save:
        f1_all = np.array(f1_all)
        iou_all = np.array(iou_all)
        lab_all = np.array(lab_all)
        pd_img_lab = np.array(pd_img_lab)
        gt_img_lab = 1 * (lab_all > 0)

        from sklearn import metrics
        fpr, tpr, thresholds = metrics.roc_curve(np.array(gt_img_lab).reshape(-1).astype(np.int), np.array(scores).reshape(-1), pos_label = 1)
        auc = metrics.roc_auc_score(np.array(gt_img_lab).reshape(-1).astype(np.int), np.array(scores).reshape(-1))
        with open(os.path.join(save_path,'roc.pkl'),'wb') as f:
            pickle.dump({'fpr':fpr,'tpr':tpr,'th':thresholds,'auc':auc},f)
            print("roc save at %s"%(os.path.join(save_path,'roc.pkl')))

        acc, sen, spe, f1_imglevel, tp, tn, fp, fn = calculate_img_score(pd_img_lab, gt_img_lab)
        print("img level acc: %.4f sen: %.4f  spe: %.4f  f1: %.4f" % (acc, sen, spe, f1_imglevel))
        f1s_pixlevel = []
        ious_pixlevel = []

        for c in range(1, int(config["train"]["nclass"])):
            type_ix = (lab_all == c)
            if (type_ix > 0).any():
                f1_avg = float(np.mean(f1_all[type_ix]))
                iou_avg = float(np.mean(iou_all[type_ix]))
                print("%d  f1_avg: %.4f iou_avg: %.4f\n" % (c, f1_avg,iou_avg))
                f1s_pixlevel.append(f1_avg)
            else:
                print("do not exists class %d" % c)
                f1_avg = 0.0
                iou_avg = 0.0

        lsa = 0
        lsa += np.sum((iou_all[lab_all !=0]>0.5)   * (pd_img_lab[lab_all !=0] !=0 ))
        lsa += np.sum(pd_img_lab[lab_all ==0] ==0)
        lsa = lsa / lab_all.shape[0]
        print("avg : ", np.mean(f1_all[lab_all!=0]),np.mean(iou_all[lab_all!=0]))
        print('auc: %.4f, lsa: %.4f'%(auc,lsa))
        print("time used: ", ed_time - st_time, ' fps:', lab_all.shape[0] / (ed_time - st_time))
