import sys
sys.path.insert(0,'..')
import os
import re
import json
import yaml
import time
import shutil
import random
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np

from apex import amp
from common.transforms import create_train_transforms
from common.utils import read_annotations_3
from common.tools import run_iter, run_validation

from model.MantraNet import MantraNet
from model.danet import get_danet, get_danet_BayerNoise, get_danet_SRMNoise
from model.unet import UNet
from model.fcn import FCN16s, FCN8s
from common.glob import opt
from train.dataset import WholeDataset
from train.schedulers import create_optimizer,default_scheduler
from train.loss import DiceLoss, WeightBceLoss

torch.backends.cudnn.benchmark = True
def str2bool(in_str):
    if in_str in [1, "1", "t", "True", "true"]:
        return True
    elif in_str in [0, "0", "f", "False", "false", "none"]:
        return False

if __name__ == "__main__":
    print(":in the head of train", opt)
    # load configs
    if not os.path.exists("../config/%s.yaml" % opt.config):
        print("../config/%s.yaml not found." % opt.config)
        exit()
    f = open("../config/%s.yaml" % opt.config, 'r', encoding='utf-8')
    config = yaml.load(f.read())
    config["model_name"] = opt.config
    print("configs",config)

    # gpu setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    if opt.gpu_id != -1:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)


    # random seed setting
    if opt.manualSeed == -1:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.benchmark = True

    # save dirs: model,params,writer
    st_time = time.strftime("%m%d%H%M", time.localtime())
    model_savedir = os.path.join(config["train"]["outf"], st_time+config["model_name"])
    os.makedirs(model_savedir, exist_ok=True)
    params = vars(opt)
    params_file = os.path.join(model_savedir, 'params.json')
    with open(params_file, 'w') as fp:
        json.dump(params, fp, indent=4)
    writer_dir = os.path.join(config["train"]["writer_root"], st_time+config["model_name"])
    if os.path.exists(writer_dir):
        shutil.rmtree(writer_dir, ignore_errors=True)
    os.makedirs(writer_dir, exist_ok=True)
    writer = SummaryWriter(logdir=writer_dir)

    # init training params
    board_num = 0
    start_epoch = 0

    # model define
    if "mantra" in config["model_name"]:
        model = MantraNet()
    elif "fcn16s" in config["model_name"]:
        model = FCN16s(nclass=1)
    elif "fcn8s" in config["model_name"]:
        model = FCN8s(nclass=1, pretrained_base=True)
    elif "danet" or "resfcn" in config["model_name"]:
        if "BayarNoise" in config["model_name"]:
            model = get_danet_BayerNoise(backbone='resnet50', pretrained_base=True, nclass=1,
                          n_input=int(config["train"]["n_input"]))
        elif "SRMNoise" in config["model_name"]:
            model = get_danet_SRMNoise(backbone='resnet50', pretrained_base=True, nclass=1,
                          n_input=int(config["train"]["n_input"]))
        else:
            model = get_danet(backbone='resnet50', pretrained_base=True, nclass=1,
                          n_input=int(config["train"]["n_input"]))
        
    elif "unet" in config["model_name"]:
        model = UNet(in_channels=3, n_classes=1)
    else:
        model = None
        print("Model %s not found " % config["model_name"])
        exit()

    # loss function
    loss_fns = [DiceLoss(smooth=1).cuda(),
                WeightBceLoss(weight=3).cuda()]

    # dataset: load data from annotations
    if str2bool(config["train"]["aug"]):
        trans = create_train_transforms()
    else:
        trans = None
    val_img_list = read_annotations_3(config["train"]["val_path"])
    annotation_val = {"img": [(each[0], each[2]) for each in val_img_list],
                      "mask": [(each[1], each[2]) for each in val_img_list]}

    train_img_list = read_annotations_3(config["train"]["train_path"])
    annotation_train = {"img": [(each[0], each[2]) for each in train_img_list],
                        "mask": [(each[1], each[2]) for each in train_img_list]}
    print("len train img list: ", len(train_img_list))
    print("len val img list: ", len(val_img_list))

    # dataloader setting
    nclass = int(config['train']['nclass'])
    data_train = WholeDataset(
        annotations=annotation_train,
        transforms=create_train_transforms(),
        hard_aug=str2bool(config['train']['hard_aug']),
        resize=int(config['train']['resize']),
        balance=str2bool(config['train']['balance']),
        mode="train",
        nclass=int(config['train']['nclass'])
    )

    train_data_loader = DataLoader(
        data_train,
        batch_size=int(config["train"]["batchSize"]),
        num_workers=opt.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=False)

    data_val = WholeDataset(
        annotations=annotation_val,
        resize=int(config['train']['resize']),
        balance=False,
        mode="val",
    )

    val_data_loader = DataLoader(
        data_val,
        batch_size= int(config['train']['nclass']) * int(config["train"]["batchSize"]),
        num_workers=opt.workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False)

    print("train_set size: %d,%d | val_set size: %d,%d"
          % (len(data_train), len(train_data_loader), len(data_val), len(val_data_loader)))

    # scheduler setting
    default_scheduler["batch_size"] = int(config["train"]["batchSize"])
    default_scheduler["learning_rate"] = float(config["train"]["lr"])
    default_scheduler["schedule"]['params']['max_iter'] = len(train_data_loader)
    optimizer, scheduler = create_optimizer(optimizer_config=default_scheduler, model=model)
    max_iters = default_scheduler["schedule"]['params']['max_iter']

    # load from resume
    if str2bool(config["train"]["resume"]):
        checkpoint = torch.load(config["path"]["resume_path"], map_location='cpu')
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in checkpoint['model_dict'].items()})
        model.train(mode=True)
        print("load %s finish" % (os.path.basename(config["path"]["resume_path"])))

    # send data to cuda
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model.cuda()
    if str2bool(config["train"]["fp16"]):
        # assert opt.gpu_num == torch.cuda.device_count()
        amp.register_float_function(torch, 'sigmoid')
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', loss_scale='dynamic')
    if opt.gpu_num > 1 and opt.gpu_id == -1:
        model = nn.DataParallel(model)
    elif opt.gpu_num > 1 and opt.gpu_id != -1:
        print("gpu_num and gpu_id not correct")
        sys.exit()

    model.train()

    # start training
    best_f1, best_iou, best_score = 0, 0, 0
    for epoch in range(start_epoch, int(config["train"]["niter"])):
        train_data_loader.dataset.reset_seed(epoch, 777)
        loss_sum_list = \
            run_iter(model, train_data_loader, epoch, config=config,
                     loss_funcs=loss_fns,
                     optimizer=optimizer, scheduler=scheduler,writer=writer)

        loss_seg, loss_cls = loss_sum_list

        if str2bool(config["train"]["step_save"]) and epoch % int(config["train"]["save_step"]) == 0:
            torch.save({
                'epoch': epoch,
                'model_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            }, os.path.join(model_savedir, 'model_%d.pt' % epoch))

        # start evaluation
        model.eval()
        with torch.no_grad():
            f1_pixlevels, f1_imglevel = run_validation(val_data_loader, model, config, epoch)
            val_score = np.mean(f1_pixlevels) + f1_imglevel
            if val_score > best_score:
                best_score = val_score
                torch.save({
                    'epoch': epoch,
                    'model_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'best_score': best_score,
                }, os.path.join(model_savedir, 'model_best.pt'))

            writer.add_scalars(config["model_name"], {"score": val_score, "bst_s": best_score}, epoch)
            print('[Epoch %d] Train loss_seg: %.4f  loss_cls: %.4f | f1_pix: %.4f  f1_img: %.4f '
                  '| val_score:%.4f  bst:%.4f'
                  % (epoch, loss_seg, loss_cls,
                     np.mean(f1_pixlevels), f1_imglevel, val_score, best_score))

        model.train()

