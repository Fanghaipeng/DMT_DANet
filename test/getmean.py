import sys
sys.path.insert(0,'..')
import cv2
import numpy as np
from tqdm import tqdm
import os
import shutil
from common.utils import remove_small
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--vote_th', type=float, default=0.3)
parser.add_argument('--avg_th', type=float, default=1.35)
parser.add_argument('--config_name',nargs='+', type=str, required=True)
opt = parser.parse_args()
print(opt)


def get_th(img, th_std):
    img_flatten = img.flatten()
    vimg = img_flatten[np.where(img_flatten > 0)]
    if not len(vimg) == 0:
        mean = np.mean(vimg)
        std = np.std(vimg)
        th = mean + float(th_std) * std
    else:
        th = 255 * 0.25
    return th


def post_process(img_paths, save_dir, remove, th_vote, th_avg):
    for img_id in tqdm(range(1,1501)):
        imgs = []
        ori_imgs = []
        for each_path in img_paths:
            x = np.load(os.path.join(each_path,"%d.npy"%img_id)) / 255.0
            img = 255 / (1 + np.exp(-(10 * x - 5)))
            ori_imgs.append(img)

            img = img.astype(np.uint8)
            th = get_th(img,th_vote)
            img = 255.0 * (img > th)
            img = img.astype(np.uint8)
            if remove:
                img = remove_small(img)
            imgs.append(img)

        avg = np.mean(ori_imgs,axis=0)
        avg = avg.astype(np.uint8)
        th = get_th(avg, th_avg)
        avg = 255.0 * (avg > th)
        avg = avg.astype(np.uint8)

        imgs.append(avg)
        if len(img_paths) % 2 == 0:
            imgs.append(imgs[0])

        final = np.sum(imgs,axis=0)
        final = 255 * (final > (0.5 * len(imgs)*255))

        img_save_path = os.path.join(save_dir,"%d.png"%img_id)
        cv2.imwrite(img_save_path, final)
    return 0


if __name__ == '__main__':
    save_dir = '../../prediction_result/images'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir,exist_ok=True)
    print("save dir ", save_dir)

    config_names = [each for each in opt.config_name]
    img_paths = ['../../user_data/temp_npy/%s'%each for each in config_names]

    post_process(img_paths, save_dir, True, opt.vote_th, opt.avg_th)
    # for each in img_paths:
    #     shutil.rmtree(each)
