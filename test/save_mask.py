import sys
sys.path.insert(0,'..')
import cv2
import numpy as np
from tqdm import tqdm
import os
import shutil
from common.utils import remove_small
import argparse


if __name__ == '__main__':
    img_dir = "/data/chenxinru/submit/user_data/Casiav2_small_mask/model_normal"
    save_dir = "/data/chenxinru/submit/prediction_result/Casiav2_small_mask/model_normal"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print("save dir ", save_dir)

    img_paths = []
    for root, dirs, files in os.walk(img_dir):
        for file in tqdm(files):
            if file.endswith(".npy"):
                pred = np.load(os.path.join(root, file))
                img_save_path = os.path.join(save_dir, file[:-4])
                cv2.imwrite(img_save_path, pred)
