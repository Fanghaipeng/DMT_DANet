import random
import torch
import cv2
import numpy as np
from albumentations.pytorch.functional import img_to_tensor
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(0, '..')
from common.transforms import create_train_transforms
import time
from tqdm import tqdm
import pdb


class WholeDataset(Dataset):
    def __init__(self,
                 annotations,
                 normalize={"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]},
                 transforms=None,
                 hard_aug=False,
                 resize=640,
                 balance=True,
                 mode="train",
                 nclass = 1,
                 ):
        super().__init__()
        self.normalize = normalize
        self.transforms = transforms
        self.balance = balance
        self.resize = resize
        self.mode = mode
        self.nclass = nclass
        if self.mode == "train":
            if hard_aug:
                self.aug_prob = 0.6
            else:
                self.aug_prob = 0.1
        else:
            self.aug_prob = 0
        img_base = annotations['img']
        mask_base = annotations['mask']
        self.resize = resize

        if self.balance:
            self.data_ix = [[x for x in img_base if x[1] == lab] for lab in [i for i in range(self.nclass)]]
            self.mask_ix = [[x for x in mask_base if x[1] == lab] for lab in [i for i in range(self.nclass)]]
            for ix, each in enumerate(self.data_ix):
                print("%d: %d|"%(ix,len(each)))

        else:
            self.data_ix = [[x for x in img_base]]
            self.mask_ix = [[x for x in mask_base]]
            print("data ", len(self.data_ix[0]))

    def __len__(self):
        if self.balance:
            return max([len(subset) for subset in self.data_ix])
        else:
            return len(self.data_ix[0])

    def load(self, class_ix, img_ix):
        img = cv2.resize(cv2.imread(self.data_ix[class_ix][img_ix][0]), (self.resize, self.resize))
        if self.mask_ix[class_ix][img_ix][0] is None or self.mask_ix[class_ix][img_ix][0] =='None':
            mask = np.zeros_like(img)
        else:
            mask = cv2.resize(cv2.imread(self.mask_ix[class_ix][img_ix][0]), (self.resize, self.resize))
        if random.random() < self.aug_prob and self.transforms:
            data = self.transforms(image=img, mask=mask)
            img = data["image"]
            mask = data['mask']
        img = img_to_tensor(img, self.normalize)
        mask = img_to_tensor(mask)[0].unsqueeze(0)
        mask = 1.0 * (mask > 0.5)
        return img, mask

    def __getitem__(self, index):
        if self.balance:
            imgs, masks = [], []
            for ix in range(self.nclass):
                safe_ix = index % len(self.data_ix[ix])
                img, mask = self.load(ix, safe_ix)
                imgs.append(img.unsqueeze(0))
                masks.append(mask.unsqueeze(0))
            return torch.cat(imgs), torch.cat(masks), torch.tensor([i for i in range(self.nclass)])

        else:
            _, lab = self.data_ix[0][index]
            img, mask = self.load(0, index)
            return img, mask, torch.tensor(lab)

    def reset_seed(self, epoch, seed):
        seed = (epoch + 1) * seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed_all(seed)  # gpu
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    train_root = "/data/dongchengbo/dataset/DEFACTO_train_sub2000.txt"
    with open(train_root, 'r') as f:
        train_img_list = f.readlines()
    train_img_list = [each.strip("\n") for each in train_img_list]
    train_mask_list = []
    for i in range(len(train_img_list)):
        if "Inpainting" in train_img_list[i]:
            train_mask_list.append(train_img_list[i].replace("-images", "-annotations").replace("/img", "/probe_mask"))
        else:
            train_mask_list.append(train_img_list[i].replace("-images", "-annotations").replace("/img", "/probe_mask").replace(".tif",".jpg"))
    print("len train img list: ", len(train_img_list))
    annotation = {"img": train_img_list, "mask": train_mask_list}

    data_train = WholeDataset(
        annotations=annotation,
        transforms=create_train_transforms(),
        resize=640,
        balance=False
    )

    train_data_loader = DataLoader(
        data_train,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        drop_last=True)

    print("_____")
    start = time.time()
    for epoch in (1, 10):
        train_data_loader.dataset.reset_seed(epoch, 777)
        for i, (img, mask, lab) in enumerate(tqdm(train_data_loader)):
            print(time.time() - start)
            print(img.shape)
            start = time.time()