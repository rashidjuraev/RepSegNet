import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2


def get_training_transform():
   return A.Compose(
       [
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.25),
        A.ShiftScaleRotate(p=0.25),
        A.CoarseDropout(),
        A.RandomBrightnessContrast(p=0.25),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
        ])

def get_val_transform():
   return A.Compose(
       [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
        ])

def train_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


class ISIC_Dataset(Dataset):
    def __init__(self, data_root, img_dir="img", mask_dir="gt", mask_suffix='_segmentation.png', image_suffix='.jpg', transform=None) -> None:
        super().__init__()
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = train_aug if transform is None else transform
        if "val" in data_root:
            self.transform = val_aug
        self.mask_suffix = mask_suffix
        self.img_suffix = image_suffix
        self.img_ids = self.get_image_ids()

    def __len__(self):
        return len(self.img_ids)


    def get_image_ids(self):
        image_names = os.listdir(osp.join(self.data_root, self.img_dir))
        image_ids = [fname.split(".")[0] for fname in image_names]
        return image_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')

        return img, mask

    def __getitem__(self, index):
        img, mask = self.load_img_and_mask(index)
        img_id = self.img_ids[index]

        if self.transform:
            img, mask = self.transform(img, mask)
        mask = mask.long()
        # img = torch.from_numpy(img).permute(2, 0, 1).float()
        # mask = torch.from_numpy(mask).long()
        mask[mask==255] = 1

        results = {'img': img, 'gt_semantic_seg': mask, 'img_id': img_id}

        return results





if __name__  == '__main__':
    dataset = ISIC_Dataset("/data1/hom1/ict04/dev/datasets/isic2018/val")
    print(dataset[0]['img'].size())

    