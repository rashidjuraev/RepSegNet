import sys
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
from torch.utils.data import DataLoader
#from datasets.cvc_clinic import CVC_Dataset
from MixerUnetcopy import Rep_UNET_down, ConvMixer_UNET_down, reparameterize_model
from Unet import UNet
from DCSAUNet import DCSAUNet

# Change working directory to the directory containing this script (models)
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Add GeoSeg directory to Python path
project_root = os.path.abspath(os.path.join(current_dir, 'GeoSeg.geoseg.models'))
sys.path.append(project_root)

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


class CVC_Dataset(Dataset):
    def __init__(self, data_root, img_dir="img", mask_dir="gt", mask_suffix='.png', image_suffix='.png', transform=None) -> None:
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



dataset = CVC_Dataset("/data1/home/ict08/skinseg/skin_datasets/cvc")
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.95), len(dataset)-int(len(dataset)*0.95)], generator=torch.Generator().manual_seed(42))
#val_dataset.dataset.load_val_transform()
val_loader = DataLoader(dataset=val_dataset,
                        batch_size=1,
                        num_workers=0,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# Corrected code to load the model on the available device (GPU 0)
model = DCSAUNet()

# Correct map_location to use the first available GPU (cuda:0)
weightpath = "/data1/home/ict08/skinseg/model_weights/cvc/dscaunet_epoch100/dscaunet_epoch100.ckpt"
state_dict = torch.load(weightpath, map_location='cuda:7')  # Mapping to GPU 0
model.load_state_dict(state_dict, strict=False)

# Move the model to the correct device
model.to("cuda:7")
model.eval()

def run_model_and_store_results(dataloader, dataset_name):
    inputs_list = []
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for data in dataloader:
            inputs = data['img'].to('cuda:7')  # Move inputs to GPU 0
            labels = data['gt_semantic_seg'].to('cuda:7')  # Move labels to GPU 0
            
            outputs = model(inputs)
            inputs_list.append(inputs.cpu().numpy())  # Store the inputs
            predictions.append(outputs.cpu().numpy())
            ground_truths.append(labels.cpu().numpy())

    # Concatenate all batches together
    inputs = np.concatenate(inputs_list, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)
    
    # Save numpy arrays to disk
    np.save(os.path.join('results', f'{dataset_name}_inputs.npy'), inputs)
    np.save(os.path.join('results', f'{dataset_name}_predictions.npy'), predictions)
    np.save(os.path.join('results', f'{dataset_name}_ground_truth.npy'), ground_truths)
    
# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

# Run the function to process the dataset
run_model_and_store_results(val_loader, 'isic_dcsaunet')
