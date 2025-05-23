import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from geoseg.models.MixerUnet import reparameterize_model
from geoseg.losses import *
from geoseg.datasets.cvc_clinic import CVC_Dataset
#from geoseg.models.MixerUnet import ConvMixer_UNET_down
from geoseg.models.MixerUnetcopy import Rep_UNET_down
from catalyst import utils
# from catalyst.contrib.nn import Lookahead
from torchinfo import summary


CLASSES=('background', 'foreground')

# training hparam
max_epoch = 100
ignore_index = len(CLASSES)
train_batch_size = 4
val_batch_size = 4
lr = 5e-3
weight_decay = 0.01
backbone_lr = 5e-3
backbone_weight_decay = 0.01
accumulate_n = 1  # accumulate gradients of n batches
num_classes = len(CLASSES)
classes = CLASSES

weights_name = f"mixer_unet_256_epoch_5e3{max_epoch}"
weights_path = "model_weights/cvc/{}".format(weights_name)  # do not change
test_weights_name = "mixer_unet_512_epoch_5e3"  # if save_top_k=3, there are v1,v2 model weights, i.e.xxx-v1, xxx-v2
log_name = 'cvc/{}'.format(weights_name)  # do not change
monitor = 'val_OA'  # monitor by val_mIoU, val_F1, val_OA also supported
monitor_mode = 'max'  # max is better
save_top_k = 1  # save the top k model weights on the validation set
save_last = True  # save the last model weight, e.g. test_weights_name='last'
check_val_every_n_epoch = 1  # run validation every n epoch
gpus = [0]  # gpu ids, 0, 1, 2.., more setting can refer to pytorch_lightning
strategy = None  # 'dp', 'ddp', multi-gpu training can refer to pytorch_lightning
pretrained_ckpt_path = None  # more setting can refer to pytorch_lightning
resume_ckpt_path = None  # more setting can refer to pytorch_lightning

#  define the network
net = Rep_UNET_down(num_classes=num_classes)
#summary(net, input_size=(train_batch_size, 3, 256, 256))
net = reparameterize_model(net)
#summary(reparam_model, input_size=(train_batch_size, 3, 256, 256))
#sys.exit()
# define the loss
#l2_lambda = 0.001
#l2_regularization = torch.tensor(0., requires_grad=True)
#for name, param in net.named_parameters():
#    if 'bias' not in name:
#        l2_regularization += torch.norm(param, p=2)
#loss += l2_lambda * l2_regularization

loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

use_aux_loss = False  # whether use auxiliary loss, default False

# define the dataloader
dataset = CVC_Dataset("skin_datasets/cvc")
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)], generator=torch.Generator().manual_seed(42))
val_dataset.dataset.load_val_transform()

# train_dataset = ("/data1/hom1/ict04/dev/datasets/isic2018/train")

# val_dataset = ISIC_Dataset("/data1/hom1/ict04/dev/datasets/isic2018/val")


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}  # 0.1xlr for backbone
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
backbone_params = [param for name, param in net.named_parameters() if "backbone" in name]
other_params = [param for name, param in net.named_parameters() if "backbone" not in name]

# #optimizer = optim.Lookahead(optim.AdamP([
#     {"params": backbone_params, "lr": backbone_lr, "weight_decay": backbone_weight_decay},
#     {"params": other_params, "lr": lr, "weight_decay": weight_decay}
# ]))
#optimizer = optim.Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

