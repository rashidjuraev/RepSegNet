import sys
from torch.utils.data import DataLoader
from geoseg.models.MixerUnet import reparameterize_model
from geoseg.losses import *
from geoseg.datasets.isic2018_dataset import ISIC_Dataset
from geoseg.models.MixerUnet import ConvMixer_UNET_down
from geoseg.models.MixerUnetcopy import Rep_UNET_down
from catalyst import utils
from torchinfo import summary


CLASSES=('background', 'foreground')

# training hparam
max_epoch = 100
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8
lr = 5e-4
weight_decay = 0.01
backbone_lr = 5e-4
backbone_weight_decay = 0.01
accumulate_n = 1  # accumulate gradients of n batches
num_classes = len(CLASSES)
classes = CLASSES

weights_name = f"repunet{max_epoch}"
weights_path = "model_weights/isic2018/{}".format(weights_name)  # do not change
test_weights_name = "repunet"  # if save_top_k=3, there are v1,v2 model weights, i.e.xxx-v1, xxx-v2
log_name = 'isic2018/{}'.format(weights_name)  # do not change
monitor = 'val_acc'  # monitor by val_mIoU, val_F1, val_OA also supported
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
#summary(net, input_size=(train_batch_size, 3, 256, 256), depth=8)
#reparam_model = reparameterize_model(net)
#summary(reparam_model, input_size=(train_batch_size, 3, 256, 256), depth=6)
#sys.exit()
# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

use_aux_loss = False  # whether use auxiliary loss, default False

# define the dataloader

train_dataset = ISIC_Dataset("skin_datasets/isic2018/train")

val_dataset = ISIC_Dataset("skin_datasets/isic2018/val")


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


