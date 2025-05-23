import sys
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.segpc import SegPCDataset
from geoseg.models.deepunet.DeepUnet import Model
from catalyst.contrib.nn import Lookahead
from catalyst import utils
from torchinfo import summary


CLASSES=('background', 'foreground')

# training hparam
max_epoch = 100
ignore_index = len(CLASSES)
train_batch_size = 4
val_batch_size = 4
lr = 5e-4 
weight_decay = 0.01
backbone_lr = 5e-4 
backbone_weight_decay = 0.01
accumulate_n = 1  # accumulate gradients of n batches
num_classes = len(CLASSES)
classes = CLASSES

weights_name = f"dscaunet_epoch{max_epoch}"
weights_path = "model_weights/segpc/{}".format(weights_name)  # do not change
test_weights_name = "dscaunet_epoch"  # if save_top_k=3, there are v1,v2 model weights, i.e.xxx-v1, xxx-v2
log_name = 'segpc/{}'.format(weights_name)  # do not change
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
net = Model(n_classes=num_classes)
summary(net, input_size=(train_batch_size, 3, 256, 256))
#reparam_model = reparameterize_model(net)
#summary(reparam_model, input_size=(train_batch_size, 3, 256, 256))
sys.exit()
# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

use_aux_loss = False  # whether use auxiliary loss, default False

# define the dataloader

train_dataset = SegPCDataset("skin_datasets/segpc/train")

val_dataset = SegPCDataset("skin_datasets/segpc/val")


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
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

