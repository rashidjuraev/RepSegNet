import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy
from torchsummary import summary


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class DepthWiseConv(nn.Module):
    def __init__(self, inc, kernel_size, stride=1, padding=1):
        super().__init__()
        if kernel_size == 1:
            padding = 0
        # self.conv = nn.Sequential(
        #     nn.Conv2d(inc, inc, kernel_size, stride, padding, groups=inc, bias=False,),
        #     nn.BatchNorm2d(inc),
        # )
        self.conv = conv_bn(inc, inc,kernel_size, stride, padding, inc)

    def forward(self, x):
        return self.conv(x)
    

class PointWiseConv(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(inc, outc, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(outc),
        # )
        self.conv = conv_bn(inc, outc, 1, 1, 0)
    def forward(self, x):
        return self.conv(x)

class MobileOneBlock(nn.Module):

    def __init__(self, in_channels, out_channels, k=3,
                 stride=1, dilation=1, padding_mode='zeros', deploy=False, use_se=False):
        super(MobileOneBlock, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = deploy
        kernel_size = 3  # modelda kernel o'zgartirish joyi
        padding = 1
        assert kernel_size == 3
        assert padding == 1
        self.k = k
        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            # self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
            ...
        else:
            self.se = nn.Identity()

        if deploy:
            self.dw_reparam = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=in_channels, bias=True, padding_mode=padding_mode)
            self.pw_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=True)

        else:
            # self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            # self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            # self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            # print('RepVGG Block, identity = ', self.rbr_identity)
            self.dw_bn_layer = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            for k_idx in range(k):
                setattr(self, f'dw_3x3_{k_idx}', 
                    DepthWiseConv(in_channels, kernel_size, stride=stride, padding=padding)
                )
            self.dw_1x1 = DepthWiseConv(in_channels, 1, stride=stride)

            self.pw_bn_layer = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            for k_idx in range(k):
                setattr(self, f'pw_1x1_{k_idx}', 
                    PointWiseConv(in_channels, out_channels)
                )


    def forward(self, inputs):
        if self.deploy:
            x = self.dw_reparam(inputs)
            x = self.nonlinearity(x)
            x = self.pw_reparam(x)
            x = self.nonlinearity(x)
            return x

        if self.dw_bn_layer is None:
            id_out = 0
        else:
            id_out = self.dw_bn_layer(inputs)
        
        x_conv_3x3 = []
        for k_idx in range(self.k):
            x = getattr(self, f'dw_3x3_{k_idx}')(inputs)
            # print(x.shape)
            x_conv_3x3.append(x)
        x_conv_1x1 = self.dw_1x1(inputs)
        # print(x_conv_1x1.shape, x_conv_3x3[0].shape)
        # print(x_conv_1x1.shape)
        # print(id_out)
        x = id_out + x_conv_1x1 + sum(x_conv_3x3)
        x = self.nonlinearity(self.se(x))

         # 1x1 conv
        if self.pw_bn_layer is None:
            id_out = 0
        else:
            id_out = self.pw_bn_layer(x)
        x_conv_1x1 = []
        for k_idx in range(self.k):
            x_conv_1x1.append(getattr(self, f'pw_1x1_{k_idx}')(x))
        x = id_out + sum(x_conv_1x1)
        x = self.nonlinearity(x)
        return x


    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        # K3 = self.rbr_dense.conv.weight
        # K1 = self.rbr_1x1.conv.weight
        # t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        # t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        # l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        # eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        # l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        # return l2_loss_eq_kernel + l2_loss_circle
        ...



#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        # kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        # kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        # kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        # return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

        dw_kernel_3x3 = []
        dw_bias_3x3 = []
        for k_idx in range(self.k):
            k3, b3 = self._fuse_bn_tensor(getattr(self, f"dw_3x3_{k_idx}").conv)
            # print(k3.shape, b3.shape)
            dw_kernel_3x3.append(k3)
            dw_bias_3x3.append(b3)
        dw_kernel_1x1, dw_bias_1x1 = self._fuse_bn_tensor(self.dw_1x1.conv)
        dw_kernel_id, dw_bias_id = self._fuse_bn_tensor(self.dw_bn_layer, self.in_channels)
        dw_kernel = sum(dw_kernel_3x3) + self._pad_1x1_to_3x3_tensor(dw_kernel_1x1) + dw_kernel_id
        dw_bias = sum(dw_bias_3x3) + dw_bias_1x1 + dw_bias_id
        # pw
        pw_kernel = []
        pw_bias = []
        for k_idx in range(self.k):
            k1, b1 = self._fuse_bn_tensor(getattr(self, f"pw_1x1_{k_idx}").conv)
            # print(k1.shape)
            pw_kernel.append(k1)
            pw_bias.append(b1)
        pw_kernel_id, pw_bias_id = self._fuse_bn_tensor(self.pw_bn_layer, 1)

        pw_kernel_1x1 = sum(pw_kernel) + pw_kernel_id
        pw_bias_1x1 = sum(pw_bias) + pw_bias_id
        return dw_kernel, dw_bias, pw_kernel_1x1, pw_bias_1x1


        

        

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch, groups=None):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            bias = branch.conv.bias
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            # if not hasattr(self, 'id_tensor'):
            input_dim = self.in_channels // groups # self.groups
            if groups == 1:
                ks = 1
            else:
                ks = 3
            kernel_value = np.zeros((self.in_channels, input_dim, ks, ks), dtype=np.float32)
            for i in range(self.in_channels):
                if ks == 1:
                    kernel_value[i, i % input_dim, 0, 0] = 1
                else:
                    kernel_value[i, i % input_dim, 1, 1] = 1
            self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)

            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        dw_kernel, dw_bias, pw_kernel, pw_bias = self.get_equivalent_kernel_bias()

        self.dw_reparam = nn.Conv2d(
            in_channels=self.pw_1x1_0.conv.conv.in_channels, 
            out_channels=self.pw_1x1_0.conv.conv.in_channels,                              
            kernel_size=self.dw_3x3_0.conv.conv.kernel_size, 
            stride=self.dw_3x3_0.conv.conv.stride,
            padding=self.dw_3x3_0.conv.conv.padding, 
            groups=self.dw_3x3_0.conv.conv.in_channels, 
            bias=True, 
            )
        self.pw_reparam = nn.Conv2d(
            in_channels=self.pw_1x1_0.conv.conv.in_channels,
            out_channels=self.pw_1x1_0.conv.conv.out_channels, 
            kernel_size=1, 
            stride=1, 
            bias=True
            )

        self.dw_reparam.weight.data = dw_kernel
        self.dw_reparam.bias.data = dw_bias
        self.pw_reparam.weight.data = pw_kernel
        self.pw_reparam.bias.data = pw_bias

        for para in self.parameters():
            para.detach_()
        self.__delattr__('dw_1x1')
        for k_idx in range(self.k):
            self.__delattr__(f'dw_3x3_{k_idx}')
            self.__delattr__(f'pw_1x1_{k_idx}')
        if hasattr(self, 'dw_bn_layer'):
            self.__delattr__('dw_bn_layer')
        if hasattr(self, 'pw_bn_layer'):
            self.__delattr__('pw_bn_layer')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True
        
     


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer_Block(in_dim, out_dim, kernel_size=3, padding=1):
    return nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=kernel_size, groups=in_dim, padding=padding),
                nn.BatchNorm2d(in_dim),
                nn.ReLU()
            )),
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
        nn.BatchNorm2d(out_dim),    
        nn.ReLU()
    )


class DepthWiseInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, int(out_channels/4), kernel_size=1)
        self.conv3_1 = MobileOneBlock(in_channels, int(out_channels/4))
        self.conv3_2 = MobileOneBlock(out_channels//4, int(out_channels/4))
        self.conv3_3 = MobileOneBlock(out_channels//4, int(out_channels/4))

        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv3_1(x)
        c = self.conv3_2(b)
        d = self.conv3_3(c)

        cat = torch.cat([a, b, c, d], dim=1)
        x = self.batchnorm(cat)
        out = self.relu(cat)
       

        return out
#######################################



class ConvMixer_UNET_down(nn.Module):
    def __init__(self, num_classes):
        super(ConvMixer_UNET_down, self).__init__()
        self.num_classes = num_classes
        # Encoder Stage 1
        self.E_S1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  DepthWiseInceptionBlock(64, 64))

        self.E_S2 = nn.Sequential(DepthWiseInceptionBlock(64, 128),
                                  DepthWiseInceptionBlock(128, 128))

        self.E_S3 = nn.Sequential(DepthWiseInceptionBlock(128, 256),
                                  DepthWiseInceptionBlock(256, 256))

        self.E_S4 = nn.Sequential(DepthWiseInceptionBlock(256, 512),
                                  DepthWiseInceptionBlock(512, 512))

        self.bridge = nn.Sequential(DepthWiseInceptionBlock(512, 1024),
                                    DepthWiseInceptionBlock(1024, 512))
        
        
        self.D_16 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                 nn.UpsamplingBilinear2d(scale_factor=2))
        
        
        self.D_8_1 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU())
        
        self.D_8_2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                  nn.UpsamplingBilinear2d(scale_factor=2))
        
        
        self.D_4_1 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU())
        
        self.D_4_2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),
                                  nn.UpsamplingBilinear2d(scale_factor=2))
        
        
        self.D_2_1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU())
        
        self.D_2_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),
                                  nn.UpsamplingBilinear2d(scale_factor=2),)

        self.D_1_1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU())

        self.D_1_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU(),
                                  nn.Conv2d(in_channels=8, out_channels=num_classes, kernel_size=1))

        self.max_pool = nn.MaxPool2d(2, 2)
        
        
    def forward(self, x):
        e_s1_out = self.E_S1(x) # out = 64, 256, 256
        x1 = self.max_pool(e_s1_out) # out = 64, 128, 128

        e_s2_out = self.E_S2(x1) # out = 128, 128, 128
        x2 = self.max_pool(e_s2_out) # out = 128, 64, 64

        e_s3_out = self.E_S3(x2) # out = 256, 64, 64
        x3 = self.max_pool(e_s3_out) # out = 256, 32, 32
        
        e_s4_out = self.E_S4(x3) # out = 512, 32, 32
        x4 = self.max_pool(e_s4_out) # out = 512, 16, 16
        
        bridge = self.bridge(x4) # out = 512, 16, 16
        
        
        d_16 = self.D_16(bridge)
        
        d_8 = self.D_8_2(self.D_8_1(x3) + d_16)
       
        d_4 = self.D_4_2(torch.cat((self.D_4_1(x2), d_8), dim=1))
        
        d_2 = self.D_2_2(torch.cat((self.D_2_1(x1), d_4), dim=1))

        d_1 = self.D_1_2(torch.cat((self.D_1_1(e_s1_out), d_2), dim=1))

        
        out = d_1


        return out

def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.
    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    return model

if __name__ == '__main__':
    model = ConvMixer_UNET_down(2).to('cuda')
    #summary(model, input_size=(3, 256, 256), device='cuda')
    
    #summary(model, input_size=(3, 256, 256), device='cuda')

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, backend='pytorch',
                                            print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    
    model = reparameterize_model(model).to('cuda')
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, backend='pytorch',
                                            print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))