#%%
#from MixerUnet import ConvMixer_UNET_down, reparameterize_model
from ptflops import get_model_complexity_info
#from Unet import UNet
#from UNetplusplus import ResNet34UnetPlus
#from AttentionUNeT import AttentionUNet
#from ResUNetplusplus import build_resunetplusplus
#from DoubleUNet import DUNet
#from transnet import TransUNet
#from DCSAUNet import DCSAUNet
#from LeViT import Build_LeViT_UNet_128s
from MixerUnetcopy import Rep_UNET_down
from MixerUnet import reparameterize_model
#%%
model = Rep_UNET_down(2).to('cuda')
model = reparameterize_model(model).to('cuda')
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

print("RepUNet model stats")
flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, backend='pytorch',
                                        print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
print('{:<30}  {:<8}'.format('Computational complexity: ', flops))

# Calculate the size of the model
model_size = get_model_size(model)
print('{:<30}  {:<8.2f} MB'.format('Model size: ', model_size))

#%%
model = DCSAUNet(3, 2).to('cuda')
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

print("DCSAUNet model stats")
flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, backend='pytorch',
                                        print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
print('{:<30}  {:<8}'.format('Computational complexity: ', flops))

# Calculate the size of the model
model_size = get_model_size(model)
print('{:<30}  {:<8.2f} MB'.format('Model size: ', model_size))



#%%
model = TransUNet(img_dim=224,
                  in_channels=3,
                  out_channels=2,
                  head_num=4,
                  mlp_dim=512,
                  block_num=8,
                  patch_dim=16,
                  class_num=2).to('cuda')
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

print("Trans UNet model stats")
flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, backend='pytorch',
                                        print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
print('{:<30}  {:<8}'.format('Computational complexity: ', flops))

# Calculate the size of the model
model_size = get_model_size(model)
print('{:<30}  {:<8.2f} MB'.format('Model size: ', model_size))




#%%
model = DUNet().to('cuda')
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

print("DoubleUNet model stats")
flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, backend='pytorch',
                                        print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
print('{:<30}  {:<8}'.format('Computational complexity: ', flops))

# Calculate the size of the model
model_size = get_model_size(model)
print('{:<30}  {:<8.2f} MB'.format('Model size: ', model_size))


#%%
model = build_resunetplusplus().to('cuda')
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

print("UNet model stats")
flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, backend='pytorch',
                                        print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
print('{:<30}  {:<8}'.format('Computational complexity: ', flops))

# Calculate the size of the model
model_size = get_model_size(model)
print('{:<30}  {:<8.2f} MB'.format('Model size: ', model_size))

#%%
model = AttentionUNet(2, 3).to('cuda')
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

print("UNet model stats")
flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, backend='pytorch',
                                        print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
print('{:<30}  {:<8}'.format('Computational complexity: ', flops))

# Calculate the size of the model
model_size = get_model_size(model)
print('{:<30}  {:<8.2f} MB'.format('Model size: ', model_size))

#%%
model = ResNet34UnetPlus(3, 2).to('cuda')
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

print("UNet model stats")
flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, backend='pytorch',
                                        print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
print('{:<30}  {:<8}'.format('Computational complexity: ', flops))

# Calculate the size of the model
model_size = get_model_size(model)
print('{:<30}  {:<8.2f} MB'.format('Model size: ', model_size))


#%%
model = ConvMixer_UNET_down(2).to('cuda')

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, backend='pytorch',
                                        print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
print('{:<30}  {:<8}'.format('Computational complexity: ', flops))


model_size = get_model_size(model)
print('{:<30}  {:<8.2f} MB'.format('Model size: ', model_size))
# test size
model = reparameterize_model(model).to('cuda')
flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, backend='pytorch',
                                        print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
print('{:<30}  {:<8}'.format('Computational complexity: ', flops))


model_size = get_model_size(model)
print('{:<30}  {:<8.2f} MB'.format('Model size: ', model_size))
# %%
model = UNet(3, 2).to('cuda')
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

print("UNet model stats")
flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, backend='pytorch',
                                        print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
print('{:<30}  {:<8}'.format('Computational complexity: ', flops))

# Calculate the size of the model
model_size = get_model_size(model)
print('{:<30}  {:<8.2f} MB'.format('Model size: ', model_size))
# %%
