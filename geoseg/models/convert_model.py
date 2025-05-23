#%%
from MixerUnetcopy import Rep_UNET_down, reparameterize_model
import torch

path = "/data1/home/ict08/skinseg/model_weights/segpc/repunet100/repunet100.ckpt"
save_path = "/data1/home/ict08/skinseg/rep_weights/segpc/repunet.ckpt"

a = torch.load(path)
class TPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Rep_UNET_down(2)
    def forward(self, x):
        return self.net(x)
model = TPModel()
model.load_state_dict(a["state_dict"])
model.to("cuda:7")
model.eval()
rep_model = reparameterize_model(model)
a["state_dict"] = rep_model.state_dict()
torch.save(a, save_path)
# %%
