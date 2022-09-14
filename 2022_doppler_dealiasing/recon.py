# -*- coding: utf-8 -*-

 #%%
import torch
import numpy as np
from torchvision import datasets, transforms
from pathlib import Path
import spyrit.misc.walsh_hadamard as wh
import torch.nn as nn

# from spyrit.misc.statistics import stat_walsh_stl10
from spyrit.misc.statistics import *
from spyrit.misc.disp import *

#%%
class Dopp_Net(nn.Module):
    def __init__(self, FO, DC_layer, Denoi):
        super().__init__()
        self.FO = FO
        self.DC_layer = DC_layer # must be Tikhonov solve
        self.Denoi = Denoi

    def forward(self, x):        
        x = self.forward_tikh(x)
        x = self.Denoi(x) # shape stays the same
        x = x.view(b,c,h,w)
        
        return x

    def forward_tikh(self, x):
        # x - of shape [b,c,h,w]
        b,c,h,w = x.shape

        # Acquisition
        x = x.view(b*c,h*w) # shape x = [b*c,h*w]
        x = self.FO(x)      # shape x = [b*c,h*w]
        x = self.reconstruct_tick(x,h,w)
        
        return x

    def reconstruct_tick(self, x, h, w):
        
        # Data consistency layer
        # measurements to the image domain 
        x = self.DC_layer(x, torch.zeros_like(x), self.FO) # shape x = [b*c, N]

        # Image-to-image mapping via convolutional networks 
        # Image domain denoising 
        x = x.view(b*c,1,h,w)
        
        return x
    
    def reconstruct(self, x, h, w):
        x = self.reconstruct_tick(x, h, w)
        x = self.Denoi(x)           # shape stays the same
        x = x.view(b,c,h,w)    
        
        return x

#%%
n_x = 87
n_y = 87
bs = 10 # Batch size
data_root = '../../datasets/'


#%% A batch of STL-10 test images
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)

transform = transforms.Compose(
    [transforms.functional.to_grayscale,
      transforms.Resize((n_x, n_y)),
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5])])

testset = \
    torchvision.datasets.STL10(root=data_root, split='test',download=False, transform=transform)
testloader =  torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)

#%%
inputs, _ = next(iter(testloader))
b,c,h,w = inputs.shape

z = inputs.view(b*c,w*h)

x = z[5,:]
x = x.numpy();
imagesc(x.reshape((w,h)))


#%% data 
# Build A
A =  np.eye(w*h) -  np.eye(w*h,w*h,1)
y = A @ x

imagesc(y.reshape((w,h)))

#%%
from spyrit.restructured.Updated_Had_Dcan import *
FO = Forward_operator(A)     # forward operator

DC_layer = Tikhonov_solve(mu=0.01)
Denoi = Unet()
model = Dopp_Net(FO, DC_layer, Denoi)

# Bruit ???


#%%
m = FO(z)
imagesc(m[5,:].numpy().reshape((w,h)))


m_rec = model.forward_tikh(x)
imagesc(m_rec[5,:].numpy().reshape((w,h)))