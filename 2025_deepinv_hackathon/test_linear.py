# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 11:43:10 2025

@author: ducros
"""

import torch
import deepinv as dinv
import spyrit.core.meas as meas
import spyrit.core.noise as noise

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False # Thiss is faster 

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


#%%

import os
import torchvision
from spyrit.misc.statistics import transform_gray_norm

spyritPath = 'D:\Creatis\Programmes\openspyrit\spyrit'
imgs_path = os.path.join(spyritPath,'tutorial\images')

# Grayscale images of size (32, 32), no normalization to keep values in (0,1)
transform = transform_gray_norm(img_size=32, normalize=False)

# Create dataset and loader (expects class folder :attr:'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=7)

x, _ = next(iter(dataloader))
x = x.to(device)

print(f"Shape of input images: {x.shape}")


#%% Forward from spyrit
torch.manual_seed(0)
H = torch.randn(1000, 32*32)

norm = torch.linalg.norm(H, ord=2)
H /= norm

meas_spyrit = meas.Linear(H, 
                         meas_shape=(32,32), 
                         device=device)

y_spyrit = meas_spyrit(x)


#%% Forward from deepinverse
# System matrix
meas_deepinv = dinv.physics.LinearPhysics(meas_spyrit.forward, 
                                          img_size = (1, x.shape[-2],x.shape[-1]),
                                          ) #x.shape[-3:]


# meas_deepinv = dinv.physics.LinearPhysics(lambda y : meas_spyrit.forward(y)/norm, 
#                                           img_size = (1, x.shape[-2],x.shape[-1]),
#                                           device=device) #x.shape[-3:]


# Noise model
#meas_deepinv.noise_model = dinv.physics.GaussianNoise(sigma=0.1)
meas_deepinv.noise_model = noise.Gaussian(sigma=0.01)

y_deepinv = meas_deepinv(x)

print("diff:", torch.linalg.norm(y_spyrit - y_deepinv))
print("diff:", torch.linalg.norm(y_spyrit - y_deepinv))

#%% Pseudo inverse from deep inverse

x_pinv = meas_deepinv.A_dagger(y_deepinv)

#%% TV reconstruction

# "stepsize":1 works well for normalised forward operator 

model_tv = dinv.optim.optim_builder(
    iteration = "PGD",
    prior = dinv.optim.TVPrior(),
    data_fidelity = dinv.optim.L2(),
    params_algo = {"lambda": 1e-2, "stepsize":1}, 
    max_iter = 50,
)

x_tv, metrics_TV = model_tv(y_deepinv, meas_deepinv, compute_metrics=True, x_gt = x)

dinv.utils.plot_curves(metrics_TV)

#%% Deep Plug and Play (DPIR) 
denoiser = dinv.models.DRUNet(in_channels=1, out_channels=1, device=device)
model_dpir = dinv.optim.DPIR(device=device, denoiser=denoiser)

with torch.no_grad():
    x_dpir = model_dpir(y_deepinv, meas_deepinv)

#%% Reconstruct Anything Model (RAM)
model_ram = dinv.models.RAM(pretrained=True, device=device)

with torch.no_grad():
    x_ram = model_ram(y_deepinv, meas_deepinv)

#%% plot
from spyrit.misc.disp import imagesc

imagesc(x[1, 0, :, :].cpu(), "GT")
imagesc(x_pinv[1, 0, :, :].cpu(), "Pinv")
imagesc(x_tv[1, 0, :, :].cpu(), "TV recon")
imagesc(x_dpir[1, 0, :, :].cpu(), "DIPR recon")
imagesc(x_ram[1, 0, :, :].cpu(), "RAM recon")