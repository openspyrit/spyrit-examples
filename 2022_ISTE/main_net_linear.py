# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 16:54:55 2021

@author: ducros
"""

#%%
from __future__ import print_function, division
import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
#import spyrit.misc.walsh_hadamard as wh
from spyrit.learning.model_Had_DCAN import *
from spyrit.learning.nets import *
from spyrit.misc.metrics import psnr, psnr_, batch_psnr

#%%
#- Acquisition
img_size = 64 # image size
batch_size = 256
M = 512  #number of measurements

#- Model and data paths
data_root = Path('./data/')
stats_root = Path('./stats_walsh/')

#- Plot options
plt.rcParams['pdf.fonttype'] = 42   # Save plot using type 1 font
plt.rcParams['text.usetex'] = True  # Latex
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)

transform = transforms.Compose(
    [transforms.functional.to_grayscale,
     transforms.Resize((img_size, img_size)),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

trainset = \
    torchvision.datasets.STL10(root=data_root, split='train+unlabeled',download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)

testset = \
    torchvision.datasets.STL10(root=data_root, split='test',download=True, transform=transform)
testloader =  torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)

dataloaders = {'train':trainloader, 'val':testloader}

#%%
inputs, _ = next(iter(dataloaders['val']))
b,c,h,w = inputs.shape

Cov = np.load(stats_root / Path("Cov_{}x{}.npy".format(img_size, img_size)))
Mean = np.load(stats_root / Path("Average_{}x{}.npy".format(img_size, img_size)))
H =  wh.walsh2_matrix(img_size)/img_size
Cov /= img_size**2

#%% STL-10 Images and network
M = 64*64//4
Ord = Cov2Var(Cov)
model = compNet(img_size, M, Mean, Cov, variant=0, H=H, Ord=Ord)
model2 = compNet(img_size, M, Mean, Cov, variant=3, H=H, Ord=Ord)

model = model.to(device)
model2 = model2.to(device)
inputs = inputs.to(device)

#%% Imgae index
ind = [1,2,3]

#%%
meas = model.forward_acquire(inputs, b, c, h, w)    # with pos/neg coefficients

# 
model_root = './models/'
title  = 'NET_free_N_64_M_1024_epo_40_lr_0.001_sss_20_sdr_0.2_bs_256_reg_1e-07'
load_net(model_root / Path(title), model2, device)

recon_mmse = model.forward_reconstruct_mmse(meas, b, c, h, w).cpu().detach().numpy()
recon_pinv = model.forward_reconstruct_pinv(meas, b, c, h, w).cpu().detach().numpy()
recon_free = model2.forward_reconstruct_mmse(meas, b, c, h, w).cpu().detach().numpy()


#%% Plot
f, axs = plt.subplots(len(ind), 4, figsize=(2*4,2*len(ind)+1),  dpi= 100)

for i_ind,v_ind in enumerate(ind): 
   
    
    img = inputs[v_ind, 0, :, :].cpu().detach().numpy()
    
    #-- Recon
    rec_mmse = recon_mmse[v_ind, 0, :, :]
    rec_pinv = recon_pinv[v_ind, 0, :, :]
    rec_free  = recon_free[v_ind, 0, :, :]
    
    #- Plot   
    axs[i_ind, 0].imshow(img, cmap='gray')
    axs[i_ind, 0].set_title("Ground-truth")
    axs[i_ind, 0].get_xaxis().set_visible(False)
    axs[i_ind, 0].get_yaxis().set_visible(False)
    axs[i_ind, 0].axis('off')
    axs[i_ind, 1].imshow(rec_free, cmap='gray')
    axs[i_ind, 1].set_title(f"Free: ${psnr_(img,rec_free):.2f}$ dB")
    axs[i_ind, 2].imshow(rec_pinv, cmap='gray')
    axs[i_ind, 2].set_title(f"Pinv: ${psnr_(img,rec_pinv):.2f}$ dB")
    axs[i_ind, 3].imshow(rec_mmse, cmap='gray')
    axs[i_ind, 3].set_title(f"MMSE: ${psnr_(img,rec_mmse):.2f}$ dB")

# remove axes
for ax in iter(axs.flatten()):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')

f.subplots_adjust(wspace=0, hspace=0)
plt.savefig("net_linear.pdf", bbox_inches=0)