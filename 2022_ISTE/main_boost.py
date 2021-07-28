# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 09:03:14 2021

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

#%%
#- Acquisition
img_size = 64 # image size
batch_size = 512
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

#%% Completion network
M = 64*64
np.random.seed(0)
#Ord = np.random.rand(h,w)
Ord = Cov2Var(Cov)
model = compNet(img_size, M, Mean, Cov, variant=2, H=H, Ord=Ord)
model = model.to(device)
inputs = inputs.to(device)

raw = model.forward_acquire(inputs, b, c, h, w) # with pos/neg coefficients
meas = model.forward_preprocess(raw, b, c, h, w)

#%% Recon from Walsh-ordered 2D
ind = [72,73]
sig = [0.1, 0.25, 2, 16] 
eps = np.random.standard_normal((M,))

#%% Plot
f, axs = plt.subplots(4, 5, figsize=(10,10),  dpi= 100)

for i_ind,v_ind in enumerate(ind): 
    #
    img = inputs[v_ind, 0, :, :].cpu().detach().numpy().astype(np.float32, copy=False)
    m = meas[v_ind, 0, :].cpu().detach().numpy()
    #    
    axs[2*i_ind, 0].imshow(img, cmap='gray')
    axs[2*i_ind, 0].set_title("Ground-truth")
    axs[2*i_ind, 0].get_xaxis().set_visible(False)
    axs[2*i_ind, 0].get_yaxis().set_visible(False)
    axs[2*i_ind+1, 0].axis('off')
    
    for i_sig, v_sig  in enumerate(sig):     
        #-- Recon
        m1 = m + v_sig*eps
        y1 = meas2img(m1, Ord)
        rec1 = wh.iwalsh2(y1)
        rs1 = img + np.reshape(v_sig*eps,img.shape)      
        # 
        axs[2*i_ind,  i_sig+1].imshow(rs1, cmap='gray')
        axs[2*i_ind+1,i_sig+1].imshow(rec1, cmap='gray')
        axs[2*i_ind,  i_sig+1].set_title(f"Direct: $\sigma={v_sig}$")
        axs[2*i_ind+1,i_sig+1].set_title(f"Had: $\sigma={v_sig}$")
        #
        axs[2*i_ind,i_sig+1].get_xaxis().set_visible(False)
        axs[2*i_ind,i_sig+1].get_yaxis().set_visible(False)
        axs[2*i_ind+1,i_sig+1].get_xaxis().set_visible(False)
        axs[2*i_ind+1,i_sig+1].get_yaxis().set_visible(False)

f.subplots_adjust(wspace=0, hspace=0)
plt.savefig("Hadamard_boost.pdf", bbox_inches=0)