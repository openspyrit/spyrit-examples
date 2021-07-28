# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 18:53:39 2021

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
import spyrit.misc.walsh_hadamard as wh
from spyrit.learning.model_Had_DCAN import *

#%%
#- Acquisition
img_size = 64 # image size
batch_size = 1024

#- Model and data paths
data_root = Path('./data/')
stat_root = Path('./stats_walsh/')

#- Save plot using type 1 font
plt.rcParams['pdf.fonttype'] = 42

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

#%% Walsh ordered 2D on STL-10
import time
time_start = time.perf_counter()
stat_walsh(dataloaders['train'], device, stat_root)
time_elapsed = (time.perf_counter() - time_start)
print(time_elapsed)

#%%
print('Loading Cov') 
stats_root = Path('./stats_ordered/')
Cov_np = np.load(stats_root / Path("Cov_{}x{}.npy".format(img_size, img_size)))
stats_root = Path('./stats_walsh/')
Cov_torch = np.load(stats_root / Path("Cov_{}x{}.npy".format(img_size, img_size)))

var_np = Cov2Var(Cov_np)
var_torch = Cov2Var(Cov_torch)
err = var_np - var_torch

err_cov = np.linalg.norm(Cov_np-Cov_torch)/np.linalg.norm(Cov_np)
err_var = np.linalg.norm(var_np-var_torch)/np.linalg.norm(var_np)

print(f"Cov error: {err_cov}")
print(f"Var error: {err_var}")

f, axs = plt.subplots(1,2)
axs[0].imshow(np.divide(var_np,var_torch), cmap='gray')
axs[0].set_title(f"Var ratio: numpy / pytorch")
axs[1].imshow(var_np-var_torch, cmap='gray'); 
axs[1].set_title(f"Var diff: numpy - pytorch")

f, axs = plt.subplots(1,2)
axs[0].imshow(np.divide(Cov_np, Cov_torch), cmap='gray')
axs[0].set_title(f"Cov ratio: numpy / pytorch")
axs[1].imshow(Cov_np-Cov_torch, cmap='gray'); 
axs[1].set_title(f"Cov diff: numpy - pytorch")

#%%
eta1 = .5
eta2 = .25
mask1 = Variance_mask(Cov_np,eta1)
mask2 = Variance_mask(Cov_np,eta2)

f, axs = plt.subplots(1, 3, figsize=(12,8),  dpi= 100)
f.suptitle('numpy')
axs[0].imshow(np.log(var_np), cmap='gray'); 
axs[1].imshow(mask1, cmap='gray');
axs[2].imshow(mask2, cmap='gray');
axs[0].set_title("variance")
axs[1].set_title(f"mask {eta1}")
axs[2].set_title(f"mask {eta2}")

eta1 = .5
eta2 = .25
mask1 = Variance_mask(Cov_torch,eta1)
mask2 = Variance_mask(Cov_torch,eta2)

f, axs = plt.subplots(1, 3, figsize=(12,8),  dpi= 100)
f.suptitle('pytorch')
axs[0].imshow(np.log(var_torch), cmap='gray'); 
axs[1].imshow(mask1, cmap='gray');
axs[2].imshow(mask2, cmap='gray');
axs[0].set_title("variance")
axs[1].set_title(f"mask {eta1}")
axs[2].set_title(f"mask {eta2}")