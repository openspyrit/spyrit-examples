# -*- coding: utf-8 -*-
"""
Created on Tue May 19 09:02:55 2026

https://spyrit.readthedocs.io/en/3.1.1/gallery/tuto_05_dcnet.html

Note: The tutorial uses the "variance" subsampling, not the square subsampling. 
It seems to work fine anyway. 

@author: ducros
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import math

# %% Download measurements from tomoradio warehouse into './data/'
# https://tomoradio-warehouse.creatis.insa-lyon.fr/#folder/6a1087225363131190d01863

# Path to the project folder
project_root = r"./data/ONE-PIX_raw_acquisition_"
suffix = '19_01_2026_16-21-39'

filename = project_root + suffix + '/spectra_' + suffix + '.npy'
spectra = np.load(filename)

filename = project_root + suffix + '/patterns_order_' + suffix + '.npy'
pattern_order = np.load(filename)

#%% Download UNet denoiser from tomoradio warehouse 
# https://tomoradio-warehouse.creatis.insa-lyon.fr/#item/67221558f03a54733161e95f
# This is the same network as is SPyRiT's tutorial 5
# https://spyrit.readthedocs.io/en/3.1.1/gallery/tuto_05_dcnet.html

model_root = r"./model/"
cnn_model_path = model_root + 'tuto5_dc-net_unet_stl10_N0_100_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07_light.pth'
print("CNN model:", cnn_model_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Mandatory as the denoiser was trained with images with values in the range 
# (-1, 1), whereas our images are in the range (0, 1).  

from spyrit.core.prep import Rerange
from spyrit.core.nnet import Unet
from typing import OrderedDict

rerange = Rerange((0, 1), (-1, 1))
denoiser = OrderedDict(
    {"rerange": rerange, "denoi": Unet(), "rerange_inv": rerange.inverse()}
)
denoiser = torch.nn.Sequential(denoiser)

from spyrit.core.train import load_net
load_net(cnn_model_path, denoiser, device, strict=False)
        
#%% Measurement, preprocessing, and reconstruction operators
# We consirer the problem of reconstructing a 64x64 image from only 1024 
# measurements. The measurements corresponds to "low frequency" Hadamard 
# patters obtained by "square" subsampling in the Hadamard domain.
from spyrit.core.meas import HadamSplit2d

# Measurement operator
Ord_rec = torch.ones((64, 64), dtype=torch.float32)

n_sub = math.ceil(32)
Ord_rec[:, n_sub:] = 0
Ord_rec[n_sub:, :] = 0

M = n_sub**2
N = 64

acqu = HadamSplit2d(N, M, Ord_rec)

# Preprocessing operator
from spyrit.core.prep import UnsplitRescaleEstim
prep = UnsplitRescaleEstim(acqu, use_fast_pinv=True)

# Pseudo-inverse reconstruction operator
from spyrit.core.recon import PinvNet
pinet = PinvNet(acqu, prep, device=device)

# Denoised completion reconstruction operator
from spyrit.core.recon import PinvNet, DCNet

# Load covariance matrix (measurement domain) from tomoradio warehouse
# https://tomoradio-warehouse.creatis.insa-lyon.fr/#item/672207cbf03a54733161e95c
# Division by 4 as covaraince was computed using images with values in the range 
# (-1, 1), whereas we consider images with values in the range (0, 1) here.  

cov_path = model_root + '/tuto5_Cov_64x64.pt' 
Cov = torch.load(cov_path, weights_only=True)
Cov /= 4 

print("Sigma : ", Cov.shape)
dcnet = DCNet(acqu, 
              prep, 
              sigma=Cov, 
              device= device,
              denoi=denoiser)

dcnet.eval()

#%% Subsample the measurements a posteriori
from spyrit.misc.sampling import reorder, Permutation_Matrix

Ord_acq = np.ones((32, 32))
Perm_acq = Permutation_Matrix (Ord_acq).T
Perm_rec = Permutation_Matrix (Ord_rec)

# 
m = reorder(spectra, Perm_acq, Perm_rec)
print("m", m.shape) 


m_torch = torch.Tensor(m[:2*M, :]).to(device)           # [2048, 1289]
m_torch = m_torch.T                                     # [1289, 2048]
m_torch = m_torch[:, None, :]      # or m_torch = torch.unsqueeze(m_torch, 1)

print("m_torch", m_torch.shape) 
    
#%% Reconstruction
with torch.no_grad():
    rec_pi =  pinet.reconstruct(m_torch)
    rec_dc =  dcnet.reconstruct_pinv(m_torch)
    rec =  dcnet.reconstruct(m_torch)

# denormalise
rec_pi = prep.alpha * rec_pi.squeeze()
rec_dc = prep.alpha * rec_dc.squeeze()
rec = prep.alpha * rec.squeeze()

rec_pi = torch.permute(rec_pi, (1, 2, 0))           # (64, 64, 1289)    
rec_dc = torch.permute(rec_dc, (1, 2, 0))           # (64, 64, 1289)    
rec = torch.permute(rec, (1, 2, 0))                 # (64, 64, 1289)

# %% Plot

# Choose spectral index (from 0 to 1288)
ind_list = [200, 600, 950]

# Show the reconstructed cube layer

fig, axs = plt.subplots(3, len(ind_list), ) #figsize=(10,5)

for i, ind in enumerate(ind_list):
    
    im = axs[0,i].imshow(rec_pi[:, :, ind].cpu(), cmap='gray')
    axs[0,i].set_title(f"Pinv {ind}")
    plt.colorbar(im, ax=axs[0,i])
    
    im = axs[1,i].imshow(rec_dc[:, :, ind].cpu(), cmap='gray')
    axs[1,i].set_title(f"DC {ind}")
    plt.colorbar(im, ax=axs[1,i])
    
    
    # Show the reconstructed cube layer
    im = axs[2,i].imshow(rec[:, :, ind].cpu(), cmap='gray')
    axs[2,i].set_title(f"DC+Unet {ind}")
    plt.colorbar(im, ax=axs[2,i])


