# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:02:01 2023

@author: ducros
"""
#%% to debug needs to run
import collections
collections.Callable = collections.abc.Callable

fig_folder = './figure/'

#%% Simulate measuments
import numpy as np
import torch
from pathlib import Path

from spyrit.misc.disp import imagesc
from spyrit.misc.walsh_hadamard import walsh_matrix
from spyrit.core.prep import SplitPoisson
from spyrit.core.noise import Poisson

from recon_dev import DC1Net, Pinv1Net, Tikho1Net
from meas_dev import Hadam1Split
from statistics_dev import data_loaders_ImageNet

import matplotlib.pyplot as plt

M = 128
N = 512
alpha = 50 # in photons/pixels
b = 0
bs = 5 # batch size

# A batch of images
data_folder = '../../data/ILSVRC2012_v10102019'
dataloaders = data_loaders_ImageNet(data_folder, img_size=N, batch_size=bs)  
x, _ = next(iter(dataloaders['train']))
x = x.view(-1,N,N) 

# Raw measurement
linop_tar = Hadam1Split(M,N)
noise_tar = Poisson(linop_tar, alpha)
prep_tar  = SplitPoisson(alpha, linop_tar)

y = noise_tar(x)
m = prep_tar(y)

fig, axs = plt.subplots(1, 3, figsize=(15,7))
fig.suptitle(fr'M = {M}, N = {N}, $\alpha$ = {alpha}')

im = axs[0].imshow(x[b,:,:].cpu())
axs[0].set_title('Ground-truth')
plt.colorbar(im, ax=axs[0])

im = axs[1].imshow(y[b,:,:].cpu())
axs[1].set_title('Meas (raw)')
plt.colorbar(im, ax=axs[1])

im = axs[2].imshow(m[b,:,:].cpu())
axs[2].set_title('Meas (after prep)')
plt.colorbar(im, ax=axs[2])

plt.savefig(Path(fig_folder) / 'measurements', bbox_inches='tight', dpi=600)

#%%
norm_tag = "simu" # "expe" or "simu"
alpha_train = 10

#%% Pinv-net
from meas_dev import Hadam1Split
from spyrit.core.train import load_net
from spyrit.core.nnet import Unet

# Choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
y = y.to(device)
y = y.view(-1,1,N,2*M)
epo_pinv = 30

# Load Pinv-Net
net_prefix = f'pinv-net_unet_imagenet_ph_{alpha_train}'
net_suffix = f'N_512_M_128_epo_{epo_pinv}_lr_0.001_sss_10_sdr_0.5_bs_20_reg_1e-07'
title = './model/' + net_prefix + '_Hadam1_' + net_suffix
pinvnet = Pinv1Net(noise_tar, prep_tar, Unet())
load_net(title, pinvnet, device, False)
pinvnet.eval()
pinvnet.to(device)  # Mandantory when batchNorm is used

# Reconstruct
with torch.no_grad():
    if norm_tag == "expe":
        x_pinv, alpha_pinv = pinvnet.reconstruct_expe(y)
    else:
        x_pinv = pinvnet.reconstruct(y)
del pinvnet

#%% Tikho-Net

# covariance prior in the image domain
stat_folder = './stat/'
cov_file   = f'Cov_1_{N}x{N}.npy'
mean_file   = f'Average_1_{N}x{N}.npy'
mean  = np.load(Path(stat_folder) / mean_file)
sigma = np.load(Path(stat_folder) / cov_file)
epo_tik = 4

prep_tar.set_expe()
    
# Load
net_prefix = f'tikho-net_unet_imagenet_ph_{alpha_train}_Hadam1'
net_suffix = f'N_512_M_128_epo_{epo_tik}_lr_0.001_sss_10_sdr_0.5_bs_20_reg_1e-07'
title = './model/' + net_prefix + '_' + net_suffix
tiknet = Tikho1Net(noise_tar, prep_tar, sigma, Unet())
load_net(title, tiknet, device, False)
tiknet.eval()   # Mandantory when batchNorm is used
tiknet.to(device)  

# Reconstruction
with torch.no_grad():
    if norm_tag == "expe":
        x_tik, _  = tiknet.reconstruct_expe(y)
    else:
        x_tik  = tiknet.reconstruct(y)
del tiknet

#%% DC
# Init
H = walsh_matrix(N)
dcnet = DC1Net(noise_tar, prep_tar, H @ sigma @ H)
dcnet.eval()
dcnet.to(device)

# Reconstruction
with torch.no_grad():
    if norm_tag == "expe":
        x_dc, _  = dcnet.reconstruct_expe(y)
    else:
        x_dc  = dcnet.reconstruct(y)
del dcnet

#%%
from spyrit.misc.disp import add_colorbar
# Plot
save_tag = True
b = 0

if norm_tag == "expe":
    estimated_str = f'estimated = {alpha_pinv.cpu().numpy()[b,0,0,0]:.3}'
else:
    estimated_str = 'estimated = actual'

fig, axs = plt.subplots(1, 4, figsize=(15,7))

fig.suptitle(fr'$\alpha$ (in ph): actual = {alpha} ; train = {alpha_train} ; '+
             estimated_str)

im = axs[0].imshow(x[b,:,:].cpu())
axs[0].set_title('GT')
add_colorbar(im, 'bottom')

im = axs[1].imshow(x_dc[b,0,:,:].cpu())
axs[1].set_title(f'DC ({0} epochs)')
add_colorbar(im, 'bottom')

im = axs[2].imshow(x_pinv[b,0,:,:].cpu())
axs[2].set_title(f'Pinv-Net ({epo_pinv} epochs)')
add_colorbar(im, 'bottom')

im = axs[3].imshow(x_tik[b,0,:,:].cpu())
axs[3].set_title(f'Tikho-Net ({epo_tik} epochs)')
add_colorbar(im, 'bottom')

if save_tag:
    save_folder = Path(fig_folder)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_folder / f'meas_Htar_{alpha}_ph_{alpha_train}_norm_{norm_tag}', 
                bbox_inches='tight', dpi=600)