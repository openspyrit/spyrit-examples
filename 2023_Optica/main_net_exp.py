# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:31:15 2023

@author: ducros
"""
#%% to debug needs to run
import collections
collections.Callable = collections.abc.Callable

fig_folder = './figure/'


#%% Load experimental measurement matrix (after split compensation)
from pathlib import Path
from spyrit.misc.disp import imagesc, add_colorbar
import matplotlib.pyplot as plt
from spyrit.misc.walsh_hadamard import walsh_matrix
import numpy as np
from PIL import Image

M = 128
N = 512
mat_folder = r'.\data\2023_03_13_2023_03_14_eGFP_DsRed_3D\Reconstruction\Mat_rc'

# load
H_exp = np.load(Path(mat_folder) / f'motifs_Hadamard_{M}_{N}.npy')
#H_exp = np.flip(H_exp,1).copy() # copy() required to remove negatives strides
H_exp /= H_exp[0,16:500].mean()
H_tar = walsh_matrix(N)
H_tar = H_tar[:M]

# plot
f, axs = plt.subplots(2, 1)
axs[0].set_title('Target measurement patterns')
im = axs[0].imshow(H_tar, cmap='gray') 
add_colorbar(im, 'bottom')
axs[0].get_xaxis().set_visible(False)

axs[1].set_title('Experimental measurement patterns')
im = axs[1].imshow(H_exp, cmap='gray') 
add_colorbar(im, 'bottom')
axs[1].get_xaxis().set_visible(False)

plt.savefig(Path(fig_folder) / 'patterns', bbox_inches='tight', dpi=600)

#%% Simulate measuments
import torch
from meas_dev import Hadam1Split
from recon_dev import DC1Net, Pinv1Net, Tikho1Net
from statistics_dev import data_loaders_ImageNet
from spyrit.core.meas import LinearSplit
from spyrit.core.prep import SplitPoisson
from spyrit.core.noise import Poisson
import matplotlib.pyplot as plt

M = 128
N = 512
alpha = 10 # in photons/pixels
b = 0
bs = 50 # batch size

# A batch of images
data_folder = '../../data/ILSVRC2012_v10102019'
dataloaders = data_loaders_ImageNet(data_folder, img_size=N, batch_size=10)  
x, _ = next(iter(dataloaders['train']))
x = x.view(-1,N,N) 

# Raw measurement
linop_tar = Hadam1Split(M,N)
noise_tar = Poisson(linop_tar, alpha)
prep_tar  = SplitPoisson(alpha, linop_tar)

linop_exp = LinearSplit(H_exp, pinv=True)
noise_exp = Poisson(linop_exp, alpha)
prep_exp = SplitPoisson(alpha, linop_exp)

y = noise_exp(x)
m = prep_exp(y)

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

#%% Pseudo inverse + unet
from spyrit.core.nnet import Unet
from spyrit.core.train import load_net

save_tag = True
net_prefix = 'pinv-net_unet_imagenet_ph_10'
net_suffix = 'N_512_M_128_epo_30_lr_0.001_sss_10_sdr_0.5_bs_20_reg_1e-07'

#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
y = y.to(device)
y = y.view(-1,1,N,2*M) 

# Reconstruction using target patterns
pinvnet_tar = Pinv1Net(noise_tar, prep_tar, Unet())

# Load trained DC-Net
title = './model/' + net_prefix + '_Hadam1_' + net_suffix
load_net(title, pinvnet_tar, device, False)
pinvnet_tar.to(device)  # Mandantory when batchNorm is used
pinvnet_tar.eval()

# Reconstruction using target patterns ----------------------------------------
x_net_tar = pinvnet_tar.reconstruct(y)
x_net_tar = x_net_tar.view(-1,N,N).detach()


# Reconstruction using exp patterns (Net) -------------------------------------
del pinvnet_tar
pinvnet_exp = Pinv1Net(noise_exp, prep_exp, Unet())

# Load trained DC-Net
title = './model/' + net_prefix + '_exp_' + net_suffix
load_net(title, pinvnet_exp, device, False)
pinvnet_exp.eval()      # Mandantory when batchNorm is used   
pinvnet_exp.to(device)  

x_net_exp = pinvnet_exp.reconstruct(y)
x_net_exp = x_net_exp.view(-1,N,N).detach()

# Reconstruction using exp patterns (pinv) ------------------------------------
pinvnet_exp.denoi = torch.nn.Identity().to(device)
x_pinv_exp = pinvnet_exp.reconstruct(y)
x_pinv_exp = x_pinv_exp.view(-1,N,N).detach()


# Plot
fig, axs = plt.subplots(1, 3, figsize=(15,5))
fig.suptitle(fr'$\alpha =$ {alpha} photons (train and test)')

im = axs[0].imshow(x_pinv_exp[b,:,:].cpu())
axs[0].set_title('Pinv, experimental patterns')
plt.colorbar(im, ax=axs[0])

im = axs[1].imshow(x_net_tar[b,:,:].cpu())
axs[1].set_title('Pinv-Net, target patterns')
plt.colorbar(im, ax=axs[1])

im = axs[2].imshow(x_net_exp[b,:,:].cpu())
axs[2].set_title('Pinv-Net, experimental patterns')
plt.colorbar(im, ax=axs[2])
            
if save_tag:
    save_folder = Path(fig_folder)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_folder / f'recon_sim_{alpha}ph_pinvnet.png', bbox_inches='tight', dpi=600)