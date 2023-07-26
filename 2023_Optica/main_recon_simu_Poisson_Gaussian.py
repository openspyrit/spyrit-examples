# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:54:06 2023

@author: ducros
"""
#%% to debug needs to run
import collections
collections.Callable = collections.abc.Callable

fig_folder = './figure/'

#%% Load experimental measurement matrix (after split compensation)
import numpy as np
from pathlib import Path
from spyrit.misc.disp import imagesc, add_colorbar
import matplotlib.pyplot as plt
from spyrit.misc.walsh_hadamard import walsh_matrix

M = 128
N = 512
mat_folder = r'.\data\2023_03_13_2023_03_14_eGFP_DsRed_3D\Reconstruction\Mat_rc'

# load
H_exp = np.load(Path(mat_folder) / f'motifs_Hadamard_{M}_{N}.npy')
H_exp = np.flip(H_exp,1).copy() # copy() required to remove negatives strides
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
from recon_dev import DC1Net, Pinv1Net, Tikho1Net
from statistics_dev import data_loaders_ImageNet
from spyrit.core.meas import LinearSplit
from spyrit.core.prep import SplitPoisson
from spyrit.core.noise import PoissonGaussian
import matplotlib.pyplot as plt

M = 128
N = 512
alpha = 50 # in photons/pixels
gain = 2.65
nbin = 20*4
mudark = nbin*105
sigdark = (nbin**.5)*5

b = 2
bs = 50 # batch size

# A batch of images
data_folder = '../../data/ILSVRC2012_v10102019'
dataloaders = data_loaders_ImageNet(data_folder, img_size=N, batch_size=10)  
x, _ = next(iter(dataloaders['train']))
x = x.view(-1,N,N) 

# Raw measurements
#meas = LinearSplit(H_tar, pinv=True) # target patterns
meas = LinearSplit(H_exp, pinv=True) # real patterns
noise = PoissonGaussian(meas, alpha, gain, mudark, sigdark)
prep = SplitPoisson(alpha, meas)
prep.set_expe(gain, mudark, sigdark)

torch.manual_seed(4)
y = noise(x)
m = prep(y)

m_2, beta = prep.forward_expe(y, meas, (-2,-1))

m_2, beta = prep.forward_expe2(y, meas)
beta = torch.amax(beta, -2, keepdim=True)

beta = beta[b,:,:].numpy().squeeze()

# 
fig, axs = plt.subplots(1, 4, figsize=(15,7))
fig.suptitle(fr'M = {M}, N = {N}, $\alpha$ = {alpha}, g = {gain}')

im = axs[0].imshow(x[b,:,:].cpu())
axs[0].set_title('Ground-truth')
plt.colorbar(im, ax=axs[0])

im = axs[1].imshow(y[b,:,:].cpu())
axs[1].set_title('Meas (raw)')
plt.colorbar(im, ax=axs[1])

im = axs[2].imshow(m[b,:,:].cpu())
axs[2].set_title('Meas (after prep)')
plt.colorbar(im, ax=axs[2])

im = axs[3].imshow(m_2[b,:,:].cpu())
axs[3].set_title(fr'Meas (full prep, $\beta^*=${beta:.3})')
plt.colorbar(im, ax=axs[3])


#%% DC versus Tikhonov (target patterns, ImageNet cov)
from spyrit.misc.disp import imagepanel
# covariance prior in the image domain
stat_folder = './stat/'
cov_file   = f'Cov_1_{N}x{N}.npy'
mean_file   = f'Average_1_{N}x{N}.npy'
mean  = np.load(Path(stat_folder) / mean_file)
sigma = np.load(Path(stat_folder) / cov_file)
b = 2

# -- Reconstrcution using target patterns -------------------------------------
pinvnet = Pinv1Net(noise, prep)
tiknet = Tikho1Net(noise, prep, sigma)
#dcnet = DC1Net(noise_tar, prep_tar, H @ sigma @ H)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Choose device
#pinvnet.to(device)
#tiknet.to(device)
#dcnet.to(device)

# Reconstruction
x_pinv, beta = pinvnet.reconstruct_expe(y)
x_tik = tiknet.reconstruct(y)

x_tik_2, beta_2 = tiknet.reconstruct_expe(y)
#beta_2 = torch.amax(beta_2, -2, keepdim=True)

beta = beta[b,:,:].numpy().squeeze()
beta_2 = beta_2[b,:,:].numpy().squeeze()

# Plot
imagepanel(x[b,:,:].cpu(),
           x_pinv[b,:,:].cpu(),
           x_tik[b,:,:].cpu(),
           x_tik_2[b,:,:].cpu(),
           f'{alpha:4g} photons, g = {gain}','Ground-truth',
          fr'Pinv-Net (calib, $\beta^*=${beta:.3})',
          'Tikhonov', fr'Tikhonov (calib, $\beta^*=${beta_2:.3})')

#plt.savefig(Path(fig_folder) / f'meas_Hexpe_rec_Htar_cov_imageNet_{alpha}ph', bbox_inches='tight', dpi=600)

#%% DC versus Tikhonov (target patterns, ImageNet cov)
from spyrit.misc.disp import imagepanel
from spyrit.core.nnet import Unet
from spyrit.core.train import load_net

b = 2

# covariance prior in the image domain
stat_folder = './stat/'
cov_file   = f'Cov_1_{N}x{N}.npy'
mean_file   = f'Average_1_{N}x{N}.npy'
mean  = np.load(Path(stat_folder) / mean_file)
sigma = np.load(Path(stat_folder) / cov_file)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
filename = 'tikho-net_unet_imagenet_ph_10_exp_N_512_M_128_epo_20_lr_0.001_sss_10_sdr_0.5_bs_20_reg_1e-07'


# -- Reconstrcution using target patterns -------------------------------------
pinvnet = Pinv1Net(noise, prep)
tiknet = Tikho1Net(noise, prep, sigma, Unet())
title = './model/' + filename
load_net(title, tiknet, device, False)
tiknet.eval()

y = y[:,np.newaxis,:,:] 

# Choose device
# tiknet.to(device)
# pinvnet.to(device)
# tiknet.to(device)
# y = y.to(device)

# Reconstruction
with torch.no_grad():
    x_pinv, beta = pinvnet.reconstruct_expe(y)
    x_tik = tiknet.reconstruct(y)
    
    x_tik_2, beta_2 = tiknet.reconstruct_expe(y)
    #beta_2 = torch.amax(beta_2, -2, keepdim=True)
    
beta = beta[b,:,:].numpy().squeeze()
beta_2 = beta_2[b,:,:].numpy().squeeze()

# Plot
imagepanel(x[b,:,:].cpu(),
           x_pinv[b,0,:,:].cpu(),
           x_tik[b,0,:,:].cpu(),
           x_tik_2[b,0,:,:].cpu(),
           f'{alpha:4g} photons, g = {gain}','Ground-truth',
          fr'Pinv-Net (calib, $\beta^*=${beta:.3})',
          'Tikhonov', fr'Tikhonov (calib, $\beta^*=${beta_2:.3})')