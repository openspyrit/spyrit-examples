# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:19:58 2023

Reconstructions with data simulated using experimental measurements 

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

M = 128
N = 512
data_folder = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D/'
mat_folder = '/Reconstruction/Mat_rc/'

# load
H_exp = np.load(Path(data_folder + mat_folder) / f'motifs_Hadamard_{M}_{N}.npy')
H_exp = np.flip(H_exp,1).copy() # copy() required to remove negatives strides
H_exp /= H_exp[0,16:500].mean()
H_exp = H_exp #/ H_exp[0,:]
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

#plt.savefig(Path(fig_folder) / 'patterns', bbox_inches='tight', dpi=600)

#%% Simulate measuments
import torch
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
bs = 10 # batch size

# A batch of images
data_folder = '../../data/ILSVRC2012_v10102019'
dataloaders = data_loaders_ImageNet(data_folder, img_size=N, batch_size=10)  
x, _ = next(iter(dataloaders['train']))
x = x.view(-1,N,N) 

# Raw measurement
linop_exp = LinearSplit(H_exp, pinv=True)
noise_exp = Poisson(linop_exp, alpha)
prep_exp = SplitPoisson(alpha, linop_exp)

y_sim = noise_exp(x)
m_sim = prep_exp(y_sim)

fig, axs = plt.subplots(1, 3, figsize=(15,7))
fig.suptitle(fr'M = {M}, N = {N}, $\alpha$ = {alpha}')

im = axs[0].imshow(x[b,:,:].cpu())
axs[0].set_title('Ground-truth')
plt.colorbar(im, ax=axs[0])

im = axs[1].imshow(y_sim[b,:,:].cpu())
axs[1].set_title('Meas (raw)')
plt.colorbar(im, ax=axs[1])

im = axs[2].imshow(m_sim[b,:,:].cpu())
axs[2].set_title('Meas (after prep)')
plt.colorbar(im, ax=axs[2])
#plt.savefig(Path(fig_folder) / 'measurements', bbox_inches='tight', dpi=600)

#%% Load prep data
data_folder = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D/'
Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/data_2023_03_13/'
Run = 'RUN0004' 
Ns = int(Run[-1])-1
save_folder = '/Preprocess/'
Nl, Nh, Nc = 512, 128, 128

filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_pos.npy'
prep_pos = np.load(Path(data_folder+save_folder) / filename)

filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_neg.npy'
prep_neg =  np.load(Path(data_folder+save_folder) / filename)

# spectral dimension comes first
prep_pos = np.moveaxis(prep_pos, -1, 0)
prep_neg = np.moveaxis(prep_neg, -1, 0)

nc, nl, nh = prep_neg.shape
y_exp = np.zeros((nc, nl, 2*nh))
y_exp[:,:,::2]  = prep_pos#/np.expand_dims(prep_pos[:,:,0], axis=2)
y_exp[:,:,1::2] = prep_neg#/np.expand_dims(prep_pos[:,:,0], axis=2)
y_exp = torch.from_numpy(y_exp)


y_exp = y_exp[55:65,:,:].to(torch.float32)
m_exp = prep_exp(y_exp)

b = 0

fig, axs = plt.subplots(1, 2)#, figsize=(15,7))
fig.suptitle(fr'M = {M}, N = {N}, $\alpha$ = {alpha}')

im = axs[0].imshow(y_exp[b,:,:].cpu())
axs[0].set_title('Meas (raw)')
plt.colorbar(im, ax=axs[0])

im = axs[1].imshow(m_exp[b,:,:].cpu())
axs[1].set_title('Meas (after prep)')
plt.colorbar(im, ax=axs[1])
#plt.savefig(Path(fig_folder) / 'measurements', bbox_inches='tight', dpi=600)


#%% DC versus Tikhonov (target patterns, diagonal cov)
from meas_dev import Hadam1Split
from spyrit.misc.disp import imagepanel


y = y_sim
y = y_exp

# Init
linop_tar = Hadam1Split(M,N)
noise_tar = Poisson(linop_tar, alpha)
prep_tar  = SplitPoisson(alpha, linop_tar)

# covariance prior in the image domain
sigma = 1e1*np.eye(N)*N # diagonal, same as DC1Net

# -- Reconstrcution using target patterns -------------------------------------
H = walsh_matrix(N)

pinvnet_tar = Pinv1Net(noise_tar, prep_tar)
tiknet_tar = Tikho1Net(noise_tar, prep_tar, sigma)
dcnet = DC1Net(noise_tar, prep_tar, H @ sigma @ H)

# Choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = x.to(device)
y = y.to(device)
dcnet.to(device)
tiknet_tar.to(device)
pinvnet_tar.to(device)

# Reconstruction
x_pinv= pinvnet_tar.reconstruct(y)
x_dc  = dcnet.reconstruct(y)
x_tik = tiknet_tar.reconstruct(y)

# Plot
b = 0
imagepanel(x[b,:,:].cpu(),
           x_pinv[b,:,:].cpu(),
           x_dc[b,:,:].cpu(),
           x_tik[b,:,:].cpu(),
           f'{alpha:4g} photons | target patterns',
          'Ground-truth','Pinv-Net','DC-Net (target)','Tikhonov')

#plt.savefig(Path(fig_folder) / f'meas_Hexpe_rec_Htar_cov_diag_{alpha}ph', bbox_inches='tight', dpi=600)

#%% DC versus Tikhonov (experimental patterns, diagonal cov)
pinvnet_exp = Pinv1Net(noise_exp, prep_exp)
tiknet_exp = Tikho1Net(noise_exp, prep_exp, sigma)

# Choose device
pinvnet_exp.to(device)
tiknet_exp.to(device)

# Reconstruction
x_pinv = pinvnet_exp.reconstruct(y)
x_tik  = tiknet_exp.reconstruct(y)

# Plot
imagepanel(x[b,:,:].cpu(),
           x_pinv[b,:,:].cpu(),
           x_dc[b,:,:].cpu(),
           x_tik[b,:,:].cpu(),
           f'{alpha:4g} photons | experimental patterns',
          'Ground-truth','Pinv-Net','DC-Net (target)','Tikhonov')
            
#plt.savefig(Path(fig_folder) / f'meas_Hexpe_rec_Hexpe_cov_diag_{alpha}ph', bbox_inches='tight', dpi=600)

#%% DC versus Tikhonov (target patterns, ImageNet cov)
# covariance prior in the image domain
stat_folder = './stat/'
cov_file   = f'Cov_1_{N}x{N}.npy'
mean_file   = f'Average_1_{N}x{N}.npy'
mean  = np.load(Path(stat_folder) / mean_file)
sigma = np.load(Path(stat_folder) / cov_file)

# -- Reconstrcution using target patterns -------------------------------------
tiknet_tar = Tikho1Net(noise_tar, prep_tar, sigma)
dcnet = DC1Net(noise_tar, prep_tar, H @ sigma @ H)

# Choose device
tiknet_tar.to(device)
dcnet.to(device)

# Reconstruction
x_pinv= pinvnet_tar.reconstruct(y)
x_dc  = dcnet.reconstruct(y)
x_tik = tiknet_tar.reconstruct(y)

# Plot
b = 0
imagepanel(x[b,:,:].cpu(),
           x_pinv[b,:,:].cpu(),
           x_dc[b,:,:].cpu(),
           x_tik[b,:,:].cpu(),
           f'{alpha:4g} photons | target patterns',
          'Ground-truth','Pinv-Net','DC-Net (target)','Tikhonov')

#plt.savefig(Path(fig_folder) / f'meas_Hexpe_rec_Htar_cov_imageNet_{alpha}ph', bbox_inches='tight', dpi=600)

#%% DC versus Tikhonov (experimental patterns, ImageNet cov)
tiknet_exp = Tikho1Net(noise_exp, prep_exp, sigma)

# Choose device
tiknet_exp.to(device)

# Reconstruction
x_pinv= pinvnet_exp.reconstruct(y)
x_dc  = dcnet.reconstruct(y)
x_tik = tiknet_exp.reconstruct(y)

# Plot
b = 0
imagepanel(x[b,:,:].cpu(),
           x_pinv[b,:,:].cpu(),
           x_dc[b,:,:].cpu(),
           x_tik[b,:,:].cpu(),
           f'{alpha:4g} photons | experimental patterns',
          'Ground-truth','Pinv-Net','DC-Net (target)','Tikhonov')

#plt.savefig(Path(fig_folder) / f'meas_Hexpe_rec_Hexpe_cov_imageNet_{alpha}ph', bbox_inches='tight', dpi=600)


#%% Compare with Seb's reconstruction
data_folder = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D/'
rec_folder = '/Reconstruction/hyper_cube/'
Run = 'RUN0004' 
Nl, Nh, Nc = 512, 512, 128
Ns = int(Run[-1])-1

Pinv = np.linalg.pinv(np.transpose(H_exp/np.max(H_exp)))
m_np = m_exp[b,:,:].cpu().numpy()
rec_seb2 = np.dot(m_np,Pinv)

filename = f'T{Ns}_{Run}_2023_03_13_Had_rc_pinv_{Nl}x{Nh}x{Nc}.npy'
rec_seb = np.load(Path(data_folder+rec_folder) / filename)

f, axs = plt.subplots(1, 3)
axs[0].set_title('Seb\'s pinv reconstruction')
im = axs[0].imshow(rec_seb[:,:,b+55], cmap='gray') 
add_colorbar(im, 'bottom')
axs[0].get_xaxis().set_visible(False)

axs[1].set_title('Nico\'s reconstrcution')
im = axs[1].imshow(x_pinv[b,:,:].cpu(), cmap='gray') 
add_colorbar(im, 'bottom')
axs[1].get_xaxis().set_visible(False)

axs[2].set_title('Seb\'s pinv reconstruction 2')
im = axs[2].imshow(rec_seb2, cmap='gray') 
add_colorbar(im, 'bottom')
axs[2].get_xaxis().set_visible(False)
