# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 18:06:44 2023

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
alpha = 1e0 # in photons/pixels
b = 0

# Raw measurement
linop_exp = LinearSplit(H_exp, pinv=True)
noise_exp = Poisson(linop_exp, alpha)
prep_exp = SplitPoisson(alpha, linop_exp)

#%% Load prep data
save_tag = False
data_folder = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D/'
data_subfolder = 'data_2023_03_14/'
Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/'
Run = 'RUN0002' 
Ns = int(Run[-1])+5
save_folder = '/Preprocess/'
Nl, Nh, Nc = 512, 128, 128

filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_pos.npy'
prep_pos = np.load(Path(data_folder+save_folder) / filename)

filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_neg.npy'
prep_neg =  np.load(Path(data_folder+save_folder) / filename)

# spectral dimension comes first
prep_pos = np.moveaxis(prep_pos, -1, 0)
prep_neg = np.moveaxis(prep_neg, -1, 0)


print(f'Pos data: range={prep_pos.max() - prep_pos.min()} counts; mean={prep_pos.mean()} counts')

nc, nl, nh = prep_neg.shape
y_exp = np.zeros((nc, nl, 2*nh))
y_exp[:,:,::2]  = prep_pos #/np.expand_dims(prep_pos[:,:,0], axis=2)
y_exp[:,:,1::2] = prep_neg #/np.expand_dims(prep_pos[:,:,0], axis=2)
y_exp = torch.from_numpy(y_exp)

y_exp = y_exp[55:65,:,:].to(torch.float32)
m_exp = prep_exp(y_exp)

b = 0

fig, axs = plt.subplots(1, 2)#, figsize=(15,7))
fig.suptitle(fr'M = {M}, N = {N}')

im = axs[0].imshow(y_exp[b,:,:].cpu())
axs[0].set_title('Meas (raw)')
plt.colorbar(im, ax=axs[0])

im = axs[1].imshow(m_exp[b,:,:].cpu())
axs[1].set_title('Meas (after prep)')
plt.colorbar(im, ax=axs[1])

if save_tag:
    save_folder = Path(fig_folder) / Path(data_subfolder)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_folder / f'T{Ns}_{Run}_raw_measurements', bbox_inches='tight', dpi=600)
    
#%% Load prep data
save_tag = False


save_folder = '/Preprocess/'
filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_pos.npy'
prep_pos = np.load(Path(data_folder+save_folder) / filename)

filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_neg.npy'
prep_neg =  np.load(Path(data_folder+save_folder) / filename)

# spectral dimension comes first
prep_pos = np.moveaxis(prep_pos, -1, 0)
prep_neg = np.moveaxis(prep_neg, -1, 0)


print(f'Pos data: range={prep_pos.max() - prep_pos.min()} counts; mean={prep_pos.mean()} counts')

# param #1
background  = prep_pos.min()
gain = 1e6

# param #2
#background  = 0
#gain = 1e0

prep_pos = gain*(prep_pos - background)
prep_neg = gain*(prep_neg - background)


nc, nl, nh = prep_neg.shape
y_exp = np.zeros((nc, nl, 2*nh))
y_exp[:,:,::2]  = prep_pos #/np.expand_dims(prep_pos[:,:,0], axis=2)
y_exp[:,:,1::2] = prep_neg #/np.expand_dims(prep_pos[:,:,0], axis=2)
y_exp = torch.from_numpy(y_exp)

y_exp = y_exp[55:65,:,:].to(torch.float32)
m_exp = prep_exp(y_exp)

b = 0

fig, axs = plt.subplots(1, 2)#, figsize=(15,7))
fig.suptitle(fr'M = {M}, N = {N}')

im = axs[0].imshow(y_exp[b,:,:].cpu())
axs[0].set_title('Meas (raw)')
plt.colorbar(im, ax=axs[0])

im = axs[1].imshow(m_exp[b,:,:].cpu())
axs[1].set_title('Meas (after prep)')
plt.colorbar(im, ax=axs[1])

if save_tag:
    save_folder = Path(fig_folder) / Path(data_subfolder)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_folder / f'T{Ns}_{Run}_measurements', bbox_inches='tight', dpi=600)

#%% DC versus Tikhonov (target patterns, diagonal cov)
from meas_dev import Hadam1Split
from spyrit.misc.disp import imagepanel

save_tag = False
y = y_exp

# Init
linop_tar = Hadam1Split(M,N)
noise_tar = Poisson(linop_tar, alpha)
prep_tar  = SplitPoisson(alpha, linop_tar)

# covariance prior in the image domain
sigma = 1e1*np.eye(N)*N # diagonal, same as DC1Net

# -- Reconstrcution using target patterns -------------------------------------
H = walsh_matrix(N)
dcnet = DC1Net(noise_tar, prep_tar, H @ sigma @ H)
pinvnet_exp = Pinv1Net(noise_exp, prep_exp)

# # Choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
y = y.to(device)
dcnet.to(device)
pinvnet_exp.to(device)

# Reconstruction
x_dc  = dcnet.reconstruct(y)
x_pinv = pinvnet_exp.reconstruct(y)

# Plot
fig, axs = plt.subplots(1, 2, figsize=(13,7))
#fig.suptitle(fr'M = {M}, N = {N}')

im = axs[0].imshow(x_pinv[b,:,:].cpu())
axs[0].set_title('Pinv')
plt.colorbar(im, ax=axs[0])

im = axs[1].imshow(x_dc[b,:,:].cpu())
axs[1].set_title('DC (eye cov)')
plt.colorbar(im, ax=axs[1])
            
if save_tag:
    save_folder = Path(fig_folder) / Path(data_subfolder)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_folder / f'T{Ns}_{Run}_pinv_dc.png', bbox_inches='tight', dpi=600)

#%% DC versus Tikhonov (expe patterns)
sigma = 1e1*np.eye(N)*N # diagonal, same as DC1Net
tiknet_exp = Tikho1Net(noise_exp, prep_exp, sigma)
x_tik_diag  = tiknet_exp.reconstruct(y)

# covariance prior in the image domain
stat_folder = './stat/'
cov_file   = f'Cov_1_{N}x{N}.npy'
mean_file   = f'Average_1_{N}x{N}.npy'
mean  = np.load(Path(stat_folder) / mean_file)
sigma = np.load(Path(stat_folder) / cov_file)

# param #1
#sigma /= 1e-7

# param 2
sigma /= 1e-9


tiknet_exp = Tikho1Net(noise_exp, prep_exp, sigma)
tiknet_exp.to(device)

# Reconstruction
x_tik_full = tiknet_exp.reconstruct(y)

# Plot
b = 0

fig, axs = plt.subplots(1, 2, figsize=(13,7))
#fig.suptitle(fr'M = {M}, N = {N}')

im = axs[0].imshow(x_tik_diag[b,:,:].cpu())
axs[0].set_title('Tikho Eye')
plt.colorbar(im, ax=axs[0])

im = axs[1].imshow(x_tik_full[b,:,:].cpu())
axs[1].set_title('Tikho Full')
plt.colorbar(im, ax=axs[1])

if save_tag:
    save_folder = Path(fig_folder) / Path(data_subfolder)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_folder / f'T{Ns}_{Run}_tikho.png', bbox_inches='tight', dpi=600)
