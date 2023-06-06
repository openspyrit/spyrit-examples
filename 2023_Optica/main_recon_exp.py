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

#%%
import numpy as np
from PIL import Image
import sys
sys.path.append('./fonction')
from load_data import Select_data
from matrix_tools import bining_colonne

def load_data_pos_neg(Dir,Run,Nl,Nh,Nc):
    
    Path_files, list_files = Select_data(Dir,Run)
    
    Data_pos = np.zeros((Nl,Nh,Nc))
    Data_neg = np.zeros((Nl,Nh,Nc))
    
    for i in range(0,2*Nh,2):       
        Data_pos[:,i//2] = np.float_(np.rot90(np.array(Image.open(Path_files+list_files[i]))))
        Data_neg[:,i//2] = np.float_(np.rot90(np.array(Image.open(Path_files+list_files[i+1]))))
    
    motif_pos = np.zeros((Nh,Nc))
    motif_neg = np.zeros((Nh,Nc))
    
    for i in range(Nh):
        motif_pos[i] = np.sum(motif_pos[1000:1048,i],0)
        motif_neg[i] = np.sum(motif_neg[1000:1048,i],0)
    
    Bin_col = 4
    motif_pos = bining_colonne(motif_pos, Bin_col)
    motif_neg = bining_colonne(motif_neg, Bin_col)
    
    return motif_pos, motif_neg


# DATA NOT IN THE WAREHOUSE!
Dir = '../../data/2023_02_28_mRFP_DsRed_3D/Raw_data_chSPSIM_and_SPIM/data_2023_02_28/'
Run = 'RUN0002' 
Nl = 2048 # number of lines of the imaging camera
Nc = 2048 # number of columns of the imaging camera
Nh = 128#64# #number of patterns acquired

#load_data_pos_neg(Dir,Run,Nl,Nh,Nc)

#%% Load experimental measurement matrix (after split compensation)
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
from spyrit.core.noise import Poisson
import matplotlib.pyplot as plt

M = 128
N = 512
alpha = 50 # in photons/pixels
b = 0
bs = 50 # batch size

# A batch of images
data_folder = '../../data/ILSVRC2012_v10102019'
dataloaders = data_loaders_ImageNet(data_folder, img_size=N, batch_size=10)  
x, _ = next(iter(dataloaders['train']))
x = x.view(-1,N,N) 

# Raw measurement
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

#%% DC versus Tikhonov (target patterns, diagonal cov)
from meas_dev import Hadam1Split
from spyrit.misc.disp import imagepanel

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

plt.savefig(Path(fig_folder) / f'meas_Hexpe_rec_Htar_cov_diag_{alpha}ph', bbox_inches='tight', dpi=600)

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
            
plt.savefig(Path(fig_folder) / f'meas_Hexpe_rec_Hexpe_cov_diag_{alpha}ph', bbox_inches='tight', dpi=600)

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

plt.savefig(Path(fig_folder) / f'meas_Hexpe_rec_Htar_cov_imageNet_{alpha}ph', bbox_inches='tight', dpi=600)

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

plt.savefig(Path(fig_folder) / f'meas_Hexpe_rec_Hexpe_cov_imageNet_{alpha}ph', bbox_inches='tight', dpi=600)


#%% Computation time
import timeit

t = timeit.timeit(lambda: pinvnet_exp.reconstruct(y), number=100)
print(f"Pinv-Net reconstruction ({device}, 100x {bs} images): {t:.3f} seconds")

t = timeit.timeit(lambda: dcnet.reconstruct(y), number=100)
print(f"DC-Net reconstruction ({device}, 100x {bs} images): {t:.3f} seconds")

t = timeit.timeit(lambda: tiknet_exp.reconstruct(y), number=100)
print(f"Tikho-Net reconstruction ({device}, 100x {bs} images): {t:.3f} seconds")
