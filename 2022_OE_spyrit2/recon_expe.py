# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 09:35:20 2022

@author: ducros



NB (15-Sep-22): to debug needs to run

import collections
collections.Callable = collections.abc.Callable

"""

#%%
import torch
import numpy as np
import spyrit.misc.walsh_hadamard as wh

from matplotlib import pyplot as plt

from spyrit.learning.model_Had_DCAN import *
from spyrit.misc.disp import torch2numpy, imagesc, plot

from spyrit.learning.nets import *
from spyrit.restructured.Updated_Had_Dcan import *
from spyrit.misc.metrics import psnr_
from spyrit.misc.disp import imagesc, add_colorbar, noaxis

from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Torch device: {device}')

# get debug in spyder
import collections
collections.Callable = collections.abc.Callable
#%% Load reconstruction network
img_size = 64
M = 1024
N0 = 10 # Check if we used 10 in the paper 
stat_folder = Path('data_online/') 
average_file= stat_folder / ('Average_{}x{}'.format(img_size,img_size)+'.npy')
cov_file    = stat_folder / ('Cov_{}x{}'.format(img_size,img_size)+'.npy')

net_arch    = 'pinv-net'  # 'dc-net' or 'pinv-net'
net_denoi   = 'unet'    # 'cnn' 'cnnbn' or 'unet'
net_data    = 'stl10'    # 'imagenet' or 'stl10'
net_suffix  = f'N0_{N0}_N_64_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07'

save_root = False

net_folder= f'{net_arch}_{net_denoi}_{net_data}/'
net_title = f'{net_arch}_{net_denoi}_{net_data}_{net_suffix}'
title = './model_v2/' + net_folder + net_title

# Init DC-Net
Mean = np.load(average_file)
Cov  = np.load(cov_file)

H =  wh.walsh2_matrix(img_size)
Ord = Cov2Var(Cov)
Perm = Permutation_Matrix(Ord)
Hperm = Perm@H
Pmat = Hperm[:M,:]

Cov_perm = Perm @ Cov @ Perm.T

Forward = Split_Forward_operator_ft_had(Pmat, Perm, img_size, img_size)
Noise = Bruit_Poisson_approx_Gauss(N0, Forward)
Prep = Split_diag_poisson_preprocess(N0, M, img_size**2)

# Image-domain denoising layer
if net_denoi == 'cnn':      # CNN no batch normalization
    Denoi = ConvNet()
elif net_denoi == 'cnnbn':  # CNN with batch normalization
    Denoi = ConvNetBN()
elif net_denoi == 'unet':   # Unet
    Denoi = Unet()

# Global Architecture
if net_arch == 'dc-net':        # Denoised Completion Network
    Cov_perm = Perm @ Cov @ Perm.T
    DC = Generalized_Orthogonal_Tikhonov(sigma_prior = Cov_perm, 
                                         M = M, 
                                         N = img_size**2)
    model = DC2_Net(Noise, Prep, DC, Denoi)
    
elif net_arch == 'pinv-net':    # Pseudo Inverse Network
    DC = Pinv_orthogonal()
    model = Pinv_Net(Noise, Prep, DC, Denoi)

# Load trained DC-Net
load_net(title, model, device)
model.eval()                    # Mandantory when batchNorm is used

#%% Load expe data and unsplit
data_root = Path('data_online/')
#-1
#data_folder = Path('usaf_x12/')
#data_file_prefix = 'zoom_x12_usaf_group5'                   
#-2
#data_folder = Path('usaf_x2/')
#data_file_prefix = 'zoom_x2_usaf_group2' 
#-3
#data_folder = Path('star_sector_x2/')
#data_file_prefix = 'zoom_x2_starsector' 
#-4
#data_folder = Path('star_sector_x12/')
#data_file_prefix = 'zoom_x12_starsector' 
#-5
#data_folder = Path('tomato_slice_x12/')
#data_file_prefix = 'tomato_slice_2_zoomx12' 
#-6
#data_folder = Path('tomato_slice_x2/')
#data_file_prefix = 'tomato_slice_2_zoomx2'
#-7
data_folder = Path('cat/')
data_file_prefix = 'Cat_whiteLamp'
#-8
# data_folder = Path('horse/')
# data_file_prefix = 'Horse_whiteLamp'

full_path = data_root / data_folder / (data_file_prefix + '_spectraldata.npz')
raw = np.load(full_path)
meas= raw['spectral_data']

#%% 
from spas import read_metadata
meta_path = data_root / data_folder / (data_file_prefix + '_metadata.json')

_, acquisition_parameters, _, _ = read_metadata(meta_path)
wavelengths = acquisition_parameters.wavelengths

# for info: wavelengths[500] = 579.097; wavelengths[1000] = 639.3931; wavelengths[1500] = 695.1232
    
#%% Reconstruct using Network
# ind_start = 500
# ind_end = ind_start + 500

# # Measurement vector
# m = torch.Tensor(meas[:2*M,ind_start:ind_end].T)
# m = m.view(-1,2*M).to(device)

# Measurement vector
ind = [50,500,1500]
m = torch.Tensor(meas[:2*M,ind].T)
m = m.view(-1,2*M).to(device)

# Net
model.to(device)
rec_net_gpu = model.reconstruct_expe(m)
#rec_net_gpu = model.reconstruct_expe(m,1,700,17)
rec_net = rec_net_gpu.cpu().detach().numpy().squeeze()

# Save
if save_root:
    (save_root/data_folder/net_title).mkdir(parents=True, exist_ok=True)
    full_path_net = save_root / data_folder /net_title / (data_file_prefix + f'_net_M_{M}')
    np.save(full_path_net, rec_net)
    
#%% Plot Net
rec_net = rec_net.reshape(-1,img_size,img_size)
ind_plot = [0, 250, 499]

ind_plot = [0, 1, 2]

fig , axs = plt.subplots(1,3)
#
im = axs[0].imshow(rec_net[ind_plot[0],:,:], cmap='gray')
add_colorbar(im)
axs[0].set_title(f"Net, channel = {wavelengths[ind_plot[0]]}")
#
im = axs[1].imshow(rec_net[ind_plot[1],:,:], cmap='gray')
add_colorbar(im)
axs[1].set_title(f"Net, channel = {wavelengths[ind_plot[1]]}")
#
im = axs[2].imshow(rec_net[ind_plot[2],:,:], cmap='gray')
add_colorbar(im)
axs[2].set_title(f"Net, channel = {wavelengths[ind_plot[2]]}")