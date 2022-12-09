# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 08:48:35 2022

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

#%% user-defined

# for reconstruction
N_rec = 128  
M = 1024    # [4096, 4095, 2048, 1024, 512, 256]   
N0 = 10     # Check if we used 10 in the paper
stat_folder_rec = Path('../../stat/ILSVRC2012_v10102019/') # works for for N = 64 only !!

# used for acquisition
N_acq = 64
stat_folder_acq =  Path('./data_online/') 

net_order   = 'var'
net_arch    = 'dc-net'      # ['dc-net','pinv-net']
net_denoi   = 'unet'        # ['unet', 'cnn']
net_data    = 'imagenet'    # 'imagenet' 
save_root = Path('./recon_128/')

#%% covariance matrix and network filnames
if N_rec < N_acq:   
    cov_rec_file= stat_folder_rec / ('Cov_{}x{}'.format(N_rec, N_rec)+'.npy')
    bs = 1024
    
elif N_rec == N_acq:
    cov_rec_file= stat_folder_rec / ('Cov_8_{}x{}'.format(N_rec, N_rec)+'.npy')
    bs = 1024

if N_rec == 128:
    stat_folder_rec = Path('../../stat/ILSVRC2012_v10102019/')
    cov_rec_file= stat_folder_rec/ ('Cov_8_{}x{}'.format(N_rec, N_rec)+'.npy')
    bs = 256
    
if N_acq==64:
    stat_folder_acq =  Path('./data_online/') 
    cov_acq_file= stat_folder_acq / ('Cov_{}x{}'.format(N_acq, N_acq)+'.npy')

elif N_acq == 128:
    stat_folder_acq = Path('../../stat/ILSVRC2012_v10102019/')
    cov_acq_file= stat_folder_acq / ('Cov_8_{}x{}'.format(N_acq, N_acq)+'.npy')
    
# 
net_suffix  = f'N0_{N0}_N_{N_rec}_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_{bs}_reg_1e-07'
net_folder= f'{net_arch}_{net_denoi}_{net_data}/'

#%% Init and load trained network
H =  wh.walsh2_matrix(N_rec)

# Covariance in hadamard domain
Cov_rec = np.load(cov_rec_file)
Cov_acq = np.load(cov_acq_file)
Ord_acq = Cov2Var(Cov_acq)

# Sampling order
if net_order == 'rect':
    Ord_rec = np.ones((N_rec, N_rec))
    n_sub = math.ceil(M**0.5)
    Ord_rec[:,n_sub:] = 0
    Ord_rec[n_sub:,:] = 0
    
elif net_order == 'var':
    Ord_rec = Cov2Var(Cov_rec)
    
    
# Init network  
Perm_rec = Permutation_Matrix(Ord_rec)
Hperm = Perm_rec @ H
Pmat = Hperm[:M,:]

Forward = Split_Forward_operator_ft_had(Pmat, Perm_rec, N_rec, N_rec)
Noise = Bruit_Poisson_approx_Gauss(N0, Forward)
Prep = Split_diag_poisson_preprocess(N0, M, N_rec**2)

Denoi = Unet()
Cov_perm = Perm_rec @ Cov_rec @ Perm_rec.T
DC = Generalized_Orthogonal_Tikhonov(sigma_prior = Cov_perm, 
                                     M = M, N = N_rec**2)
model = DC2_Net(Noise, Prep, DC, Denoi)

# Load trained DC-Net
net_title = f'{net_arch}_{net_denoi}_{net_data}_{net_order}_{net_suffix}'
title = './model_v2/' + net_folder + net_title
load_net(title, model, device)
model.eval()                    # Mandantory when batchNorm is used

#%% Load expe data and unsplit
data_root = Path('data_online/')

if N_acq == 128:
    data_folder = Path('star_sector_x6_128/')
    data_file_prefix = 'Star_Sector_test_+4_rounds_image_size_128x128_zoom_x6'

elif N_acq == 64: 
    data_folder = Path('star_sector_x12/')
    data_file_prefix = 'zoom_x12_starsector'
    data_folder = Path('tomato_slice_x12')
    data_file_prefix ='tomato_slice_2_zoomx12'
    data_folder = Path('usaf_x12/')
    data_file_prefix = 'zoom_x12_usaf_group5'

#%% 
from spas import read_metadata, plot_color, spectral_binning, spectral_slicing
meta_path = data_root / data_folder / (data_file_prefix + '_metadata.json')

_, acquisition_parameters, _, _ = read_metadata(meta_path)
wavelengths = acquisition_parameters.wavelengths    

#%% Load data
print(data_folder / data_file_prefix)
 
full_path = data_root / data_folder / (data_file_prefix + '_spectraldata.npz')
raw = np.load(full_path)
meas= raw['spectral_data']

# Order used for acquisistion
Perm_acq = Permutation_Matrix(Ord_acq)

# zero filling when reconstrcution res is higher than acquisition res
if N_rec > N_acq:
    
    # Natural order measurements (N_acq resolution)
    Perm_raw = np.zeros((2*N_acq**2,2*N_acq**2))
    Perm_raw[::2,::2] = Perm_acq.T     
    Perm_raw[1::2,1::2] = Perm_acq.T
    meas = Perm_raw @ meas
    
    # Square subsampling in the "natural" order
    Ord_sub = np.zeros((N_rec,N_rec))
    Ord_sub[:N_acq,:N_acq]= -np.arange(-N_acq**2,0).reshape(N_acq,N_acq)
    Perm_sub = Permutation_Matrix(Ord_sub) 
    
    # zero filled measurement (N_res resolution)
    zero_filled = np.zeros((2*N_rec**2,len(wavelengths)))
    zero_filled[:2*N_acq**2,:] = meas
    
    meas = zero_filled
    
    Perm_raw = np.zeros((2*N_rec**2,2*N_rec**2))
    Perm_raw[::2,::2] = Perm_sub.T     
    Perm_raw[1::2,1::2] = Perm_sub.T
    
    meas = Perm_raw @ meas

# Reorder measurements  
if N_rec == N_acq:
    # To reorder measurements
    Perm_sub = Perm_acq[:N_rec**2,:].T
    #Perm = Perm_rec @ Perm_acq[:N_rec**2,:].T
  
elif N_rec < N_acq:
    # Square subsampling in the "natural" order
    Ord_sub = np.zeros((N_acq,N_acq))
    Ord_sub[:N_rec,:N_rec]= -np.arange(-N_rec**2,0).reshape(N_rec,N_rec)
    Perm_sub = Permutation_Matrix(Ord_sub) 
    Perm_sub = Perm_sub[:N_rec**2,:]
    Perm_sub = Perm_sub @ Perm_acq.T    

if N_rec <= N_acq:   
    # Get both positive and negative coefficients permutated
    Perm = Perm_rec @ Perm_sub
    Perm_raw = np.zeros((2*N_rec**2,2*N_acq**2))
    Perm_raw[::2,::2] = Perm     
    Perm_raw[1::2,1::2] = Perm
    meas = Perm_raw @ meas
    
elif N_rec > N_acq:
    Perm = Perm_rec
    Perm_raw = np.zeros((2*N_rec**2,2*N_rec**2))
    Perm_raw[::2,::2] = Perm     
    Perm_raw[1::2,1::2] = Perm
    meas = Perm_raw @ meas

#%% Reconstruct a few spectral slice from full reconstruction
wav_min = 530 
wav_max = 730
wav_num = 8
meas_slice, wavelengths_slice, _ = spectral_slicing(meas.T, 
                                                wavelengths, 
                                                wav_min, 
                                                wav_max, 
                                                wav_num)
model.to(device)
model.PreP.set_expe()

with torch.no_grad():
    m = torch.Tensor(meas_slice[:2*M,:]).to(device)
    rec_gpu = model.reconstruct_expe(m)
    rec = rec_gpu.cpu().detach().numpy().squeeze()
    
#%% Plot or save 
# rotate
rec = np.rot90(rec,2,(1,2))
full_path = save_root / (data_folder.name + '_slice_' + net_title + '.pdf')
plot_color(rec, wavelengths_slice, fontsize=7, filename = full_path)

#%% Reconstruct a few spectral slice from full reconstruction
wav_min = 579#530 
wav_max = 579.1#730
wav_num = 1#8
meas_slice, wavelengths_slice, _ = spectral_slicing(meas.T, 
                                                wavelengths, 
                                                wav_min, 
                                                wav_max, 
                                                wav_num)
with torch.no_grad():
    m = torch.Tensor(meas_slice[:2*M,:]).to(device)
    rec_gpu = model.reconstruct_expe(m)
    rec = rec_gpu.cpu().detach().numpy().squeeze()
    
#%% Plot or save 
# rotate
rec = np.rot90(rec,2)

fig , axs = plt.subplots(1,1)
im = axs.imshow(rec, cmap='gray')
noaxis(axs)
add_colorbar(im, 'bottom')

full_path = save_root / (data_folder.name + '_' + net_title + '.pdf')
fig.savefig(full_path, bbox_inches='tight')# -*- coding: utf-8 -*-
