# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 15:07:16 2025

@author: ducros
"""
# %%
# Imports
# --------------------------------------------------------------------
import ast
import json
from pathlib import Path
from typing import OrderedDict

import torch
import torch.nn as nn
import numpy as np

import spyrit.core.meas as meas
import spyrit.core.prep as prep
import spyrit.core.recon as recon
import spyrit.core.nnet as nnet
import spyrit.core.train as train
import spyrit.misc.sampling as samp

# %%
# General
# ====================================================================
# Experimental data
data_folder = "data/"  # measurements
model_folder = "model/"  # reconstruction models
stat_folder = '../../tomoradio-spyrit/_stat/ILSVRC2012_v10102019/'  # statistics

recon_folder = "recon/figure_4/"  # reconstructed images

# Full paths
data_folder_full = Path.cwd() / Path(data_folder)
model_folder_full = Path.cwd() / Path(model_folder)
stat_folder_full = Path.cwd() / Path(stat_folder)
recon_folder_full = Path.cwd() / Path(recon_folder)
recon_folder_full.mkdir(parents=True, exist_ok=True)

# choose by name which experimental data to use
data_subfolder = "2025-12-05_test_demo"
data_title = "obj_cat_bicolor_source_white_LED_Walsh_im_64x64_ti_4ms_zoom_x1"

suffix = {"data": "_spectraldata.npz", "metadata": "_metadata.json"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %%
# Measurement operators
# ====================================================================
# Size of the reconstructed images
img_size = 128  

# number of measurements
subsampling_factor = 2      # acquisition was 64 = 128 / 2
M = (img_size // subsampling_factor) ** 2

# Measurement and noise operators
Ord_rec = torch.ones(img_size, img_size)
Ord_rec[:, img_size // 2 :] = 0
Ord_rec[img_size // 2 :, :] = 0

meas_op = meas.HadamSplit2d(img_size, M, Ord_rec, device=device)
prep_op = prep.UnsplitRescaleEstim(meas_op, use_fast_pinv=True)
prep_unsplit  = prep.Unsplit()
rerange = prep.Rerange((0, 1), (-1, 1))

# %%
# Load experimental data and prepare it for reconstruction
# ====================================================================
print("Looking for data in", data_folder_full)

# Collect data in numpy
exp_data = np.load(data_folder_full / 
        data_subfolder / 
        data_title / 
        (data_title + suffix["data"]))["spectral_data"]
# Read metadata
# Note: we replace "np.int32(" with an empty string and ")" with an empty string

file = open(data_folder_full / data_subfolder 
            / data_title / (data_title + suffix["metadata"]), "r")
json_metadata = json.load(file)[4]
file.close()

tmp = json_metadata["patterns"]
tmp = tmp.replace("np.int32(", "").replace(")", "")

patterns = ast.literal_eval(tmp)
wavelengths = ast.literal_eval(json_metadata["wavelengths"])
wavelengths = np.array(wavelengths)
n_lambda = wavelengths.shape[0]

# %% spectral subsampling
lambda_list = [500, 1500, 2000] 
wav_list = [wavelengths[lamb] for lamb in lambda_list]
n_lambda = len(lambda_list)

l1,l2 = 450,650
exp_sum  = exp_data[:, l1:l2].sum(axis=1)
exp_data = exp_data[:, lambda_list]

# concatenate sum of all wavelengths
exp_data = np.concatenate((exp_data, exp_sum[:,None]), axis=1)
wav_list = wav_list + [f'pinv \n[{wavelengths[l1]:.1f}--{wavelengths[l2]:.1f}] nm']

# %%
# Reorder measurements to match with the reconstruction order
# ====================================================================
acq_size = img_size // subsampling_factor
Ord_acq = (-np.array(patterns)[::2]//2).reshape((acq_size, acq_size))

# Define the two permutation matrices used to reorder the measurements
# measurement order -> natural order -> reconstruction order
Perm_rec = samp.Permutation_Matrix(Ord_rec)
Perm_acq = samp.Permutation_Matrix(Ord_acq).T
measurements = samp.reorder(exp_data, Perm_acq, Perm_rec)

# %%
# Reconstruct all spectral slices
# ====================================================================
measurements_slice = np.zeros((2 * M, n_lambda+1))

# take only the first 2*M measurements of the right wavelength
measurements_slice = measurements[: 2 * M, :].T.reshape((n_lambda+1, 1, 2 * M))
measurements_slice = torch.from_numpy(measurements_slice).to(
    device, dtype=torch.float32
    )
reconstruct_size = torch.Size((n_lambda+1, 1, img_size, img_size))
        
#%% Pseudo-inverse

m = prep_unsplit(measurements_slice)
x_pinv = meas_op.fast_pinv(m)

#%% Plot pseudo-inverse
from spyrit.misc.disp import imagesc 
import matplotlib.pyplot as plt
plt.ion()


fig, axs = plt.subplots(1,4)
fig.suptitle('Pseudo-inverse (least squares)')

for ii, wav in enumerate(wav_list):
    
    imagesc(x_pinv[ii,0].rot90(k=2).cpu(),
            colormap = wav if ii<n_lambda else None,
            #colormap = wavelength_to_colormap(wav, gamma=0.8),
            cbar_pos = 'bottom',
            title=f'{wav:.1f} nm' if ii<n_lambda else f'{wav}',
            gamma = .8,
            fig = fig,
            ax=axs[ii],
            )

# %%
# Pinv-Net
# ====================================================================
model_name = "pinv-net_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07_retrained_light.pth"
denoiser = OrderedDict(
    {"rerange": rerange, "denoi": nnet.Unet(), "rerange_inv": rerange.inverse()}
)
denoiser = nn.Sequential(denoiser)
# this function loads the model into the '.denoi' key present in the second
# argument. It fails if it does not find the '.denoi' key.
train.load_net(model_folder_full / model_name, denoiser, device, False)

# Init
pinvnet = recon.PinvNet(meas_op, prep_op, denoiser, device=device)
pinvnet.eval()

# Reconstruct
x_pinvnet = torch.zeros(reconstruct_size, device=device)

lambda_batch_size = 256
lambda_batch_indices = np.arange(0, n_lambda, lambda_batch_size, dtype=np.uint64)

with torch.no_grad():
    for lambda_start in lambda_batch_indices:
        lambda_end = int(lambda_start + lambda_batch_size)
    
        print(f'Channels: {lambda_start}--{lambda_end}')
        
        rec = pinvnet.reconstruct(measurements_slice[lambda_start:lambda_end])
        rec *= pinvnet.prep.alpha[...,None]
        x_pinvnet[lambda_start:lambda_end, ...] = rec
            
del pinvnet
del denoiser
torch.cuda.empty_cache()

#%% Plot Pinv-Net
from spyrit.misc.disp import imagesc 
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1,n_lambda+1)
fig.suptitle('Pinv-Net')

for ii, wav in enumerate(wav_list):
    
    imagesc(x_pinvnet[ii,0].rot90(k=2).cpu() if ii<n_lambda else x_pinv[ii,0].rot90(k=2).cpu(),
            colormap = wav if ii<n_lambda else None,
            #colormap = wavelength_to_colormap(wav, gamma=0.8),
            cbar_pos = 'bottom',
            title=f'{wav:.1f} nm' if ii<n_lambda else f'{wav}',
            gamma = .8,
            fig = fig,
            ax=axs[ii],
            )
    
# %%
# DC-Net
# ====================================================================
model_name = "dc-net_unet_imagenet_rect_N0_10_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07_light.pth"
denoiser = OrderedDict(
    {"rerange": rerange, "denoi": nnet.Unet(), "rerange_inv": rerange.inverse()}
)
denoiser = nn.Sequential(denoiser)
# this function loads the model into the '.denoi' key present in the second
# argument. It fails if it does not find the '.denoi' key.
train.load_net(model_folder_full / model_name, denoiser, device, False)

# Load covariance prior (image domain)
cov_name = stat_folder_full / 'ILSVRC2012_v10102019_resize'/ "Cov_im2_{}x{}.pt".format(img_size, img_size)
Cov = torch.load(cov_name, weights_only=True).to(device)

# Covariance in Hadamard domain
meas_full = meas.HadamSplit2d(img_size, device=device) # full transform
Cov = meas_full.adjoint_H(Cov.T)            # Cov^T H^T
Cov = meas_full.adjoint_H(Cov.T).T          # (Cov^T H^T)^T H^T = H Cov H^T

# Init
dcnet = recon.DCNet(meas_op, prep_op, Cov, denoiser, device=device)

dcnet.eval()

# Reconstruct
x_dcnet = torch.zeros(reconstruct_size, device=device)
        
lambda_batch_size = 256
lambda_batch_indices = np.arange(0, n_lambda, lambda_batch_size, dtype=np.uint64)

with torch.no_grad():
    for lambda_start in lambda_batch_indices:
        lambda_end = int(lambda_start + lambda_batch_size)
        
        print(f'Channels: {lambda_start}--{lambda_end}')
        rec = dcnet.reconstruct(measurements_slice[lambda_start:lambda_end])       
        rec*= dcnet.prep.alpha[...,None]
        x_dcnet[lambda_start:lambda_end, ...] = rec      
            
del dcnet
del denoiser
torch.cuda.empty_cache()

#%% Plot DC-Net
from spyrit.misc.disp import imagesc 
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1,n_lambda+1)
fig.suptitle('DC-Net')

for ii, wav in enumerate(wav_list):
    
    imagesc(x_dcnet[ii,0].rot90(k=2).cpu() if ii<n_lambda else x_pinv[ii,0].rot90(k=2).cpu(),
            colormap = wav if ii<n_lambda else None,
            #colormap = wavelength_to_colormap(wav, gamma=0.8),
            cbar_pos = 'bottom',
            title=f'{wav:.1f} nm' if ii<n_lambda else f'{wav}',
            gamma = .8,
            fig = fig,
            ax=axs[ii],
            )