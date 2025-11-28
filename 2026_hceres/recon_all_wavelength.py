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
data_subfolder = ["2025-11-10_test_HCERES",
                "2025-11-10_test_HCERES"
                ]
data_title = [#"tomato_slice_2_zoomx2", 
              #"zoom_x12_starsector",
              "obj_Cat_bicolor_source_white_LED_Walsh_im_64x64_ti_10ms_zoom_x1",
              "obj_Cat_bicolor_thin_overlap_source_white_LED_Walsh_im_64x64_ti_9ms_zoom_x1"
              ]

savenames = ["tomato", "starsector"]
suffix = {"data": "_spectraldata.npz", "metadata": "_metadata.json"}
n_meas = len(data_title)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)

img_size = 128  # image size
subsampling_factor = 2  # measure an equivalent image of size img_size // 2


# %%
# Measurement (and reconstruction) operators
# ====================================================================
# number of measurements
M = (img_size // subsampling_factor) ** 2

# Measurement and noise operators
Ord_rec = torch.ones(img_size, img_size)
Ord_rec[:, img_size // 2 :] = 0
Ord_rec[img_size // 2 :, :] = 0

meas_op = meas.HadamSplit2d(img_size, M, Ord_rec, device=device)
# noise_op = noise.Poisson(meas_op, 2).to(device)  # parameter alpha is unimportant here
prep_op = prep.UnsplitRescaleEstim(meas_op, use_fast_pinv=True)
rerange = prep.Rerange((0, 1), (-1, 1))


# %%
# Load experimental data and prepare it for reconstruction
# ====================================================================
print("Looking for data in", data_folder_full)

# Collect data in numpy
exp_data = [
    np.load(data_folder_full / subfolder / 
            title / (title + suffix["data"]))["spectral_data"]
    for (subfolder,title) in zip(data_subfolder,data_title) 
]
# Collect metadata
patterns = [[] for _ in range(n_meas)]
wavelengths = [[] for _ in range(n_meas)]
for ii, (subfolder,title) in enumerate(zip(data_subfolder,data_title)):
    file = open(data_folder_full / subfolder 
                / title / (title + suffix["metadata"]), "r")
    
    # "old" metadata
    # json_metadata = json.load(file)[3]
    # file.close()
    # patterns[ii] = ast.literal_eval(json_metadata["patterns"])
    # wavelengths[ii] = ast.literal_eval(json_metadata["wavelengths"])
    
    # "new" metadata
    json_metadata = json.load(file)[4]
    file.close()
    
    # replace "np.int32(" with an empty string and ")" with an empty string
    tmp = json_metadata["patterns"]
    tmp = tmp.replace("np.int32(", "").replace(")", "")
    patterns[ii] = ast.literal_eval(tmp)
    wavelengths[ii] = ast.literal_eval(json_metadata["wavelengths"])

# %%
# Reorder measurements to match with the reconstruction order
# ====================================================================
acq_size = img_size // subsampling_factor
Ord_acq = [
    (-np.array(patterns[i])[::2] // 2).reshape((acq_size, acq_size))
    for i in range(n_meas)
]

# %%
# Define the two permutation matrices used to reorder the measurements
# measurement order -> natural order -> reconstruction order
Perm_rec = samp.Permutation_Matrix(Ord_rec)
Perm_acq = [samp.Permutation_Matrix(Ord_acq[i]).T for i in range(n_meas)]
# each element of 'measurements' has shape (measurements, wavelengths)
measurements = [samp.reorder(exp_data[i], Perm_acq[i], Perm_rec) for i in range(n_meas)]

# %%
# Reconstruct all spectral slices
# ====================================================================
n_lambda = measurements[0].shape[1] # This is 2048
measurements_slice = [np.zeros((2 * M, n_lambda)) for _ in range(n_meas)]

# select the measurements in the right spectral slice
for i in range(n_meas):

    # take only the first 2*M measurements of the right wavelength
    measurements_slice[i] = measurements[i][: 2 * M, :].T.reshape(
        (n_lambda, 1, 2 * M)
    )
    measurements_slice[i] = torch.from_numpy(measurements_slice[i]).to(
        device, dtype=torch.float32
    )
reconstruct_size = torch.Size([n_meas]) + (n_lambda, 1, img_size, img_size)

# %%
# Pinv
# ====================================================================

# Init
pinvnet = recon.PinvNet(meas_op, prep_op, device=device)

# Reconstruct
x_pinvnet = torch.zeros(reconstruct_size, device=device)

lambda_batch_size = 256
lambda_batch_indices = np.arange(0, 2048, lambda_batch_size, dtype=np.uint64)

with torch.no_grad():
    for ii, y in enumerate(measurements_slice):
        for lambda_start in lambda_batch_indices:
            lambda_end = lambda_start + lambda_batch_size
            
            print(f'Channels: {lambda_start}--{lambda_end}')
            x_pinvnet[ii, lambda_start:lambda_end, ...] = pinvnet.reconstruct(
                                                y[lambda_start:lambda_end]
                                                )

#%% Plot Pinv
from spyrit.misc.disp import imagesc 

lambda_plot = 500 

for ii, _ in enumerate(measurements_slice):
    imagesc(x_pinvnet[ii,lambda_plot,0].cpu(), 
            title=f'{wavelengths[ii][lambda_plot]} nm')

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
lambda_batch_indices = np.arange(0, 2048, lambda_batch_size, dtype=np.uint64)

with torch.no_grad():
    for ii, y in enumerate(measurements_slice):
        for lambda_start in lambda_batch_indices:
            lambda_end = lambda_start + lambda_batch_size
            
            print(f'Channels: {lambda_start}--{lambda_end}')
            x_pinvnet[ii, lambda_start:lambda_end, ...] = pinvnet.reconstruct(
                                                y[lambda_start:lambda_end]
                                                )
            # denormalize
            x_pinvnet[ii, lambda_start:lambda_end, ...] *= pinvnet.prep.alpha[...,None]
            
del pinvnet
del denoiser
torch.cuda.empty_cache()

#%% Plot Pinv-Net
from spyrit.misc.disp import imagesc 

data_ind = 0
lambda_list = [500, 1500, 2000] 


wav_list = [wavelengths[data_ind][lamb] for lamb in lambda_list]

for lamb, wav in zip(lambda_list, wav_list):
    
    imagesc(x_pinvnet[data_ind,lamb,0].rot90(k=2).cpu(),
            colormap = wav,
            #colormap = wavelength_to_colormap(wav, gamma=0.8),
            title=f'Pinv-Net, {wav:.1f} nm'
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
#dcnet = recon.DCNet(meas_op, prep_op, Cov, device=device)

dcnet.eval()

# Reconstruct
x_dcnet = torch.zeros(reconstruct_size, device=device)
        
lambda_batch_size = 256
lambda_batch_indices = np.arange(0, 2048, lambda_batch_size, dtype=np.uint64)

with torch.no_grad():
    for ii, y in enumerate(measurements_slice):
        for lambda_start in lambda_batch_indices:
            lambda_end = lambda_start + lambda_batch_size
            
            print(f'Channels: {lambda_start}--{lambda_end}')
            x_dcnet[ii, lambda_start:lambda_end, ...] = dcnet.reconstruct(
                                                y[lambda_start:lambda_end]
                                                )            
            
            # denormalize
            x_dcnet[ii, lambda_start:lambda_end, ...] *= dcnet.prep.alpha[...,None]
            
del dcnet
del denoiser
torch.cuda.empty_cache()

#%% Plot DC-Net

from spyrit.misc.disp import imagesc 

data_ind = 0
lambda_list = [500, 1500, 2000] 

wav_list = [wavelengths[data_ind][lamb] for lamb in lambda_list]

for lamb, wav in zip(lambda_list, wav_list):
    
    imagesc(x_dcnet[data_ind,lamb,0].rot90(k=2).cpu(),
            colormap = wav,
            title=f'DC-Net, {wav:.1f} nm',
            gamma = .8
            )