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
import matplotlib.pyplot as plt

import spyrit.core.meas as meas
import spyrit.core.prep as prep
import spyrit.core.recon as recon
import spyrit.core.nnet as nnet
import spyrit.core.train as train
import spyrit.misc.sampling as samp
import spyrit.external.drunet as drunet

import utility_dpgd as dpgd
# %%
# General
# --------------------------------------------------------------------
# Experimental data
data_folder = "data/"  # measurements
model_folder = "model/"  # reconstruction models
stat_folder = "stat/"  # statistics
recon_folder = "recon/supplemental_figure_S6/"  # reconstructed images

# Full paths
data_folder_full = Path.cwd() / Path(data_folder)
model_folder_full = Path.cwd() / Path(model_folder)
stat_folder_full = Path.cwd() / Path(stat_folder)
recon_folder_full = Path.cwd() / Path(recon_folder)
recon_folder_full.mkdir(parents=True, exist_ok=True)

# choose by name which experimental data to use
data_title = [
    "zoom_x12_usaf_group5",
    "zoom_x12_starsector",
    "tomato_slice_2_zoomx12",
    "tomato_slice_2_zoomx2",
]
suffix = {"data": "_spectraldata.npz", "metadata": "_metadata.json"}
n_meas = len(data_title)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
img_size = 128
subsampling_factor = 2


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
prep_op = prep.UnsplitRescaleEstim(meas_op, use_fast_pinv=True)
rerange = prep.Rerange((0, 1), (-1, 1))

# %%
# Load experimental data and prepare it for reconstruction
# ====================================================================
print("Looking for data in", data_folder_full)

# Collect data in numpy
exp_data = [
    np.load(data_folder_full / (title + suffix["data"]))["spectral_data"]
    for title in data_title
]
# Collect metadata
patterns = [[] for _ in range(n_meas)]
wavelengths = [[] for _ in range(n_meas)]
for ii, title in enumerate(data_title):
    file = open(data_folder_full / (title + suffix["metadata"]), "r")
    json_metadata = json.load(file)[3]
    file.close()
    patterns[ii] = ast.literal_eval(json_metadata["patterns"])
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
# Reconstruct using a single spectral slice
# ====================================================================
lambda_select = 579.0970
measurements_slice = [np.zeros((2 * M, 1)) for _ in range(n_meas)]

# select the measurements in the right spectral slice
for i in range(n_meas):

    lambda_index = wavelengths[i].index(lambda_select)
    # take only the first 2*M measurements of the right wavelength
    measurements_slice[i] = measurements[i][: 2 * M, lambda_index].reshape(
        (1, 1, 2 * M)
    )
    measurements_slice[i] = torch.from_numpy(measurements_slice[i]).to(
        device, dtype=torch.float32
    )
reconstruct_size = torch.Size([n_meas]) + (1, 1, img_size, img_size)

# %%
# Pinv
# ====================================================================
# Init
pinv = recon.PinvNet(meas_op, prep_op)

# Use GPU if available
pinv = pinv.to(device)

with torch.no_grad():
    for ii, y in enumerate(measurements_slice):
        x_pinv = pinv.reconstruct(y)#[0]

        filename = f"{data_title[ii]}_{M}_{img_size}_pinv.png"
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_pinv[0, 0, :, :].cpu().detach().numpy(), cmap="gray")


# %%
# Pinv-Net
# ====================================================================
model_name = "pinv-net_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07_retrained_light.pth"

# /!\ spyrit v3 works with images in [0,1], but denoisers were trained for 
# images in [-1,1]
denoiser = OrderedDict(
    {"rerange": rerange, 
     "denoi": nnet.Unet(), 
     "rerange_inv": rerange.inverse()}
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

# Reconstruct
with torch.no_grad():
    for ii, y in enumerate(measurements_slice):
        #pinvnet.prep.set_expe()
        x_pinvnet = pinvnet.reconstruct(y)#[0]

        filename = f"{data_title[ii]}_{M}_{img_size}_pinvnet_unet.png"
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_pinvnet[0, 0, :, :].cpu().detach().numpy(), cmap="gray")


# %%
# LPGD
# ====================================================================
model_name = "lpgd_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_128_reg_1e-07_uit_3_sdec0-9_light.pth"

# /!\ spyrit v3 works with images in [0,1], but denoisers were trained for 
# images in [-1,1]
denoiser = OrderedDict(
    {"rerange": rerange, 
     "denoi": nnet.Unet(), 
     "rerange_inv": rerange.inverse()}
)
denoiser = nn.Sequential(denoiser)
# this function loads the model into the '.denoi' key present in the second
# argument. It fails if it does not find the '.denoi' key.
train.load_net(model_folder_full / model_name, denoiser, device, False)

# Initialize network
lpgd = recon.LearnedPGD(meas_op, prep_op, denoiser, step_decay=0.9)
lpgd.eval()

# load net and use GPU if available
lpgd = lpgd.to(device)  

with torch.no_grad():
    for ii, y in enumerate(measurements_slice):
        #lpgd.prep.set_expe()
        x_lpgd = lpgd.reconstruct(y)

        filename = f"{data_title[ii]}_{M}_{img_size}_lpgd_unet.png"
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_lpgd[0, 0, :, :].cpu().detach().numpy(), cmap="gray")


# %%
# DC-Net
# ====================================================================
model_name = "dc-net_unet_imagenet_rect_N0_10_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07_light.pth"
cov_name = stat_folder_full / "Cov_8_{}x{}.pt".format(img_size, img_size)

# /!\ spyrit v3 works with images in [0,1], but denoisers were trained for 
# images in [-1,1]
denoiser = OrderedDict(
    {"rerange": rerange, 
     "denoi": nnet.Unet(), 
     "rerange_inv": rerange.inverse()}
)
denoiser = nn.Sequential(denoiser)
# this function loads the model into the '.denoi' key present in the second
# argument. It fails if it does not find the '.denoi' key.
train.load_net(model_folder_full / model_name, denoiser, device, False)

# Load covariance prior
Cov = torch.load(cov_name, weights_only=True).to(device)
# divide by 4 because the measurement covariance has been computed on images
# with values in [-1, 1] (total span 2) whereas our image is in [0, 1] (total
# span 1). The covariance is thus 2^2 = 4 times larger than expacted.
Cov /= 4

# Init
dcnet = recon.DCNet(meas_op, prep_op, Cov, denoiser, device=device)
dcnet.eval()

# Reconstruct
x_dcnet = torch.zeros(reconstruct_size, device=device)

with torch.no_grad():
    for ii, y in enumerate(measurements_slice):
        x_dcnet = dcnet.reconstruct(y)

        filename = f"{data_title[ii]}_{M}_{img_size}_dc-net_unet.png"
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_dcnet[0, 0, :, :].cpu().detach().numpy(), cmap="gray")


# %%
# Pinv - PnP
# ====================================================================
model_name = "drunet_gray.pth"

# in order: USAF target, starsector, tomato x12, tomato x2
nu_list = [30, 45, 40, 35]  # noise levels for each label

# /!\ spyrit v3 works with images in [0,1], but denoisers were trained for 
# images in [-1,1]
denoiser = OrderedDict(
    {
     # No rerange() needed with normalize=False
     #"rerange": rerange,
     "denoi": drunet.DRUNet(normalize=False),
     # No rerange.inverse() here as DRUNet works for images in [0,1] 
     #"rerange_inv": rerange.inverse(), 
     }
)
denoiser = nn.Sequential(denoiser)

# Initialize network
pinvpnp = recon.PinvNet(meas_op, prep_op, denoiser, device=device)
pinvpnp.denoi.denoi.load_state_dict(
    torch.load(model_folder_full / model_name, weights_only=True), strict=False
)
pinvpnp.eval()

# Reconstruct and save
x_pinvpnp = torch.zeros(reconstruct_size, device=device)

with torch.no_grad():
    for ii, y in enumerate(measurements_slice):
        nu = nu_list[ii]
        pinvpnp.denoi.denoi.set_noise_level(nu)
        x_pinvpnp = pinvpnp.reconstruct(y)

        filename = f"{data_title[ii]}_{M}_{img_size}_pinv-net_drunet_nlevel_{nu}.png"
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_pinvpnp[0, 0, :, :].cpu().detach().numpy(), cmap="gray")


# %%
# DPGD-PnP
# ====================================================================
# load denoiser
n_channel, n_feature, n_layer = 1, 100, 20
model_name = "DFBNet_l1_patchsize=50_varnoise0.1_feat_100_layers_20.pth"
denoi = dpgd.load_model(
    pth=(model_folder_full / model_name).as_posix(),
    n_ch=n_channel,
    features=n_feature,
    num_of_layers=n_layer,
)

denoi.module.update_lip((1, 50, 50))
denoi.eval()

# Reconstruction hyperparameters

# in order: USAF target, starsector, tomato x12, tomato x2
mu_list = [2000, 4000, 3000, 4000]  # noise levels for each label
gamma = 1 / img_size**2
max_iter = 101
crit_norm = 1e-4

# Init
dpgdnet = dpgd.DualPGD(meas_op, prep_op, denoi, gamma, mu_list[0], max_iter, crit_norm)

with torch.no_grad():
    for ii, y in enumerate(measurements_slice):
        dpgdnet.mu = mu_list[ii]
        x_dpgd = dpgdnet.reconstruct(y)

        # save
        filename = f"{data_title[ii]}_{M}_{img_size}_dfb-net_dfb_mu_{mu_list[ii]}.png"
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_dpgd[0, 0, :, :].cpu().detach().numpy(), cmap="gray")