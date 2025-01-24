# %%
# Imports
# --------------------------------------------------------------------
from pathlib import Path
from typing import OrderedDict

import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

import spyrit.core.meas as meas
import spyrit.core.noise as noise
import spyrit.core.prep as prep
import spyrit.core.recon as recon
import spyrit.core.nnet as nnet
import spyrit.core.train as train
import spyrit.external.drunet as drunet

import utility_dpgd as dpgd


# %%
# General
# --------------------------------------------------------------------
# Experimental data
image_folder = "data/images/"  # images for simulated measurements
model_folder = "model/"  # reconstruction models
stat_folder = "stat/"  # statistics
recon_folder = "recon/figure_3/"  # reconstructed images

# Full paths
image_folder_full = Path.cwd() / Path(image_folder)
model_folder_full = Path.cwd() / Path(model_folder)
stat_folder_full = Path.cwd() / Path(stat_folder)
recon_folder_full = Path.cwd() / Path(recon_folder)
recon_folder_full.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# %%
# Load images
# --------------------------------------------------------------------

img_size = 128  # image size

print("Loading image...")

image_path = image_folder_full / "cropped/ILSVRC2012_val_00000003_crop.JPEG"
x = torchvision.io.read_image(image_path, torchvision.io.ImageReadMode.GRAY)
# Resize image
x = torchvision.transforms.functional.resize(x, (img_size, img_size)).reshape(
    1, 1, img_size, img_size
)

# Select image
x = x.detach().clone()
x = x / 255
b, c, h, w = x.shape
print(f"Shape of input image: {x.shape}")

# save image as original
plt.imsave(recon_folder_full / "original.png", x[0, 0, :, :], cmap="gray")


# %%
# Simulate measurements for three image intensities
# --------------------------------------------------------------------
# Measurement parameters
alpha_list = [2, 10, 50]  # Poisson law parameter for noisy image acquisitions
n_alpha = len(alpha_list)
M = 128 * 128 // 4  # Number of measurements (here, 1/4 of the pixels)

# Measurement and noise operators
Ord_rec = torch.ones(img_size, img_size)
Ord_rec[:, img_size // 2 :] = 0
Ord_rec[img_size // 2 :, :] = 0

# Send to GPU if available
noise_model = noise.Poisson(alpha_list[0])
meas_op = meas.HadamSplit2d(h, M, Ord_rec, noise_model=noise_model, device=device)
prep_op = prep.UnsplitRescale(alpha_list[0])
rerange = prep.Rerange((0, 1), (-1, 1))
x = x.to(device)

# Measurement vectors
y_shape = torch.Size([n_alpha]) + meas_op(x).shape
y = torch.zeros(y_shape, device=device)

for ii, alpha in enumerate(alpha_list):
    torch.manual_seed(0)  # for reproducibility
    noise_model.alpha = alpha
    y[ii, ...] = meas_op(x)

reconstruct_size = torch.Size([n_alpha]) + x.shape


# %%
# Pinv
# ====================================================================
# Init
pinv = recon.PinvNet(meas_op, prep_op, use_fast_pinv=True)
# pinv.denoi = rerange
# Use GPU if available
# pinv = pinv.to(device)

# Reconstruct
x_pinv = torch.zeros(reconstruct_size, device=device)

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        pinv.prep.alpha = alpha
        x_pinv[ii] = pinv.reconstruct(y[ii, ...])
        filename = f"pinv_alpha_{alpha:02}.png"
        full_path = recon_folder_full / filename
        plt.imsave(
            full_path, x_pinv[ii, 0, 0, :, :].cpu().detach().numpy(), cmap="gray"
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
pinvnet = recon.PinvNet(meas_op, prep_op, denoiser, use_fast_pinv=True)
pinvnet.eval()

# Load net and use GPU if available
pinvnet = pinvnet.to(device)

# Reconstruct
x_pinvnet = torch.zeros(reconstruct_size, device=device)

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        pinvnet.prep.alpha = alpha
        x_pinvnet[ii, ...] = pinvnet.reconstruct(y[ii, ...])

        filename = f"pinvnet_alpha_{alpha:02}.png"
        full_path = recon_folder_full / filename
        plt.imsave(
            full_path, x_pinvnet[ii, 0, 0, :, :].cpu().detach().numpy(), cmap="gray"
        )


# %%
# LPGD
# ====================================================================
model_name = "lpgd_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_128_reg_1e-07_uit_3_sdec0-9_light.pth"
denoiser = OrderedDict(
    {"rerange": rerange, "denoi": nnet.Unet(), "rerange_inv": rerange.inverse()}
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

# Reconstruct and save
x_lpgd = torch.zeros(reconstruct_size, device=device)

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        lpgd.prep.alpha = alpha
        x_lpgd[ii, ...] = lpgd.reconstruct(y[ii, ...])

        # save
        filename = f"lpgd_alpha_{alpha:02}.png"
        full_path = recon_folder_full / filename
        plt.imsave(
            full_path, x_lpgd[ii, 0, 0, :, :].cpu().detach().numpy(), cmap="gray"
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

# Load covariance prior
cov_name = stat_folder_full / "Cov_8_{}x{}.pt".format(img_size, img_size)
Cov = torch.load(cov_name, weights_only=True).to(device)
# divide by 4 because the measurement covariance has been computed on images
# with values in [-1, 1] (total span 2) whereas our image is in [0, 1] (total
# span 1). The covariance is thus 2^2 = 4 times larger than expacted.
Cov /= 4

# Init
dcnet = recon.DCNet(meas_op, prep_op, Cov, denoiser)
dcnet.eval()

# Load net and use GPU if available
dcnet = dcnet.to(device)

# Reconstruct
x_dcnet = torch.zeros(reconstruct_size, device=device)

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        dcnet.prep.alpha = alpha
        x_dcnet[ii, ...] = dcnet.reconstruct(y[ii, ...])

        filename = f"dcnet_alpha_{alpha:02}.png"
        full_path = recon_folder_full / filename
        plt.imsave(
            full_path, x_dcnet[ii, 0, 0, :, :].cpu().detach().numpy(), cmap="gray"
        )


# %%
# Pinv - PnP
# ====================================================================
model_name = "drunet_gray.pth"
noise_levels = [115, 45, 20]  # noise levels from 0 to 255 for each alpha
denoiser = OrderedDict(
    {
        "rerange": rerange,
        "denoi": drunet.DRUNet(),
        "rerange_inv": rerange.inverse(),
    }
)
denoiser = nn.Sequential(denoiser)

# Initialize network
pinvpnp = recon.PinvNet(meas_op, prep_op, denoiser, use_fast_pinv=True)
pinvpnp.denoi.denoi.load_state_dict(
    torch.load(model_folder_full / model_name, weights_only=True), strict=False
)
pinvpnp.eval()
pinvpnp = pinvpnp.to(device)

# Reconstruct and save
x_pinvpnp = torch.zeros(reconstruct_size, device=device)

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):

        # set noise level for measurement operator and PnP denoiser
        pinvpnp.prep.alpha = alpha
        nu = noise_levels[ii]
        pinvpnp.denoi.denoi.set_noise_level(nu)
        x_pinvpnp[ii, ...] = pinvpnp.reconstruct(y[ii, ...])

        # save
        filename = f"pinv_pnp_alpha_{alpha:02}_nu_{nu:03}.png"
        full_path = recon_folder_full / filename
        plt.imsave(
            full_path, x_pinvpnp[ii, 0, 0, :, :].cpu().detach().numpy(), cmap="gray"
        )


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
gamma = 1 / img_size**2
max_iter = 101
mu_list = [6000, 3500, 1500]
crit_norm = 1e-4

# Init
dpgdnet = dpgd.DualPGD(meas_op, prep_op, denoi, gamma, mu_list[0], max_iter, crit_norm)
dpgdnet = dpgdnet.to(device)
x_dpgd = torch.zeros(reconstruct_size, device=device)

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        dpgdnet.prep.alpha = alpha
        dpgdnet.mu = mu_list[ii]
        x_dpgd[ii, ...] = dpgdnet.reconstruct(y[ii, ...])

        # save
        filename = f"dpgd_alpha_{alpha:02}.png"
        full_path = recon_folder_full / filename
        plt.imsave(
            full_path, x_dpgd[ii, 0, 0, :, :].cpu().detach().numpy(), cmap="gray"
        )

# %%
