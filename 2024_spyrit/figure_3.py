# %%
# Imports
# --------------------------------------------------------------------
from pathlib import Path

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import spyrit.core.meas as meas
import spyrit.core.noise as noise
import spyrit.core.prep as prep
import spyrit.core.recon as recon
import spyrit.core.nnet as nnet
import spyrit.core.train as train
import spyrit.misc.statistics as stats
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
# crop to desired size, set to black and white, normalize
transform = stats.transform_gray_norm(img_size)

# define dataset and dataloader. `image_folder_full` should contain
# a class folder with the images
dataset = torchvision.datasets.ImageFolder(image_folder_full, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# select the image
x, _ = next(iter(dataloader))
x = x[0]
c, h, w = x.shape
print("Image shape:", x.shape)

x_plot = x.view(-1, h, h).cpu().numpy()
# save image as original
plt.imsave(recon_folder_full / "original.png", x_plot[0, :, :], cmap="gray")


# %%
# Simulate measurements for three image intensities
# --------------------------------------------------------------------
# Measurement parameters
alpha_list = [2, 10, 50]  # Poisson law parameter for noisy image acquisitions
n_alpha = len(alpha_list)
M = 128 * 128 // 4  # Number of measurements (here, 1/4 of the pixels)

# Measurement and noise operators
Ord_rec = np.ones((img_size, img_size))
Ord_rec[:, img_size // 2 :] = 0
Ord_rec[img_size // 2 :, :] = 0

meas_op = meas.HadamSplit(M, h, torch.from_numpy(Ord_rec))
noise_op = noise.Poisson(meas_op, alpha_list[0])
prep_op = prep.SplitPoisson(2, meas_op)

# Vectorized image
x = x.view(1, h * w)

# Measurement vectors
y = torch.zeros(n_alpha, 2 * M)
for ii, alpha in enumerate(alpha_list):
    torch.manual_seed(0)  # for reproducibility
    noise_op.alpha = alpha
    y[ii, :] = noise_op(x)

# Send to GPU if available
y = y.to(device)


# %%
# Pinv
# ====================================================================
# Init
pinv = recon.PinvNet(noise_op, prep_op)

# Use GPU if available
pinv = pinv.to(device)

# Reconstruct
x_pinv = torch.zeros(n_alpha, 1, img_size, img_size)

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        pinv.prep.alpha = alpha
        x_pinv[ii] = pinv.reconstruct(
            y[ii : ii + 1, :]
        )  # NB: shape of measurement is (1,8192)

        filename = f"pinv_alpha_{alpha:02}.png"
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_pinv[ii, 0].cpu().detach().numpy(), cmap="gray")  #


# %%
# Pinv-Net
# ====================================================================
model_name = "pinv-net_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07_retrained_light.pth"

# Init
pinvnet = recon.PinvNet(noise_op, prep_op, nnet.Unet())
pinvnet.eval()

# Load net and use GPU if available
train.load_net(model_folder_full / model_name, pinvnet, device, False)
pinvnet = pinvnet.to(device)

# Reconstruct
x_pinvnet = torch.zeros(n_alpha, 1, img_size, img_size)

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        pinvnet.prep.alpha = alpha
        x_pinvnet[ii] = pinvnet.reconstruct(
            y[ii : ii + 1, :]
        )  # NB: shape of measurement is (1,8192)

        filename = f"pinvnet_alpha_{alpha:02}.png"
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_pinvnet[ii, 0].cpu().detach().numpy(), cmap="gray")  #


# %%
# LPGD
# ====================================================================
model_name = "lpgd_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_128_reg_1e-07_uit_3_sdec0-9_light.pth"

# Initialize network
denoi = nnet.Unet()
lpgd = recon.LearnedPGD(noise_op, prep_op, denoi, step_decay=0.9)
lpgd.eval()

# load net and use GPU if available
train.load_net(model_folder_full / model_name, lpgd, device, False)
lpgd = lpgd.to(device)

# Reconstruct and save
x_lpgd = torch.zeros(1, 1, img_size, img_size)

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        lpgd.prep.alpha = alpha
        x_lpgd = lpgd.reconstruct(
            y[ii : ii + 1, :]
        )  # NB: shape of measurement is (1,8192) as expected

        # save
        filename = f"lpgd_alpha_{alpha:02}.png"
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_lpgd[0, 0].cpu().detach().numpy(), cmap="gray")


# %%
# DC-Net
# ====================================================================
model_name = "dc-net_unet_imagenet_rect_N0_10_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07_light.pth"
cov_name = stat_folder_full / "Cov_8_{}x{}.npy".format(img_size, img_size)

# Load covariance prior
Cov = torch.from_numpy(np.load(cov_name))

# Init
denoi = nnet.Unet()  # torch.nn.Identity()
dcnet = recon.DCNet(noise_op, prep_op, Cov, denoi)
dcnet.eval()

# Load net and use GPU if available
train.load_net(model_folder_full / model_name, dcnet, device, False)
dcnet = dcnet.to(device)

# Reconstruct
x_dcnet = torch.zeros(n_alpha, 1, img_size, img_size)

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        dcnet.prep.alpha = alpha
        x_dcnet[ii] = dcnet.reconstruct(
            y[ii : ii + 1, :]
        )  # NB: shape of measurement is (1,8192) as expected

        filename = f"dcnet_alpha_{alpha:02}.png"
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_dcnet[ii, 0].cpu().detach().numpy(), cmap="gray")  #


# %%
# Pinv - PnP
# ====================================================================
model_name = "drunet_gray.pth"
noise_levels = [115, 45, 20]  # noise levels from 0 to 255 for each alpha

# Initialize network
denoi = drunet.DRUNet()
pinvnet = recon.PinvNet(noise_op, prep_op, denoi)
pinvnet.eval()

# load_net(model_folder_full / model_name, pinvnet, device, False)
pinvnet.denoi.load_state_dict(torch.load(model_folder_full / model_name), strict=False)
pinvnet.denoi.eval()
pinvnet = pinvnet.to(device)

# Reconstruct and save
x_pinvnet = torch.zeros(1, 1, img_size, img_size)

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):

        # set noise level for measurement operator and PnP denoiser
        pinvnet.prep.alpha = alpha
        nu = noise_levels[ii]
        pinvnet.denoi.set_noise_level(nu)
        x_pinvnet = pinvnet.reconstruct(y[ii : ii + 1, :])

        # save
        filename = f"pinv_pnp_alpha_{alpha:02}_nu_{nu:03}.png"
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_pinvnet[0, 0].cpu().detach().numpy(), cmap="gray")


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
dpgdnet = dpgd.DualPGD(noise_op, prep_op, denoi, gamma, mu_list[0], max_iter, crit_norm)
x_dpgd = torch.zeros(1, 1, img_size, img_size)

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        dpgdnet.prep.alpha = alpha
        dpgdnet.mu = mu_list[ii]
        x_dpgd = dpgdnet.reconstruct(
            y[ii : ii + 1, :]
        )  # NB: shape of measurement is (1,8192) as expected

        # save
        filename = f"dpgd_alpha_{alpha:02}.png"
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_dpgd[0, 0].cpu().detach().numpy(), cmap="gray")
