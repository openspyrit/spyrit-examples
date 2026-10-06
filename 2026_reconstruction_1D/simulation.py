#%%

# %matplotlib qt6
import numpy as np
from spyrit.misc.statistics import data_loaders_stl10
import matplotlib.pyplot as plt
import torchvision

from spyrit.core.prep import Rerange, Unsplit, Identity, UnsplitRescale, UnsplitRescaleEstim, RescaleEstim
from spyrit.core.noise import PoissonGaussian
from spyrit.core.recon import OrderedDict, TikhoNet, PinvNet
from spyrit.core.meas import LinearSplit, Linear
import torch.nn as nn
from tools import hadamard

from pathlib import Path

import json
import ast

import imageio.v3 as iio

from PIL import Image
import torch
from spyrit.misc.walsh_hadamard import walsh_matrix, walsh_matrix_2d
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from spyrit.misc.disp import imagesc



#%% Obtain STL10 images

img_size = 128
batch_size = 25
data_root = './data'
dataloaders = data_loaders_stl10(data_root, img_size=img_size, batch_size=batch_size, seed=7, shuffle=False, download=True, normalize=False)

print(dataloaders.keys())
images, _ = next(iter(dataloaders['train']))
print(f"Ground-truth images: {images.shape}")

# %% PARAMETERS

alpha = 50 # level a which lines are present
K = 128 # if you want to downsample
N = 128

# Noise parameters - as in experimentation
gamma =  0.115   # 0.1123

mu = np.load("data/mu_dark_image_binned_x3.npy")
var = np.load("data/var_dark_image_binned_x3.npy")

mudark = mu.mean()  # 254.9
std_mudark = mu.std()

sigdark = np.sqrt(var.mean())
std_sigdark = np.sqrt(var.std())
 
mu_dark_image = torch.normal( mean=mudark, std=std_mudark, size=(N,1)).to(device).expand(-1, K*2) 
sig_dark_image = torch.normal( mean=sigdark, std=std_sigdark, size=(N,1)).to(device).expand(-1, K*2)   # don't change size of first dimension, expand (repeat) P times

#%% SIMULATE MEASUREMENTS

H  = walsh_matrix(N)
H = torch.from_numpy(H).to(device)

meas_op = LinearSplit(H[0:K, :], noise_model = PoissonGaussian(alpha=alpha, mu=mu_dark_image, g=gamma, sigma=sig_dark_image), device = device)

x = images.to(device)
y = meas_op(images.to(device))
y = y.to(device)

#%% PSEUDOINVERSE
# load net
from spyrit.core.nnet import Unet
from spyrit.core.train import load_net

title = f'model/TIKHO_UNET_128x128_K=128_alpha100_20epochs.pth'
denoiser = torch.nn.Sequential(OrderedDict({"denoi": Unet()}))
load_net(title, denoiser, device, False)

i_plot = 5

prep_op = Unsplit()
model = PinvNet(meas_op, prep=prep_op, denoi = denoiser, store_H_pinv = True, device = device) 
model.eval()
model = model.to(device)


with torch.no_grad():
    x_hat = model.reconstruct(y/gamma)
    x_pinv = x_hat.detach().cpu().numpy().squeeze()
del model 

plt.imshow(x_pinv[i_plot, :,:])
plt.title("PSEUDOINVERSE")
plt.colorbar()


#%% TIKHONOV

# adding deviations
# sigma_value = 20
# sigdarke = sigdark*(1)
# mudarke = mudark*(1)

# # to set Gamma as a constant along diagonals
# mu_gamma = 1       # mean of diagonal values
# var_gamma = 0 # variance of diagonal values

# std_gamma = var_gamma ** 0.5

# diag_values = (
#     mu_gamma
#     + std_gamma * torch.randn(
#         batch_size, 1, N, N,
#         device=device
#     )
# )

# Gamma_const = torch.diag_embed(diag_values)

# normalisation parameter estimated from pseudoinverse
alpha_est = x_pinv.reshape(batch_size, N*N).max(-1)
alpha_est = torch.from_numpy(alpha_est).to(device)

# Gamma calculation (measurement covariance)

z = (y[..., 0::2] + y[..., 1::2])
cov_meas = gamma * (z - 2*mu_dark_image[:,0:128]) + 2 * sig_dark_image[:,0:128]**2   
cov_meas = cov_meas.to(device)
norm = gamma*alpha_est[:,None, None, None]
cov_meas = cov_meas / norm**2
Gamma_image = torch.diag_embed(cov_meas)

# Gamma calculation (measurement covariance)
# z = (y[..., 0::2] + y[..., 1::2])
# cov_meas = gamma * (z - 2*mudarke) + 2 * sigdarke**2   
# cov_meas = cov_meas.to(device)
# norm = gamma*alpha_est[:,None, None, None]
# cov_meas = cov_meas / norm**2
# Gamma_const = torch.diag_embed(cov_meas)

# load Sigma
image_cov = np.load('data/Cov_1_128x128.npy')
Sigma = torch.from_numpy(image_cov).to(device)

# load net
from spyrit.core.nnet import Unet
from spyrit.core.train import load_net

title = f'model/TIKHO_UNET_128x128_K=128_alpha100_20epochs.pth'
# title = f'model/tikho_unet_128x128_K=128_alpha2600_20epochs.pth'
# title = f'model/tikho-net_unet_imagenet_ph_250_exp_N_512_M_128_epo_20_lr_0.001_sss_10_sdr_0.5_bs_20_reg_1e-07.pth'
# title = f'model/LINES_tikho_unet_128x128_K=128_alpha20_20epochs.pth'

# rerange = Rerange((0, 1), (-1, 1))
# denoiser = OrderedDict({"rerange": rerange, "denoi": Unet(), "rerange_inv": rerange.inverse()})
# denoiser = nn.Sequential(denoiser)

# load_net(title, denoiser, device, False)


denoiser = torch.nn.Sequential(OrderedDict({"denoi": Unet()}))
load_net(title, denoiser, device, False)

n = nn.Upsample(scale_factor = 512/128)
prep_op = UnsplitRescale(alpha_est[:,None, None, None])
model = TikhoNet(meas_op, prep = prep_op , denoi=denoiser, sigma = Sigma,  device = device) 
model.eval()
model = model.to(device)

with torch.no_grad():
    m_t = model.prep(y/gamma)
    x_hat = model.tikho(m_t, Gamma_image)

    # upsample to 512x512 
    x_up = n(x_hat)

    x_unet  = model.denoi(x_up)
    x_tiko = x_up.detach().cpu().numpy().squeeze()
    x_tikonet = x_unet.detach().cpu().numpy().squeeze()
    # x_tiko_const = x_const.detach().cpu().numpy().squeeze()

del model

x_tiko_denorm = x_tiko*alpha_est.detach().cpu().numpy().squeeze()[:,None, None]
x_tikonet_denorm = x_tikonet*alpha_est.detach().cpu().numpy().squeeze()[:,None, None]


# x_tiko_denorm_image= x_tiko_image*alpha_est.detach().cpu().numpy().squeeze()[:,None, None]
# x_tiko_denorm_const = x_tiko_const*alpha_est.detach().cpu().numpy().squeeze()[:,None, None]
# x_tiko_denorm_none = x_tiko_none*alpha_est.detach().cpu().numpy().squeeze()[:,None, None]

# plt.imshow(x_tiko_denorm[i_plot, :,:])
# plt.title("TIKHONOV")
# plt.colorbar()


#%% PLOT
import matplotlib.pyplot as plt

# User-defined batch indices
i_plot = 3
j_plot = 14

indices = [i_plot, j_plot]

fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(14, 7)
)

for row, idx in enumerate(indices):

    # Ground Truth
    gt = x[idx, :].detach().cpu().numpy().squeeze() * alpha

   # Pseudoinverse reconstruction
    pinv = x_pinv[idx, :]

    # Tikhonov reconstructions
    tiko = x_tiko_denorm[idx, :]
    tikonet = x_tikonet_denorm[idx,:]


    # tiko_const = x_tiko_denorm_const[idx, :]
    # tiko_image = x_tiko_denorm_image[idx, :]
    # tiko_none = x_tiko_denorm_none[idx,:]
    # -------------------------
    # Ground Truth
    # -------------------------
    im0 = axes[row, 0].imshow(gt)
    axes[row, 0].set_title("GROUND TRUTH")

    fig.colorbar(
        im0,
        ax=axes[row, 0],
        fraction=0.045,
        pad=0.02,
        shrink=0.85
    )

    # -------------------------
    # Pseudoinverse
    # -------------------------
    im1 = axes[row, 1].imshow(pinv)
    axes[row, 1].set_title("PSEUDOINVERSE")

    fig.colorbar(
        im1,
        ax=axes[row, 1],
        fraction=0.045,
        pad=0.02,
        shrink=0.85
    )

    # im1 = axes[row, 1].imshow(tiko_none)
    # axes[row, 1].set_title(rf"TIKHONOV without estimated $\sigma_{{d}}$ and $\mu_{{d}}$")
    # axes[row, 1].axis("off")
    # fig.colorbar(
    #     im1,
    #     ax=axes[row, 1],
    #     fraction=0.045,
    #     pad=0.02,
    #     shrink=0.85
    # )


    

    # -------------------------
    # Tikhonov - estimated Sigma
    # -------------------------
    im2 = axes[row, 2].imshow(tiko)
    axes[row, 2].set_title(
        rf"TIKHONOV with $\Gamma$ using $\sigma_{{d,image}}$ and $\mu_{{d,image}}$"
    )
   
    fig.colorbar(
        im2,
        ax=axes[row, 2],
        fraction=0.045,
        pad=0.02,
        shrink=0.85
    )

    # -------------------------
    # Tikhonov - constant Sigma
    # -------------------------
    im3 = axes[row, 3].imshow(tikonet)
    axes[row, 3].set_title(
        rf"DENOISED with UNet()"
    )
    
    fig.colorbar(
        im3,
        ax=axes[row, 3],
        fraction=0.045,
        pad=0.02,
        shrink=0.85
    )

# Global title
fig.suptitle(
    rf"$\alpha = {alpha}$",
    fontsize=16
)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()
# %%

i = 100
plt.imshow(Sigma[0,0,i,:,:].detach().cpu())
plt.colorbar()
# %%
plt.imshow(cov_meas[1,0,:,:].detach().cpu())
plt.colorbar()
# %% 
