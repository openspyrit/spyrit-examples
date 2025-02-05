# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:10:10 2025

@author: ducros
"""
# %%
# Imports
# --------------------------------------------------------------------
from pathlib import Path
from typing import OrderedDict

import torch
import torch.nn as nn

import spyrit.core.meas as meas
import spyrit.core.prep as prep
import spyrit.core.noise as noise
import spyrit.core.recon as recon
import spyrit.misc.statistics as stats
import spyrit.external.drunet as drunet

import utility_dpgd as dpgd


# %%
# General Parameters
# --------------------------------------------------------------------
img_size = 128  # image size
batch_size = 4
n_batches = 2  # iterate over how many batches to get statistical data

# Experimental
val_folder = "data/ILSVRC2012/val/"  # used for statistical analysis
model_folder = "model/"  # reconstruction models
stat_folder = "stat/"  # statistics
recon_folder = "recon/table_1/"  # table output

# Full paths
val_folder_full = Path.cwd() / Path(val_folder)
model_folder_full = Path.cwd() / Path(model_folder)
stat_folder_full = Path.cwd() / Path(stat_folder)
recon_folder_full = Path.cwd() / Path(recon_folder)
val_folder_full.mkdir(parents=True, exist_ok=True)
recon_folder_full.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %%
# Load images
# --------------------------------------------------------------------
# /!\ spyrit v3 works with images in [0,1]. Therefore, we use normalize=False

dataloader = stats.data_loaders_ImageNet(
    val_folder_full, val_folder_full, img_size, batch_size, normalize=False
)["val"]


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

noise_op = noise.Poisson(alpha_list[0])
meas_op = meas.HadamSplit2d(img_size, M, Ord_rec, noise_model=noise_op, device=device)
prep_op = prep.UnsplitRescale(alpha_list[0])
rerange = prep.Rerange((0, 1), (-1, 1))

# %%
# Pinv - PnP
# ====================================================================
model_name = "drunet_gray.pth"
#noise_levels = [115, 50, 20]  # noise levels from 0 to 255 for each alpha
noise_levels = [
    [70, 80, 90, 95, 100, 105, 110],    # 2 photons
    [30, 35, 40, 45,  50,  55,  60],    # 10 photons
    [10, 15, 20, 25, 30, 35],           # 50 photons
]

# Initialize network
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
print("\nPinv-PnP reconstruction metrics")

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        
        metric_file = recon_folder_full / 'table_S1.tex'
        
        with open(metric_file, 'a') as f:
            
            f.write('\n')
            f.write(f'$\\alpha={alpha}$ & & \\\\ \n')
            
            # Set alpha for the simulation of the measurements
            # pinvpnp.acqu_modules.acqu.noise_model.alpha = alpha # works too
            pinvpnp.acqu.noise_model.alpha = alpha
            
            # Set alpha for reconstruction
            # pinvpnp.recon_modules.prep.alpha = alpha # works too
            pinvpnp.prep.alpha = alpha 
            
            for nu in noise_levels[ii]:
                
                torch.manual_seed(0)  # for reproducibility
                
                print(f"For alpha={alpha} and nu={nu}")
                pinvpnp.denoi.denoi.set_noise_level(nu)
        
                # PSNR
                mean_psnr, var_psnr = stats.stat_psnr(pinvpnp, dataloader, device, 
                                                      num_batchs=n_batches, 
                                                      img_dyn=1.0)
                mean_psnr = mean_psnr.cpu().numpy()
                std_psnr = torch.sqrt(var_psnr).cpu().numpy()
                print(f"psnr = {mean_psnr:.2f} +/- {std_psnr:.2f} dB")
                
                # SSIM
                mean_ssim, var_ssim = stats.stat_ssim(pinvpnp, dataloader, 
                                                      device, num_batchs=n_batches, 
                                                      img_dyn=1.0)
                mean_ssim = mean_ssim.cpu().numpy()
                std_ssim = torch.sqrt(var_ssim).cpu().numpy()
                print(f"ssim = {mean_ssim:.3f} +/- {std_ssim:.3f}")
                
                # sample list
                f.write(f'$\\nu={nu}$ & {mean_psnr:.2f} ({std_psnr:.2f}) & {mean_ssim:.3f} ({std_ssim:.3f}) \\\\ \n')

del pinvpnp
del denoiser

# %% DPGD-PnP
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
crit_norm = 1e-4
mu_list = [
    [2000, 4000, 4500, 5000, 5500, 6000, 8000, 10000],  # 2 photons
    [1000, 2000, 2500, 3000, 3500, 4000, 5000],         # 10 photons
    [500, 1000, 1200, 1500, 1800, 2000, 3000, 4000],    # 50 photons
]

# Init
dpgdnet = dpgd.DualPGD(meas_op, prep_op, denoi, gamma, mu_list[0], max_iter, crit_norm)
dpgdnet = dpgdnet.to(device)

print("\nDPGD-PnP reconstruction metrics")

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        
        metric_file = recon_folder_full / 'table_S2.tex'
        
        with open(metric_file, 'a') as f:
            
            f.write('\n')
            f.write(f'$\\alpha={alpha}$ & & \\\\ \n')
            
            # Set alpha for the simulation of the measurements
            # dpgdnet.acqu_modules.acqu.noise_model.alpha = alpha # works too
            dpgdnet.acqu.noise_model.alpha = alpha
            
            # Set alpha for reconstruction
            # dpgdnet.recon_modules.prep.alpha = alpha # works too
            dpgdnet.prep.alpha = alpha 
            
            for mu in mu_list[ii]:
                
                torch.manual_seed(0)  # for reproducibility
                
                dpgdnet.mu = mu
                
                print(f"For alpha={alpha} and mu={mu}")
                
                # PSNR
                mean_psnr, var_psnr = stats.stat_psnr(dpgdnet, dataloader, 
                                                      device, num_batchs=n_batches, 
                                                      img_dyn=1.0)
                mean_psnr = mean_psnr.cpu().numpy()
                std_psnr = torch.sqrt(var_psnr).cpu().numpy()
                print(f"psnr = {mean_psnr:.2f} +/- {std_psnr:.2f} dB")
                
                # SSIM
                mean_ssim, var_ssim = stats.stat_ssim(dpgdnet, dataloader, 
                                                      device, num_batchs=n_batches, 
                                                      img_dyn=1.0)
                mean_ssim = mean_ssim.cpu().numpy()
                std_ssim = torch.sqrt(var_ssim).cpu().numpy()
                print(f"ssim = {mean_ssim:.3f} +/- {std_ssim:.3f}")
                
                # sample list 
                f.write(f'$\\mu={mu}$ & {mean_psnr:.2f} ({std_psnr:.2f}) & {mean_ssim:.3f} ({std_ssim:.3f}) \\\\')

del dpgdnet
del denoi

