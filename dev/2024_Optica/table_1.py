# %% 
# Imports
# --------------------------------------------------------------------
import os
from pathlib import Path
import sys
import math
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import spyrit.core.recon as recon
import spyrit.external.drunet as drunet

import aux_functions as aux
# %% 
# General Parameters
# --------------------------------------------------------------------
img_size = 128 # image size
batch_size = 128
n_batches = 3 # iterate over how many batches to get statistical data

# Experimental data
val_folder = 'data/ImageNet_validation/val(1)/val' # used for statistical analysis
# val_folder = r'C:\Users\phan\Downloads\images\Images'
model_folder = 'model/'             # reconstruction models
stat_folder  = 'stat/'              # statistics
recon_folder = 'recon/table_1/'     # table output

# Full paths
val_folder_full = Path.cwd() / Path(val_folder)
model_folder_full = Path.cwd() / Path(model_folder)
stat_folder_full  = Path.cwd() / Path(stat_folder)
recon_folder_full = Path.cwd() / Path(recon_folder)
val_folder_full.mkdir(parents=True, exist_ok=True)
recon_folder_full.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %% 
# Load images
# --------------------------------------------------------------------
import spyrit.misc.statistics as stats

dataloader = stats.data_loaders_ImageNet(val_folder_full, val_folder_full, 
    img_size, batch_size)['val'] 
metrics_eval = ['nrmse', 'ssim'] 


# %% 
# Simulate measurements for three image intensities
# --------------------------------------------------------------------
from spyrit.core.meas import HadamSplit
from spyrit.core.noise import Poisson
from spyrit.core.prep import SplitPoisson

# Measurement parameters
alpha_list = [2, 10, 50] # Poisson law parameter for noisy image acquisitions
n_alpha = len(alpha_list)
M = 128 * 128 // 4  # Number of measurements (here, 1/4 of the pixels)

# Measurement and noise operators
Ord_rec = np.ones((img_size, img_size))
Ord_rec[:,img_size//2:] = 0
Ord_rec[img_size//2:,:] = 0

meas_op = HadamSplit(M, img_size, torch.from_numpy(Ord_rec))
noise_op = Poisson(meas_op, alpha_list[0])
prep_op = SplitPoisson(2, meas_op)


# %% 
# Pinv
# ====================================================================
from spyrit.core.recon import PinvNet

# Init
pinv = PinvNet(noise_op, prep_op)

# Use GPU if available
pinv = pinv.to(device)

print("\nPinv reconstruction metrics")

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        print(f"For {alpha=}")
        pinv.prep.alpha = alpha
        pinv.acqu.alpha = alpha
        results = aux.eval_model_metrics_batch_cum(pinv, dataloader, device, metrics_eval[0:1], n_batches)
        print(results)
        

# %% 
# Pinv-Net
# ====================================================================
from spyrit.core.recon import PinvNet
from spyrit.core.nnet import Unet
from spyrit.core.train import load_net

# retrained is same as original
model_name = 'pinv-net_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07_retrained_light.pth'

# Init
pinvnet = PinvNet(noise_op, prep_op, Unet())
pinvnet.eval() 

# Load net and use GPU if available
load_net(model_folder_full / model_name, pinvnet, device, False)
pinvnet = pinvnet.to(device)

print("\nPinv-Net reconstruction metrics")

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        print(f"For {alpha=}")
        pinvnet.prep.alpha = alpha
        pinvnet.acqu.alpha = alpha
        results = aux.eval_model_metrics_batch_cum(pinvnet, dataloader, device, metrics_eval, n_batches)
        print(results)
        
        
# %% 
# LPGD
# ====================================================================

model_name = "lpgd_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_128_reg_1e-07_uit_3_sdec0-9_light.pth"

# Initialize network
denoi = Unet()
lpgd = recon.LearnedPGD(noise_op, prep_op, denoi, step_decay=0.9)
lpgd.eval()

# load net and use GPU if available
load_net(model_folder_full / model_name, lpgd, device, False)
lpgd = lpgd.to(device)

print("\nLPGD reconstruction metrics")

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        print(f"For {alpha=}")
        lpgd.prep.alpha = alpha
        lpgd.acqu.alpha = alpha
        results = aux.eval_model_metrics_batch_cum(lpgd, dataloader, device, metrics_eval, n_batches)
        print(results)
        

# %% 
# DC-Net
# ====================================================================
from spyrit.core.recon import DCNet
from spyrit.core.nnet import Unet
from spyrit.core.train import load_net

model_name = 'dc-net_unet_imagenet_rect_N0_10_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07_light.pth'
cov_name = stat_folder_full / 'Cov_8_{}x{}.npy'.format(img_size, img_size)

# Load covariance prior
Cov = torch.from_numpy(np.load(cov_name)) 

# Init
denoi = Unet()         # torch.nn.Identity()
dcnet = DCNet(noise_op, prep_op, Cov, denoi)
dcnet.eval() 

# Load net and use GPU if available
load_net(model_folder_full / model_name, dcnet, device, False)
dcnet = dcnet.to(device)

print("\nDC-Net reconstruction metrics")

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        print(f"For {alpha=}")
        dcnet.prep.alpha = alpha
        dcnet.Acq.alpha = alpha
        results = aux.eval_model_metrics_batch_cum(dcnet, dataloader, device, metrics_eval, n_batches)
        print(results)


# %% 
# Pinv - PnP
# ====================================================================

model_name = "drunet_gray.pth"
noise_levels = [115, 50, 20] # noise levels from 0 to 255 for each alpha

# Initialize network
denoi = drunet.DRUNet()
pinvnet = recon.PinvNet(noise_op, prep_op, denoi)
pinvnet.eval()

#load_net(model_folder_full / model_name, pinvnet, device, False)
pinvnet.denoi.load_state_dict(
    torch.load(model_folder_full / model_name), 
    strict=False)
pinvnet.denoi.eval()
pinvnet = pinvnet.to(device)

print("\nPinv-PnP reconstruction metrics")

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        pinvnet.prep.alpha = alpha
        pinvnet.acqu.alpha = alpha
        nu = noise_levels[ii]
        pinvnet.denoi.set_noise_level(nu)
        print(f"For {alpha=} and {nu=}")
        results = aux.eval_model_metrics_batch_cum(pinvnet, dataloader, device, metrics_eval, n_batches)
        print(results)
        

# %% DPGD-PnP
from utility_dpgd import load_model, DualPGD

# load denoiser
n_channel, n_feature, n_layer = 1, 100, 20
model_name = 'DFBNet_l1_patchsize=50_varnoise0.1_feat_100_layers_20.pth'
denoi = load_model(pth = (model_folder_full / model_name).as_posix(), 
                    n_ch = n_channel, 
                    features = n_feature, 
                    num_of_layers = n_layer)

denoi.module.update_lip((1,50,50))
denoi.eval()

# Reconstruction hyperparameters
gamma = 1/img_size**2
max_iter = 101
mu_list = [6000, 3500, 1500]
crit_norm = 1e-4

# Init 
dpgdnet = DualPGD(noise_op, prep_op, denoi, gamma, mu_list[0], max_iter, crit_norm)

print("\nDPGD-PnP reconstruction metrics")

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        dpgdnet.prep.alpha = alpha
        dpgdnet.acqu.alpha = alpha
        dpgdnet.mu = mu_list[ii]
        mu = mu_list[ii]
        print(f"For {alpha=} and {mu=}")
        results = aux.eval_model_metrics_batch_cum(dpgdnet, dataloader, device, metrics_eval, n_batches)
        print(results)