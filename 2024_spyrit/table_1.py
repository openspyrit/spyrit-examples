# %%
# Imports
# --------------------------------------------------------------------
from pathlib import Path
from typing import OrderedDict

import torch
import torch.nn as nn

import spyrit.core.meas as meas
import spyrit.core.prep as prep
import spyrit.core.nnet as nnet
import spyrit.core.noise as noise
import spyrit.core.recon as recon
import spyrit.core.train as train
import spyrit.misc.statistics as stats
import spyrit.external.drunet as drunet

import aux_functions as aux
import utility_dpgd as dpgd

# %%
# General Parameters
# --------------------------------------------------------------------
img_size = 128  # image size
batch_size = 128
n_batches = 3  # iterate over how many batches to get statistical data

# Experimental
val_folder = "data/ILSVRC2012/val/"     # statistical analysis
model_folder = "model/"                 # reconstruction models
stat_folder = "stat/"                   # statistics
recon_folder = "recon/table_1/"         # table output

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
metrics_eval = ["nrmse", "ssim"]


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
# Pinv
# ====================================================================
# Init
pinv = recon.PinvNet(meas_op, prep_op, device=device)

# Use GPU if available
pinv = pinv.to(device)

print("\nPinv reconstruction metrics")

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        print(f"For {alpha=}")
        torch.manual_seed(0)  # for reproducibility
        
        # Set alpha for the simulation of the measurements
        # pinv.acqu_modules.acqu.noise_model.alpha = alpha # works too
        pinv.acqu.noise_model.alpha = alpha      
        
        # Set alpha for reconstruction
        # pinv.recon_modules.prep.alpha = alpha # works too
        pinv.prep.alpha = alpha
        
        results = aux.eval_model_metrics_batch_cum(
            pinv, dataloader, device, metrics_eval[0:1], n_batches
        )
        print(results)

del pinv

# %%
# Pinv-Net
# ====================================================================
# retrained is same as original
model_name = "pinv-net_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07_retrained_light.pth"

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

print("\nPinv-Net reconstruction metrics")

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        print(f"For {alpha=}")
        torch.manual_seed(0)  # for reproducibility
        
        # Set alpha for the simulation of the measurements
        # pinvnet.acqu_modules.acqu.noise_model.alpha = alpha # works too
        pinvnet.acqu.noise_model.alpha = alpha      
        
        # Set alpha for reconstruction
        # pinvnet.recon_modules.prep.alpha = alpha # works too
        pinvnet.prep.alpha = alpha 
        
        results = aux.eval_model_metrics_batch_cum(
            pinvnet, dataloader, device, metrics_eval, n_batches
        )
        print(results)

del pinvnet

# %%
# LPGD
# ====================================================================
model_name = "lpgd_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_128_reg_1e-07_uit_3_sdec0-9_light.pth"

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

print("\nLPGD reconstruction metrics")

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        print(f"For {alpha=}")
        torch.manual_seed(0)  # for reproducibility
        
        # Set alpha for the simulation of the measurements
        # lpgd.acqu_modules.acqu.noise_model.alpha = alpha # works too
        lpgd.acqu.noise_model.alpha = alpha      
        
        # Set alpha for reconstruction
        # lpgd.recon_modules.prep.alpha = alpha # works too
        lpgd.prep.alpha = alpha 

        results = aux.eval_model_metrics_batch_cum(
            lpgd, dataloader, device, metrics_eval, n_batches
        )
        print(results)

del lpgd

# %%
# DC-Net
# ====================================================================
model_name = "dc-net_unet_imagenet_rect_N0_10_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07_light.pth"
cov_name = stat_folder_full / "Cov_8_{}x{}.pt".format(img_size, img_size)

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
cov_name = stat_folder_full / "Cov_8_{}x{}.pt".format(img_size, img_size)
Cov = torch.load(cov_name, weights_only=True).to(device)
# divide by 4 because the measurement covariance has been computed on images
# with values in [-1, 1] (total span 2) whereas our image is in [0, 1] (total
# span 1). The covariance is thus 2^2 = 4 times larger than expected.
Cov /= 4

# Init
#prep_op = prep.UnsplitRescaleEstim(meas_op, use_fast_pinv=True)
dcnet = recon.DCNet(meas_op, prep_op, Cov, denoiser, device=device)
dcnet.eval()

print("\nDC-Net reconstruction metrics")

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        print(f"For {alpha=}")
        torch.manual_seed(0)  # for reproducibility
        
        # Set alpha for the simulation of the measurements
        # dcnet.acqu_modules.acqu.noise_model.alpha = alpha # works too
        dcnet.acqu.noise_model.alpha = alpha      
        
        # Set alpha for reconstruction
        # dcnet.recon_modules.prep.alpha = alpha # works too
        dcnet.prep.alpha = alpha 
        
        results = aux.eval_model_metrics_batch_cum(
            dcnet, dataloader, device, metrics_eval, n_batches
        )
        print(results)

del dcnet

# %%
# Pinv - PnP
# ====================================================================
model_name = "drunet_gray.pth"
#noise_levels = [115, 50, 20]  # noise levels from 0 to 255 for each alpha
noise_levels = [
    [115],   # 2 photons
    [50],    # 10 photons
    [20],    # 50 photons
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
        torch.manual_seed(0)  # for reproducibility
        
        # Set alpha for the simulation of the measurements
        # pinvpnp.acqu_modules.acqu.noise_model.alpha = alpha # works too
        pinvpnp.acqu.noise_model.alpha = alpha
        
        # Set alpha for reconstruction
        # pinvpnp.recon_modules.prep.alpha = alpha # works too
        pinvpnp.prep.alpha = alpha 
        
        for nu in noise_levels[ii]:
            pinvpnp.denoi.denoi.set_noise_level(nu)
            print(f"For {alpha=} and {nu=}")
    
            results = aux.eval_model_metrics_batch_cum(
                pinvpnp, dataloader, device, metrics_eval, n_batches
            )
            print(results)

del pinvpnp

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
    [6000],  # 2 photons
    [3500],  # 10 photons
    [1500],  # 50 photons
]

# Init
dpgdnet = dpgd.DualPGD(meas_op, prep_op, denoi, gamma, mu_list[0], max_iter, crit_norm)
dpgdnet = dpgdnet.to(device)

print("\nDPGD-PnP reconstruction metrics")

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        torch.manual_seed(0)  # for reproducibility
        
        # Set alpha for the simulation of the measurements
        # dpgdnet.acqu_modules.acqu.noise_model.alpha = alpha # works too
        dpgdnet.acqu.noise_model.alpha = alpha
        
        # Set alpha for reconstruction
        # dpgdnet.recon_modules.prep.alpha = alpha # works too
        dpgdnet.prep.alpha = alpha 
        
        for mu in mu_list[ii]:
            dpgdnet.mu = mu
            
            print(f"For {alpha=} and {mu=}")
            results = aux.eval_model_metrics_batch_cum(
                dpgdnet, dataloader, device, metrics_eval, n_batches
            )
            print(results)
            
del dpgdnet