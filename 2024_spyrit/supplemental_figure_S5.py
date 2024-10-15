# %% 
# Imports
# --------------------------------------------------------------------
from pathlib import Path

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import utility_dpgd as dpgd
import spyrit.core.meas as meas
import spyrit.core.nnet as nnet
import spyrit.core.prep as prep
import spyrit.core.noise as noise
import spyrit.core.recon as recon
import spyrit.core.train as train
import spyrit.misc.statistics as stats
import spyrit.external.drunet as drunet


# %% 
# General
# --------------------------------------------------------------------
# Experimental data
image_folder = 'data/images/'       # images for simulated measurements
model_folder = 'model/'             # reconstruction models
stat_folder  = 'stat/'              # statistics
recon_folder = 'recon/supplemental_figure_S5/'    # reconstructed images

# Full paths
image_folder_full = Path.cwd() / Path(image_folder)
model_folder_full = Path.cwd() / Path(model_folder)
stat_folder_full  = Path.cwd() / Path(stat_folder)
recon_folder_full = Path.cwd() / Path(recon_folder)
recon_folder_full.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# %% 
# Load images
# --------------------------------------------------------------------

img_size = 128 # image size

print("Loading image...")
# crop to desired size, set to black and white, normalize
transform = stats.transform_gray_norm(img_size)

# define dataset and dataloader. `image_folder_full` should contain
# a class folder with the images
dataset = torchvision.datasets.ImageFolder(
    image_folder_full, 
    transform=transform
    )

dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=6, 
    shuffle=False
    )

# select the two images
x, _ = next(iter(dataloader))
label_list = ['brain', 'dog', 'panther', 'box', 'bird', 'car']
b, c, h, w = x.shape
print("Batch shape:", x.shape)

for i, label in enumerate(label_list):
    x_plot = x[i, :, :, :].cpu().numpy()
    plt.imsave(recon_folder_full / f'sim{i}_{img_size}_gt.png', x_plot[0, :, :], cmap='gray')


# %% 
# Simulate measurements
# --------------------------------------------------------------------
# Measurement parameters
alpha = 2 # Poisson law parameter for noisy image acquisitions
M = img_size**2 // 4  # Number of measurements (here, 1/4 of the pixels)

# Measurement and noise operators
Ord_rec = torch.ones((img_size, img_size))
Ord_rec[:,img_size//2:] = 0
Ord_rec[img_size//2:,:] = 0

meas_op = meas.HadamSplit(M, h, Ord_rec)
noise_op = noise.Poisson(meas_op, alpha)
prep_op = prep.SplitPoisson(alpha, meas_op)

# Vectorized images
x = x.view(b, 1, h * w)
# Measurement vectors
y = torch.zeros(b, 2*M)

for ii in range(b):
    torch.manual_seed(0) # for reproducibility
    y[ii, :] = noise_op(x[ii, :])

# Send to GPU if available
y = y.to(device)


# %% 
# Pinv
# ====================================================================
# Init
pinv = recon.PinvNet(noise_op, prep_op)

# Use GPU if available
pinv = pinv.to(device)

with torch.no_grad():
    x_pinv =  pinv.reconstruct(y) # NB: shape of measurement is (1,8192)

    for ii, label in enumerate(label_list):
        filename = f'sim{ii}_{img_size}_N0_{alpha}_M_{M}_rect_pinv.png'
        full_path = recon_folder_full / filename
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_pinv[ii,0].cpu().detach().numpy(), cmap='gray')


# %% 
# Pinv-Net
# ====================================================================
model_name = 'pinv-net_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07_retrained_light.pth'

# Init
pinvnet = recon.PinvNet(noise_op, prep_op, nnet.Unet())
pinvnet.eval() 

# Load net and use GPU if available
train.load_net(model_folder_full / model_name, pinvnet, device, False)
pinvnet = pinvnet.to(device)

with torch.no_grad():
    x_pinvnet = pinvnet.reconstruct(y) # NB: shape of measurement is (1,8192)
    
    for ii, label in enumerate(label_list):
        filename = f'sim{ii}_{img_size}_N0_{alpha}_M_{M}_rect_pinvnet_unet.png'
        full_path = recon_folder_full / filename
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_pinvnet[ii,0].cpu().detach().numpy(), cmap='gray')


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

with torch.no_grad():
    x_lpgd = lpgd.reconstruct(y)
    
    for ii, label in enumerate(label_list):
        filename = f'sim{ii}_{img_size}_N0_{alpha}_M_{M}_rect_lpgd_unet.png'
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_lpgd[ii,0].cpu().detach().numpy(), cmap='gray')
    

# %% 
# DC-Net
# ====================================================================
model_name = 'dc-net_unet_imagenet_rect_N0_10_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07_light.pth'
cov_name = stat_folder_full / 'Cov_8_{}x{}.npy'.format(img_size, img_size)

# Load covariance prior
Cov = torch.from_numpy(np.load(cov_name)) 

# Init
denoi = nnet.Unet()         # torch.nn.Identity()
dcnet = recon.DCNet(noise_op, prep_op, Cov, denoi)
dcnet.eval() 

# Load net and use GPU if available
train.load_net(model_folder_full / model_name, dcnet, device, False)
dcnet = dcnet.to(device)

with torch.no_grad():
    x_dcnet = dcnet.reconstruct(y)
    
    for ii, label in enumerate(label_list):
        filename = f'sim{ii}_{img_size}_N0_{alpha}_M_{M}_rect_dc-net_unet.png'
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_dcnet[ii,0].cpu().detach().numpy(), cmap='gray')
        

# %% 
# Pinv - PnP
# ====================================================================
model_name = "drunet_gray.pth"
# label_list = ['brain', 'dog', 'panther', 'box', 'bird', 'car']
nu_list = [70, 115, 115, 100, 100, 100] # noise levels for each label

# Initialize network
denoi = drunet.DRUNet()
pinvnet = recon.PinvNet(noise_op, prep_op, denoi)
pinvnet.eval()

#load_net(model_folder_full / model_name, pinvnet, device, False)
pinvnet.denoi.load_state_dict(
    torch.load(model_folder_full / model_name, weights_only=True), 
    strict=False)
pinvnet.denoi.eval()
pinvnet = pinvnet.to(device)

with torch.no_grad():
    
    for ii, nu in enumerate(nu_list):
        pinvnet.denoi.set_noise_level(nu)
        x_pinvPnP = pinvnet.reconstruct(y[ii:ii+1, :])
        
        filename = f'sim{ii}_{img_size}_N0_{alpha}_M_{M}_rect_pinv-net_drunet_nlevel_{nu}.png'
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_pinvPnP[0,0].cpu().detach().numpy(), cmap='gray')


# %% 
# DPGD-PnP
# ====================================================================
# load denoiser
n_channel, n_feature, n_layer = 1, 100, 20
model_name = 'DFBNet_l1_patchsize=50_varnoise0.1_feat_100_layers_20.pth'
denoi = dpgd.load_model(pth = (model_folder_full / model_name).as_posix(), 
                    n_ch = n_channel, 
                    features = n_feature, 
                    num_of_layers = n_layer)

denoi.module.update_lip((1,50,50))
denoi.eval()

# Reconstruction hyperparameters
# label_list = ['brain', 'dog', 'panther', 'box', 'bird', 'car']
mu_list = [4000, 6000, 6000, 6000, 6000, 6000] # noise levels for each label
gamma = 1/img_size**2
max_iter = 101
crit_norm = 1e-4

# Init 
dpgdnet = dpgd.DualPGD(noise_op, prep_op, denoi, gamma, mu_list[0], max_iter, crit_norm)

with torch.no_grad():
    
    for ii, mu in enumerate(mu_list):
        dpgdnet.mu = mu
        x_dpgd =  dpgdnet.reconstruct(y[ii:ii+1, :]) # NB: shape of measurement is (1,8192) as expected
        
        filename = f'sim{ii}_{img_size}_N0_{alpha}_M_{M}_rect_dfb-net_dfb_mu_{mu}.png'
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_dpgd[0,0].cpu().detach().numpy(), cmap='gray')
# %%
