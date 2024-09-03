# %% Imports
# --------------------------------------------------------------------
import os
from pathlib import Path
import sys
import math
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# %% General
# --------------------------------------------------------------------

# Experimental data
image_folder = 'data/images/'       # images for simulated measurements
model_folder = 'model/'             # reconstruction models
stat_folder  = 'stat/'              # statistics
recon_folder = 'recon/figure_3/'    # reconstructed images

# Full paths
image_folder_full = Path.cwd() / Path(image_folder)
model_folder_full = Path.cwd() / Path(model_folder)
stat_folder_full  = Path.cwd() / Path(stat_folder)
recon_folder_full = Path.cwd() / Path(recon_folder)
recon_folder_full.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %% Load images
# --------------------------------------------------------------------
import spyrit.misc.statistics as stats
from spyrit.misc.disp import imagesc
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
    batch_size=1, 
    shuffle=False
    )

# select the image
x, _ = next(iter(dataloader))
x = x[0]
c, h, w = x.shape
print("Image shape:", x.shape)

x_plot = x.view(-1, h, h).cpu().numpy()
imagesc(x_plot[0, :, :])

# %% Simulate measurements for three image intensities
# --------------------------------------------------------------------
alpha_list = [2, 10, 50] # Poisson law parameter for noisy image acquisitions

from spyrit.core.meas import HadamSplit
from spyrit.core.noise import Poisson
from spyrit.misc.sampling import meas2img
from spyrit.core.prep import SplitPoisson

# Measurement parameters
M = 128 * 128 // 4  # Number of measurements (here, 1/4 of the pixels)

# Measurement and noise operators
Ord_rec = np.ones((img_size, img_size))
Ord_rec[:,img_size//2:] = 0
Ord_rec[img_size//2:,:] = 0

meas_op = HadamSplit(M, h, torch.from_numpy(Ord_rec))
noise_op = Poisson(meas_op, alpha_list[0])
prep_op = SplitPoisson(2, meas_op)

# Vectorized image
x = x.view(1, h * w)

# Measurement vectors
y = torch.zeros(3,2*M)
for ii, alpha in enumerate(alpha_list): 
    torch.manual_seed(0)    # for reproducibility
    noise_op.alpha = alpha
    y[ii,:] = noise_op(x)

# Send to GPU if available
y = y.to(device)

# %% Pinv
# --------------------------------------------------------------------
from spyrit.core.recon import PinvNet

# Init
pinv = PinvNet(noise_op, prep_op)

# Use GPU if available
pinv = pinv.to(device)

# Reconstruct
x_pinv = torch.zeros(3,1,img_size,img_size)

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        pinv.prep.alpha = alpha
        x_pinv[ii] =  pinv.reconstruct(y[ii:ii+1, :]) # NB: shape of measurement is (1,8192)

# %% Save reconstructions from pinv
# --------------------------------------------------------------------
save_tag = True

if save_tag:
    for ii, alpha in enumerate(alpha_list):
        filename = f'pinv_alpha_{alpha:02}.png'
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_pinv[ii,0].cpu().detach().numpy(), 
            cmap='gray') #

# %% -----------------------------------------------------------------
# Pinv-Net
# --------------------------------------------------------------------
from spyrit.core.recon import PinvNet
from spyrit.core.nnet import Unet
from spyrit.core.train import load_net

model_name = 'pinv-net_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07_retrained_light.pth'

# Init
pinvnet = PinvNet(noise_op, prep_op, Unet())
pinvnet.eval() 

# Load net and use GPU if available
load_net(model_folder_full / model_name, pinvnet, device, False)
pinvnet = pinvnet.to(device)

# Reconstruct
x_pinvnet = torch.zeros(3,1,img_size,img_size)

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        pinvnet.prep.alpha = alpha
        x_pinvnet[ii] =  pinvnet.reconstruct(y[ii:ii+1, :]) # NB: shape of measurement is (1,8192)

# %% -----------------------------------------------------------------
# Save reconstructions from Pinv-Net
# --------------------------------------------------------------------
save_tag = True

if save_tag:
    for ii, alpha in enumerate(alpha_list):
        filename = f'pinvnet_alpha_{alpha:02}.png'
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_pinvnet[ii,0].cpu().detach().numpy(), 
            cmap='gray') #

# %% DC-Net
# --------------------------------------------------------------------
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

# Reconstruct
x_dcnet = torch.zeros(3,1,img_size,img_size)

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        dcnet.prep.alpha = alpha
        x_dcnet[ii] =  dcnet.reconstruct(y[ii:ii+1, :]) # NB: shape of measurement is (1,8192) as expected

# %% Save reconstructions from DC-Net
# --------------------------------------------------------------------
save_tag = True

if save_tag:
    for ii, alpha in enumerate(alpha_list):
        filename = f'dcnet_alpha_{alpha:02}.png'
        full_path = recon_folder_full / filename
        plt.imsave(full_path, x_dcnet[ii,0].cpu().detach().numpy(), 
            cmap='gray') #



# %% DPGD-PnP
# --------------------------------------------------------------------
gamma = 1/img_size**2
max_iter = 101
mu_list = [6000, 3500, 1500]
crit_norm = 1e-4

from utility import get_model, load_model
def get_dfb_model():
    #-- load denoiser
    n_channel, n_feature, n_layer = 1, 100, 20

    model_name = f'../../model_pnp/DFBNet_l1_patchsize=50_varnoise0.1_feat_{n_feature}_layers_{n_layer}/'
    model, net_name, clip_val, lr = get_model('DFBNet', n_channel,  n_feature, n_layer)
    model = load_model(pth = model_dir + net_name + '.pth', 
                        net = model, 
                        n_ch = n_channel, 
                        features = n_feature, 
                        num_of_layers = n_layer)
    model.module.update_lip((1,50,50))
    model.eval()    
    return model

denoi = get_dfb_model() 
model = PnP(noise_op, prep_op, denoi, gamma, mu, max_iter, crit_norm)