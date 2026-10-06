#%% 

import numpy as np
from pathlib import Path
from tools import binArray
import torch 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Read in formatted data -> i.e unsplit and in shape (NxLxK)

data_folder = Path("data/lisaCat") 
#Path("data/lisaCat") #Path("data/USAF2") #Path("data/USAF") # Path("data/opticalTuningCat3") # Path("data/opticalTuningCat2")

m = np.load(data_folder/'m_binned.npy')
N,L,P = m.shape
K = P//2

m = np.moveaxis(m,0,1) # move spectral axis to front 
m = torch.from_numpy(m).to(torch.float32).to(device)

# to pre sum channels if you want
# m = m.sum(0)
# L = 1

print("Read in formatted data from ", data_folder)
print("Shape of formatted measurements: ", m.shape)
#%% parameters
calib_data = Path('C:/Users/ceidigh/Documents/PFE/CalibrationData')

# choose which params to use based on acquisition config & formatting of raw data
# eg lisaCat was done using 12.04dB camera gain and then binned spatially x3:
gamma = torch.from_numpy(np.load(calib_data / "bright" / "slope_g3_bin_x3_12.04.npy")).to(device)
mudark_image = torch.from_numpy(np.load(calib_data / "dark" / "mu_dark_image_binned_x3_1.0_12.04.npy")).to(device)
sigdark_image = torch.from_numpy(np.load(calib_data / "dark" / "sigma_dark_image_binned_x3_1.0_12.04.npy")).to(device)

mudark = mudark_image.mean()
sigdark =sigdark_image.mean()


#%% Pseudoinverse Reconstruction

import torch
from spyrit.core.recon import PinvNet 
from spyrit.core.meas import Linear
from spyrit.misc.walsh_hadamard import walsh_matrix
from spyrit.core.prep import Unsplit
import matplotlib.pyplot as plt


H = walsh_matrix(K)
H = torch.from_numpy(H).to(device)
meas_op = Linear(H, device=device) 
prep_op = Unsplit()



model = PinvNet(meas_op, prep=prep_op, store_H_pinv = True, device = device) 
model.eval()
model = model.to(device)


with torch.no_grad():

    x_hat = model.reconstruct(m/gamma)
    x_pinv = x_hat.detach().cpu().numpy().squeeze()

del model 

plt.imshow(x_pinv.sum(0))
plt.title("PSEUDOINVERSE")
plt.colorbar()


#%% Tikhonov Reconstruction
from spyrit.core.prep import UnsplitRescale
from spyrit.core.recon import TikhoNet

# normalisation parameter estimated from pseudoinverse
alpha_est = x_pinv.reshape(L, N*N).max(-1)
alpha_est = torch.from_numpy(alpha_est).to(device)

# Sigma calculation (measurement covariance)
z = (m[..., 0::2] + m[..., 1::2]) # .detach().cpu() 
cov_meas = gamma * (z - 2*mudark) + 2 * sigdark**2   
cov_meas = cov_meas.to(device)
norm = gamma*alpha_est[:,None, None]
cov_meas = cov_meas / norm**2
Gamma = torch.diag_embed(cov_meas)

# load Gamma
image_cov = np.load('data/Cov_1_128x128.npy')
Sigma = torch.from_numpy(image_cov).to(device)

# load net
from spyrit.core.nnet import Unet
from spyrit.core.recon import OrderedDict, TikhoNet, PinvNet
from spyrit.core.train import load_net
import torch.nn as nn
from spyrit.core.prep import Rerange

title = f'model/TIKHO_UNET_128x128_K=128_alpha100_20epochs.pth'

#title = f'model/tikho_unet_128x128_K=128_alpha2600_20epochs.pth'
#title = f'model/tikho-net_unet_imagenet_ph_10_exp_N_512_M_128_epo_20_lr_0.001_sss_10_sdr_0.5_bs_20_reg_1e-07.pth'



denoiser = torch.nn.Sequential(OrderedDict({"denoi": Unet()}))
load_net(title, denoiser, device, False)

prep_op = UnsplitRescale(alpha_est[:,None, None])
model = TikhoNet(meas_op, prep = prep_op , sigma = Sigma, denoi = denoiser, device = device) 
model.eval()
model = model.to(device)

with torch.no_grad():

    m_hat = model.prep(m/gamma)
    x_hat = model.tikho(m_hat, Gamma)

    x_d = model.denoi(x_hat.reshape(L, 1, 128, 128))
    x_tiko = x_hat.detach().cpu().numpy().squeeze()
    x_tikonet = x_d.detach().cpu().numpy().squeeze()
del model

x_tikonet_denorm = x_tikonet*alpha_est.detach().cpu().numpy().squeeze()[:,None, None]
x_tiko_denorm = x_tiko*alpha_est.detach().cpu().numpy().squeeze()[:,None, None]

plt.imshow(x_tiko_denorm.sum(0))
plt.title("TIKHONOV")
plt.colorbar()

plt.figure()
plt.imshow(x_tikonet_denorm.sum(0))
plt.title("TIKHONOV + UNet()")
plt.colorbar()


# %% PLOT

from tools import read_acquisitionData, binArray
import re
from spyrit.misc.color import wavelength_to_colormap
import matplotlib.pyplot as plt

# choose which to plot
x_recon = x_tikonet_denorm #x_tikonet_denorm #x_pinv


meta_path = data_folder / "metadata.json"
meta = read_acquisitionData(meta_path)
acquisition_parameters = meta['acquisition_params']


wavelengths = acquisition_parameters.wavelengths # list of wavelengths measured
cmap = wavelength_to_colormap(wavelengths[0], gamma=0.6)


fig, axs = plt.subplots(2, 3)

# Global title
fig.suptitle("Tikhonov Regularisation + UNet() at selected wavelengths", fontsize=16)
im = axs[0,0].imshow(np.fliplr(x_recon[0,:,:]), cmap=cmap)
axs[0,0].set_title( rf" $\lambda$ = {wavelengths[0]}")
plt.colorbar(im, ax = axs[0,0], orientation = 'horizontal')

cmap = wavelength_to_colormap(wavelengths[100], gamma=0.6)
im = axs[0,1].imshow(np.fliplr(x_recon[100,:,:]), cmap=cmap)
axs[0,1].set_title( rf" $\lambda$ = {wavelengths[100]}")
plt.colorbar(im, ax = axs[0,1], orientation = 'horizontal')

cmap = wavelength_to_colormap(wavelengths[200], gamma=0.6)
im = axs[0,2].imshow(np.fliplr(x_recon[200,:,:]), cmap=cmap)
axs[0,2].set_title( rf" $\lambda$ = {wavelengths[200]}")
plt.colorbar(im, ax = axs[0,2], orientation = 'horizontal')

cmap = wavelength_to_colormap(wavelengths[300], gamma=0.6)
im = axs[1,0].imshow(np.fliplr(x_recon[300,:,:]), cmap=cmap)
axs[1,0].set_title( rf" $\lambda$ = {wavelengths[300]}")
plt.colorbar(im, ax = axs[1,0], orientation = 'horizontal')

cmap = wavelength_to_colormap(wavelengths[400], gamma=0.6)
im = axs[1,1].imshow(np.fliplr(x_recon[400,:,:]), cmap=cmap)
axs[1,1].set_title( rf" $\lambda$ = {wavelengths[400]}")
plt.colorbar(im, ax = axs[1,1], orientation = 'horizontal')

cmap = wavelength_to_colormap(wavelengths[500], gamma=0.6)
im = axs[1,2].imshow(np.fliplr(x_recon[500,:,:]), cmap=cmap)
axs[1,2].set_title( rf" $\lambda$ = {wavelengths[500]}")
plt.colorbar(im, ax = axs[1,2], orientation = 'horizontal')


# %% Reconstructing only negative measures

# %%
