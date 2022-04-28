# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:20:50 2022

@author: ducros
"""
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import spyrit.misc.walsh_hadamard as wh

from matplotlib import pyplot as plt

from spyrit.learning.model_Had_DCAN import *
from spyrit.misc.disp import torch2numpy
from spyrit.misc.statistics import Cov2Var
from spyrit.learning.nets import *

from spas import read_metadata, reconstruction_hadamard
from spas import ReconstructionParameters, setup_reconstruction, load_noise, reconstruct
from spas.noise import noiseClass
from spas.visualization import *
#from siemens_star_analysis import *

plt.rcParams['text.usetex'] = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Torch device: {device}')



# In[ ]:

f = np.load('./data/zoom_x1_starsector/zoom_x1_starsector_spectraldata.npz')
spectral_data = f['spectral_data']

metadata, acquisition_metadata, spectrometer_parameters, dmd_parameters = read_metadata('./data/zoom_x1_starsector/zoom_x1_starsector_metadata.json')
wavelengths = np.asarray(acquisition_metadata.wavelengths)

print(f'Spectral data dimensions: {spectral_data.shape}')
print(f'Wavelength range: {wavelengths[0]} - {wavelengths[-1]} nm')

print('\nAcquired data description:')
print(f'Light source: {metadata.light_source}')
print(f'Object: {metadata.object}')
print(f'Filter: {metadata.filter}')
print(f'Patterns: {acquisition_metadata.pattern_amount}')
print(f'Integration time: {spectrometer_parameters.integration_time_ms} ms')


print(metadata)

plt.figure()
plt.plot(wavelengths,spectral_data[0,:])
plt.title('First pattern spectrum')

#%% Load noise experimental parameters
data = np.load('./fit_model2.npz')
mu = data['mu']
sigma = data['sigma']
coeff = data['k']
noise = noiseClass(mu, sigma, coeff)

#%% Spectral binning (and checking a few things)
CR = 2048
lambda_min = 530
lambda_max = 530.1

def subsample(spectral_data, CR):
    
    # If only one wavelength is considered
    if spectral_data.ndim == 1:
        torch_img = np.zeros((2*CR))
        
        pos = spectral_data[0::2][:CR]
        neg = spectral_data[1::2][:CR]
        
        torch_img[0::2] = pos
        torch_img[1::2] = neg
    
    # If spectral_data contains all wavelengths
    if spectral_data.ndim == 2:
        
        torch_img = np.zeros((2*CR, spectral_data.shape[1]))
        
        pos = spectral_data[0::2][:CR,:]
        neg = spectral_data[1::2][:CR,:]
        
        torch_img[0::2,:] = pos
        torch_img[1::2,:] = neg
    
    return torch_img

F_bin_GT, wavelengths_bin_recon, bin_width, noise_bin = spectral_binning(
    spectral_data.T, 
    wavelengths, 
    lambda_min, 
    lambda_max, 
    1, 
    noise)


plt.figure()
plt.plot(F_bin_GT.squeeze())
plt.title(f'Raw measurements at $\lambda$={lambda_min} nm')
plt.xlabel('Raw pattern indices')
plt.ylabel('Intensity (in counts)')

imgs = subsample(spectral_data, CR).T
F_bin, wavelengths_bin_recon, bin_width, noise_bin = spectral_binning(
    imgs, 
    wavelengths, 
    lambda_min, 
    lambda_max, 
    1, 
    noise)

#%% Load reconstrcution network
img_size = 64
net_arch = 0 # Network variant

# Intensity distribution
N0 = 10000
sig = 0.5

#- Training parameters
num_epochs = 30
lr = 1e-3 
step_size = 10
gamma = 0.5
batch_size = 512
reg = 1e-7

suffix = '_N0_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(
           img_size, CR, num_epochs, lr, step_size,
           gamma, batch_size, reg)

H = wh.walsh2_matrix(img_size) / img_size

Mean = np.load('../../spas/data/Average_64x64.npy')/img_size
Cov  = np.load('../../spas/data/Cov_64x64.npy')/img_size**2

model = DenoiCompNet(img_size, CR, Mean, Cov, net_arch, N0, sig, H, Cov2Var(Cov))
model.to(device)

torch_img = torch.from_numpy(F_bin)
torch_img = torch_img.float()
torch_img = torch.reshape(torch_img, (1, 1, 2*CR)) # batches, channels, patterns
torch_img = torch_img.to(device)


'../2022_ISTE/models_v1.2'

#%%
def meas2img_from_ind(meas, ind, N):
    """Return image from measurement vector

    Args:
        meas (ndarray): Measurement vector. Must be 1-D.
        ind (ndarray):  Pattern indices. Must be 1-D.
        N (int):        Returned image will be N x N

    Returns:
        Img (ndarray): Measurement image. 2-D array
    """
    y = np.pad(meas, (0, N**2-len(meas)))
    Perm = permutation_from_ind(ind+1)
    Img = (Perm @ y).reshape((N, N))
    return Img

def completion(meas, M, Perm, Cov, Mean):
    
    nx, ny = Mean.shape
    
    # reorder measurements and statistics
    y     = Perm @ (meas.ravel())
    mu    = Perm @ (Mean.ravel())
    Sigma = Perm @ Cov @ Perm.T
    
    # extract blocks corresponding to acquired and missing measurements 
    mu1 = mu[:M]            # mean of acquired measurements
    mu2 = mu[M:]            # mean of missing measurements
    Sigma1  = Sigma[:M,:M]  # covariance of acquired measurements
    Sigma21 = Sigma[M:,:M]  # covariance of missing measurements

    m = y[:M];
    
    # Bayesian denoised completion
    #y1 = mu1 + W1 @ ( m - mu1)
    y1 = m
    #y2 = mu2 + Sigma21 @ np.linalg.lstsq(Sigma1,(y1 - mu1), rcond=None)[0]
    y2 = Sigma21 @ np.linalg.lstsq(Sigma1, y1, rcond=None)[0]
    #y2 = Sigma21 @ np.linalg.inv(Sigma1) @ y1
    y[:M] = y1
    y[M:] = y2
    
    # reorder output
    y = Perm.T @ y
    y = np.reshape(y,(nx, ny))
    
    return y

def unsplit(raw):
    had = raw[0::2] - raw[1::2]
    return had

#%% Recon from iwalsh ()
M = CR
M = 4096


had = unsplit(F_bin_GT.squeeze()[:2*M])
ind = np.array(acquisition_metadata.patterns)[::2]//2
had_img = meas2img_from_ind(had, ind, 64) #np.array(acquisition_metadata.patterns)[:2*CR],64)

rec_had = wh.iwalsh2(had_img)
#
plt.figure()
plt.imshow(np.log10(np.abs(had_img)), cmap='gray')
plt.title(f"Hadamard coefficients, M = {M}")
plt.colorbar()

plt.figure()
plt.imshow(rec_had, cmap='gray')
plt.colorbar()
plt.title(f"Reconstruction iwalsh, M = {M}")

#%% PINV recon
recon_pinv = model.forward_reconstruct_pinv(
    torch_img, 1, 1, img_size, img_size,
)

recon_pinv = (recon_pinv+1)*model.N0/2
recon_pinv = recon_pinv.cpu().detach().numpy().squeeze()

plt.figure()
plt.title(f"Pinv Reconstruction, M = {CR}")
plt.imshow(recon_pinv, cmap='gray')
plt.colorbar()

#%% COMP recon
model.N0 = recon_pinv.max()
 
result = model.forward_reconstruct_comp(
    torch_img, 1, 1, img_size, img_size, 
)

#result = (result+1) * model.N0/2
result = result.cpu().detach().numpy().squeeze()

plt.figure()
plt.imshow(result, cmap='gray')
plt.colorbar()
plt.title(f'COMP recon, M = {CR}')

#%% MMSE recon 
model.N0 = torch.tensor(100)
model.N0 = 2*recon_pinv.max()
model.N0 = 4*recon_pinv.max()

result = model.forward_reconstruct_mmse(
    torch_img, 1, 1, img_size, img_size, 
)

result = (result+1) * recon_pinv.max()
result = result.cpu().detach().numpy().squeeze()

plt.figure()
plt.imshow(result, cmap='gray')
plt.colorbar()
plt.title(f'MMSE recon, N0 = {model.N0:.1f} (from GT), M = {CR}')