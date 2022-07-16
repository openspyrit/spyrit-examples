# -*- coding: utf-8 -*-

#%%
import torch
import numpy as np
import spyrit.misc.walsh_hadamard as wh

from matplotlib import pyplot as plt

from spyrit.learning.model_Had_DCAN import *
from spyrit.misc.disp import torch2numpy, imagesc, plot

from spyrit.learning.nets import *
from spyrit.restructured.Updated_Had_Dcan import *
from spyrit.misc.metrics import psnr_

from pathlib import Path

#plt.rcParams['text.usetex'] = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Torch device: {device}')


#%% Spectral binning (and checking a few things)
def subsample(spectral_data, CR):
    # If only one wavelength is considered
    if spectral_data.ndim == 1:
        torch_img = np.zeros((2*CR))
        
        pos = spectral_data[0::2][:CR]
        neg = spectral_data[1::2][:CR]
        
        torch_img[0::2] = pos
        torch_img[1::2] = neg
    
    # If spectral_data contains  all wavelengths
    if spectral_data.ndim == 2:
        
        torch_img = np.zeros((2*CR, spectral_data.shape[1]))
        
        pos = spectral_data[0::2][:CR,:]
        neg = spectral_data[1::2][:CR,:]
        
        torch_img[0::2,:] = pos
        torch_img[1::2,:] = neg
    
    return torch_img


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


def unsplit(raw):
    had = raw[0::2,:] - raw[1::2,:]
    return had

#%% Load reconstruction network
img_size = 64
M = 2048
N0 = 10
stat_folder = Path('models_online/') 
average_file = stat_folder / ('Average_{}x{}'.format(img_size,img_size)+'.npy')
cov_file = stat_folder / ('Cov_{}x{}'.format(img_size,img_size)+'.npy')
    
title = f'./model_exp/DC_Net_Conv_net_N0_10_N_64_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07'

# Init DC-Net
Mean = np.load(average_file)
Cov  = np.load(cov_file)

H =  wh.walsh2_matrix(img_size)
Ord = Cov2Var(Cov)
Perm = Permutation_Matrix(Ord)
Hperm = Perm@H
Pmat = Hperm[:M,:]

Forward = Split_Forward_operator_ft_had(Pmat, Perm)
Noise = Bruit_Poisson_approx_Gauss(N0, Forward)
Prep = Split_diag_poisson_preprocess(N0, M, img_size**2)
DC = Generalized_Orthogonal_Tikhonov(sigma_prior = Cov, M = M, 
                                     N = img_size**2)
Denoi = ConvNet()
model = DC_Net(Noise, Prep, DC, Denoi)

# Load trained DC-Net
load_net(title, model, device)
model.eval()                    # Mandantory when batchNorm is used

#%% Load expe data and unsplit
data_root = Path('data_online/')
#-1
data_folder = Path('usaf_x12/')
data_file_prefix = 'zoom_x12_usaf_group5'                   
#-2
# data_folder = Path('usaf_x2/')
# data_file_prefix = 'zoom_x2_usaf_group2' 
#-3
# data_folder = Path('star_sector_x2/')
# data_file_prefix = 'zoom_x2_starsector' 
#-4
# data_folder = Path('star_sector_x12/')
# data_file_prefix = 'zoom_x12_starsector' 
#-5
# data_folder = Path('tomato_slice_x12/')
# data_file_prefix = 'tomato_slice_2_zoomx12' 
#-6
# data_folder = Path('tomato_slice_x2/')
# data_file_prefix = 'tomato_slice_2_zoomx2'
#-7
#data_folder = Path('cat/')
#data_file_prefix = 'Cat_whiteLamp'
#-8
data_folder = Path('horse/')
data_file_prefix = 'Horse_whiteLamp'

full_path = data_root / data_folder / (data_file_prefix + '_spectraldata.npz')
raw = np.load(full_path)

meas    = raw['spectral_data']
had_all = unsplit(meas)    # unsplit

#%% Plot sample (all channels summed up)
had_sum = Perm.T @ had_all.sum(axis=1)
had_sum = np.reshape(had_sum,(64,64))
rec_sum = wh.iwalsh2(had_sum)

plt.figure()
plt.imshow(rec_sum, cmap='gray')
plt.colorbar()
plt.title(f"All channels summed up, all meas")


#%% Reconstruction of all channels independently (all meas, i.e., CR = 1)
rec_all = (Hperm.T @ had_all)/img_size**2
rec_all = np.reshape(rec_all, (img_size, img_size, -1))

wav_0 = 50 
wav_1 = 500
wav_2 = 1500

fig , axs = plt.subplots(1,3)
#
im0 = axs[0].imshow(rec_all[:,:,wav_0], cmap='gray')
fig.colorbar(im0, ax=axs[0])
axs[0].set_title(f"All meas, channel = {wav_0}")
#
im1 = axs[1].imshow(rec_all[:,:,wav_1], cmap='gray')
fig.colorbar(im1, ax=axs[1])
axs[1].set_title(f"All meas, channel = {wav_1}")
#
im2 = axs[2].imshow(rec_all[:,:,wav_2], cmap='gray')
fig.colorbar(im2, ax=axs[2])
axs[2].set_title(f"All meas, channel = {wav_2}")

#%%  Reconstruction of all channels independently (subsampled, i.e., CR < 1)
data_root = Path('data_online/')
save_root = Path('recon_exp/') # no save if False

# Init Pinv
Pinv = Pinv_orthogonal()
Forward.to(device) # to compute pinv on GPU
Prep.to(device)
DC.to(device)
Denoi.to(device)

# Subsample
had = had_all[:M,:]

# Init Output
rec_pinv = np.zeros((meas.shape[1], img_size, img_size))
rec_mmse = np.zeros((meas.shape[1], img_size, img_size))
rec_net  = np.zeros((meas.shape[1], img_size, img_size))

x_0 = torch.zeros((1, Forward.N)).to(device)

# Loop
for wav in range(meas.shape[1]):
    
    print(f'wavelength: {wav}')
    
    # Measurement vector
    m = torch.Tensor(meas[:2*M,wav])
    m = m.view(1,1,2*M).to(device)

    # Pseudo-inverse
    rec_gpu = Pinv(m[...,0::2]-m[...,1::2],Forward)
    rec_cpu = rec_gpu.cpu().detach().numpy().squeeze()
    rec_cpu = np.reshape(rec_cpu,(img_size,img_size))
    rec_pinv[wav,:,:] = rec_cpu

    # MMSE
    model.PreP.N0 = rec_pinv[wav,:,:].max()/10 # /!\ NOT WORKING 
    rec_mmse_gpu = model.reconstruct_mmse(m)
    rec_mmse_gpu = (rec_mmse_gpu + 1) * model.PreP.N0/2
    rec_mmse_cpu = rec_mmse_gpu.cpu().detach().numpy().squeeze()
    rec_mmse[wav,:,:] = rec_mmse_cpu
    
    # Net
    rec_net_gpu = model.reconstruct(m)
    rec_net_gpu = (rec_net_gpu + 1) * model.PreP.N0/2 # /!\ NOT WORKING
    rec_net_cpu = rec_net_gpu.cpu().detach().numpy().squeeze()
    rec_net[wav,:,:] = rec_net_cpu;

# save
if save_root:
    
    (save_root/data_folder).mkdir(parents=True, exist_ok=True)
    
    full_path_pinv = save_root / data_folder / (data_file_prefix + f'_pinv_M_{M}')
    full_path_mmse = save_root / data_folder / (data_file_prefix + f'_mmmse_M_{M}')
    full_path_net = save_root / data_folder / (data_file_prefix + f'_net_M_{M}')
    
    np.save(full_path_pinv, rec_pinv)
    np.save(full_path_mmse, rec_mmse)
    np.save(full_path_net, rec_net)

#%% Plot pinv
wav_0 = 50 
wav_1 = 500
wav_2 = 1500

fig , axs = plt.subplots(1,3)
#
im0 = axs[0].imshow(rec_pinv[wav_0,:,:], cmap='gray')
fig.colorbar(im0, ax=axs[0])
axs[0].set_title(f"Pinv, channel = {wav_0}")
#
im1 = axs[1].imshow(rec_pinv[wav_1,:,:], cmap='gray')
fig.colorbar(im1, ax=axs[1])
axs[1].set_title(f"Pinv, channel = {wav_1}")
#
im2 = axs[2].imshow(rec_pinv[wav_2,:,:], cmap='gray')
fig.colorbar(im2, ax=axs[2])
axs[2].set_title(f"Pinv, channel = {wav_2}")

# save
if save_root:
    (save_root/data_folder).mkdir(parents=True, exist_ok=True)
    full_path = save_root / data_folder / (data_file_prefix + f'_pinv_M_{M}.png')
    fig.savefig(full_path)

#%% Plot mmse
fig , axs = plt.subplots(1,3)
#
im0 = axs[0].imshow(rec_mmse[wav_0,:,:], cmap='gray')
fig.colorbar(im0, ax=axs[0])
axs[0].set_title(f"MMSE, channel = {wav_0}")
#
im1 = axs[1].imshow(rec_mmse[wav_1,:,:], cmap='gray')
fig.colorbar(im1, ax=axs[1])
axs[1].set_title(f"MMSE, channel = {wav_1}")
#
im2 = axs[2].imshow(rec_mmse[wav_2,:,:], cmap='gray')
fig.colorbar(im2, ax=axs[2])
axs[2].set_title(f"MMSE, channel = {wav_2}")

# save
if save_root:
    (save_root/data_folder).mkdir(parents=True, exist_ok=True)
    full_path = save_root / data_folder / (data_file_prefix + f'_mmse_M_{M}.png')
    fig.savefig(full_path)

#%% Plot net
fig , axs = plt.subplots(1,3)
#
im0 = axs[0].imshow(rec_net[wav_0,:,:], cmap='gray')
fig.colorbar(im0, ax=axs[0])
axs[0].set_title(f"Net, channel = {wav_0}")
#
im1 = axs[1].imshow(rec_net[wav_1,:,:], cmap='gray')
fig.colorbar(im1, ax=axs[1])
axs[1].set_title(f"Net, channel = {wav_1}")
#
im2 = axs[2].imshow(rec_net[wav_2,:,:], cmap='gray')
fig.colorbar(im2, ax=axs[2])
axs[2].set_title(f"Net, channel = {wav_2}")

# save
if save_root:
    (save_root/data_folder).mkdir(parents=True, exist_ok=True)
    full_path = save_root / data_folder / (data_file_prefix + f'_net_M_{M}.png')
    fig.savefig(full_path)

#%% Plot sum
fig , axs = plt.subplots(1,3)
#
im0 = axs[0].imshow(rec_pinv.sum(axis = 0), cmap='gray')
fig.colorbar(im0, ax=axs[0])
axs[0].set_title("Pinv, sum of all channels")
#
im1 = axs[1].imshow(rec_mmse.sum(axis = 0), cmap='gray')
fig.colorbar(im1, ax=axs[1])
axs[1].set_title("MMSE, sum of all channels")
#
im2 = axs[2].imshow(rec_net.sum(axis = 0), cmap='gray')
fig.colorbar(im2, ax=axs[2])
axs[2].set_title("NET, sum of all channels")

# save
if save_root:
    (save_root/data_folder).mkdir(parents=True, exist_ok=True)
    full_path = save_root / data_folder / (data_file_prefix + f'_sum_M_{M}.png')
    fig.savefig(full_path)