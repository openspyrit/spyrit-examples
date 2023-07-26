# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 20:09:51 2023

@author: ducros
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 18:06:44 2023

Reconstructions with data simulated using experimental measurements 

@author: ducros
"""
#%% to debug needs to run
import collections
collections.Callable = collections.abc.Callable

fig_folder = './figure/'

#%% Load experimental measurement matrix (after split compensation)
from pathlib import Path
from spyrit.misc.disp import imagesc, add_colorbar, noaxis
import matplotlib.pyplot as plt
from spyrit.misc.walsh_hadamard import walsh_matrix
import numpy as np
import torch

M = 128
N = 512
data_folder = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D/'
mat_folder = '/Reconstruction/Mat_rc/'

nbin = 20*4
mudark = 105.0
sigdark = 5.0
gain = 2.6

# load
H_exp = np.load(Path(data_folder + mat_folder) / f'motifs_Hadamard_{M}_{N}.npy')
#H_exp /= H_exp.max()
H_exp /= H_exp[0,16:500].mean()
# H_exp = H_exp #/ H_exp[0,:]
H_tar = walsh_matrix(N)
H_tar = H_tar[:M]

# plot
f, axs = plt.subplots(2, 1)
axs[0].set_title('Target measurement patterns')
im = axs[0].imshow(H_tar, cmap='gray') 
add_colorbar(im, 'bottom')
axs[0].get_xaxis().set_visible(False)

axs[1].set_title('Experimental measurement patterns (diff)')
im = axs[1].imshow(np.flip(H_exp,1).copy(), cmap='gray') 
add_colorbar(im, 'bottom')
axs[1].get_xaxis().set_visible(False)

axs[1].set_title('Experimental measurement patterns (sum)')
im = axs[1].imshow(np.flip(H_exp,1).copy(), cmap='gray') 
add_colorbar(im, 'bottom')
axs[1].get_xaxis().set_visible(False)

#plt.savefig(Path(fig_folder) / 'patterns', bbox_inches='tight', dpi=600)

    
#%% Init physics operators for experimental patterns
import torch
from recon_dev import Pinv1Net
from spyrit.core.meas import LinearSplit
from spyrit.core.prep import SplitPoisson
from spyrit.core.noise import Poisson
import matplotlib.pyplot as plt

M = 128
N = 512
alpha = 1e1 # in photons/pixels

linop = LinearSplit(H_exp, pinv=True)
noise = Poisson(linop, alpha)
prep  = SplitPoisson(alpha, linop)
prep.set_expe(gain, mudark, sigdark, nbin)


#%% Tikhonet + unet
from spyrit.core.nnet import Unet
from spyrit.core.train import load_net
from recon_dev import Tikho1Net, Tikhonov

save_rec = True
save_fig = True

div = 1.5

alpha = 10
channel = 10, 55, 100  # to be plotted
Nl, Nh, Nc = 512, 128, 128 # shape of preprocessed data

# raw data
save_tag = False
data_folder = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D/'
data_subfolder = 'data_2023_03_14/'
T_list = range(1,7)    # slice indices

# covariance prior in the image domain
stat_folder = './stat/'
cov_file   = f'Cov_1_{N}x{N}.npy'
mean_file   = f'Average_1_{N}x{N}.npy'
mean  = np.load(Path(stat_folder) / mean_file)
sigma = np.load(Path(stat_folder) / cov_file)

# Load net
net_prefix = f'tikho-net_unet_imagenet_ph_{alpha}'
net_suffix = 'N_512_M_128_epo_20_lr_0.001_sss_10_sdr_0.5_bs_20_reg_1e-07'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

recnet = Tikho1Net(noise, prep, sigma, Unet())
recnet.to(device)  
title = './model/' + net_prefix + '_exp_' + net_suffix
load_net(title, recnet, device, False)
recnet.tikho = Tikhonov(recnet.acqu.meas_op.get_H(), 
                        torch.as_tensor(sigma/div, 
                                        dtype=torch.float32, 
                                        device=device)
                        )
recnet.eval() # Mandantory when batchNorm is used

# Reconstruct all channels per batch
n_batch = 8 # a power of two
n_wav = Nc // n_batch

for t in T_list:
    if t<6:
        #Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/data_2023_03_13/'
        date = '2023_03_13'
        Run = f'RUN{t+1:04}'
    else:
        #Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/data_2023_03_14/'
        date = '2023_03_14'
        Run = f'RUN{t-5:04}'

    # Load prep data
    save_folder = '/Preprocess/'
    filename = f'T{t}_{Run}_{date}_Had_{Nl}_{Nh}_{Nc}_pos.npy'
    prep_pos = np.load(Path(data_folder+save_folder) / filename)
    
    filename = f'T{t}_{Run}_{date}_Had_{Nl}_{Nh}_{Nc}_neg.npy'
    prep_neg =  np.load(Path(data_folder+save_folder) / filename)
    
    # spectral dimension comes first
    prep_pos = np.moveaxis(prep_pos, -1, 0)
    prep_neg = np.moveaxis(prep_neg, -1, 0)
    
    # param #2
    background  = (2**15-1)*nbin
    
    prep_pos = prep_pos - background
    prep_neg = prep_neg - background
    
    nc, nl, nh = prep_neg.shape
    y = np.zeros((nc, nl, 2*nh))
    y[:,:,::2]  = prep_pos #/np.expand_dims(prep_pos[:,:,0], axis=2)
    y[:,:,1::2] = prep_neg #/np.expand_dims(prep_pos[:,:,0], axis=2)
    y = torch.from_numpy(y)
    
    print(f'Loaded: {filename[:-8]}')
    print(f'Pos data: range={prep_pos.max() - prep_pos.min()} counts; mean={prep_pos.mean()} counts')  
    
    y = y.to(device)
    y = y.view(-1,1,N,2*M) 
    
    rec = np.zeros((Nc, N, N))
    beta = np.zeros(Nc)
    
    with torch.no_grad():
        for b in range(n_batch):       
            ind = range(b*n_wav, (b+1)*n_wav)            
            m = y[ind,:,:].to(device, torch.float32)
            
            print(f'reconstructing batch {ind.start}--{ind.stop}')
            rec_gpu, beta_gpu = recnet.reconstruct_expe(m)
            rec[ind,:,:] = rec_gpu.cpu().detach().numpy().squeeze()
            beta[ind] = beta_gpu.cpu().detach().numpy().squeeze()
            
            print(beta[ind])
                    
        rec = np.moveaxis(rec, 0, -1) # spectral channel is now the last axis
        
        if save_rec:
            save_folder = 'Reconstruction/hypercube'  
            Path(data_folder+save_folder).mkdir(parents=True, exist_ok=True)
            
            save_filename = filename[:-20].replace('Had', f'rec_tikhonet{alpha}_div{div}_exp') + f'_{N}x{N}x{nc}.npy'
            np.save(Path(data_folder+save_folder) / save_filename, rec)
            
            save_filename = filename[:-20].replace('Had', f'beta_tikhonet{alpha}_div{div}_exp') + f'_{N}x{N}x{nc}.npy'
            np.save(Path(data_folder+save_folder) / save_filename, beta)
            
        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(10,5))
        fig.suptitle(f'Tikho + Unet; Sigma / {div}; Trained at {alpha} photons; Experimental patterns')
        
        for i, c in enumerate(channel):
            im = axs[i].imshow(rec[:,:,c])
            axs[i].set_title(f'Channel: {c}; gain x alpha: {beta[c]:.4}')
            plt.colorbar(im, ax=axs[i])
            
        if save_fig:
            save_filename = filename[:-20].replace('Had',f'rec_tikhonet{alpha}_div{div}_exp') + f'_{N}x{N}x{nc}.png'
            plt.savefig(Path(data_folder+save_folder)/save_filename, bbox_inches='tight', dpi=600)
    
del recnet

#%% Pinv
from recon_dev import Pinv1Net

save_rec = True
save_fig = True

channel = 10, 55, 100  # to be plotted
Nl, Nh, Nc = 512, 128, 128 # shape of preprocessed data

# raw data
data_folder = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D/'
T_list = range(21,27)    # slice indices

# Load net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

recnet = Pinv1Net(noise, prep)
recnet.to(device)  

# Reconstruct all channels per batch
n_batch = 8 # a power of two
n_wav = Nc // n_batch

for t in T_list:

    if t<6:
        date = '2023_03_13'
        Run = f'RUN{t+1:04}'
    else:
        date = '2023_03_14'
        Run = f'RUN{t-5:04}'

    # Load prep data
    save_folder = '/Preprocess/'
    filename = f'T{t}_{Run}_{date}_Had_{Nl}_{Nh}_{Nc}_pos.npy'
    prep_pos = np.load(Path(data_folder+save_folder) / filename)
    
    filename = f'T{t}_{Run}_{date}_Had_{Nl}_{Nh}_{Nc}_neg.npy'
    prep_neg =  np.load(Path(data_folder+save_folder) / filename)
    
    # spectral dimension comes first
    prep_pos = np.moveaxis(prep_pos, -1, 0)
    prep_neg = np.moveaxis(prep_neg, -1, 0)
    
    # param #2
    background  = (2**15-1)*nbin
    
    prep_pos = prep_pos - background
    prep_neg = prep_neg - background
    
    nc, nl, nh = prep_neg.shape
    y = np.zeros((nc, nl, 2*nh))
    y[:,:,::2]  = prep_pos #/np.expand_dims(prep_pos[:,:,0], axis=2)
    y[:,:,1::2] = prep_neg #/np.expand_dims(prep_pos[:,:,0], axis=2)
    y = torch.from_numpy(y)
    
    print(f'Loaded: {filename[:-8]}')
    print(f'Pos data: range={prep_pos.max() - prep_pos.min()} counts; mean={prep_pos.mean()} counts')  
    
    y = y.to(device)
    y = y.view(-1,1,N,2*M) 
    
    rec = np.zeros((Nc, N, N))
    beta = np.zeros(Nc)
    
    with torch.no_grad():
        for b in range(n_batch):       
            ind = range(b*n_wav, (b+1)*n_wav)            
            m = y[ind,:,:].to(device, torch.float32)
            
            print(f'reconstructing batch {ind.start}--{ind.stop}')
            rec_gpu, beta_gpu = recnet.reconstruct_expe(m)
            rec[ind,:,:] = rec_gpu.cpu().detach().numpy().squeeze()
            beta[ind] = beta_gpu.cpu().detach().numpy().squeeze()
            
            print(beta[ind])
                    
        rec = np.moveaxis(rec, 0, -1) # spectral channel is now the last axis
        
        if save_rec:
            save_folder = 'Reconstruction/hypercube'  
            Path(data_folder+save_folder).mkdir(parents=True, exist_ok=True)
            
            save_filename = filename[:-20].replace('Had', 'rec_pinv_exp') + f'_{N}x{N}x{nc}.npy'
            np.save(Path(data_folder+save_folder) / save_filename, rec)
            
            save_filename = filename[:-20].replace('Had', 'beta_pinv_exp') + f'_{N}x{N}x{nc}.npy'
            np.save(Path(data_folder+save_folder) / save_filename, beta)
            
        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(10,5))
        fig.suptitle('Pinverse solution; Experimental patterns')
        
        for i, c in enumerate(channel):
            im = axs[i].imshow(rec[:,:,c])
            axs[i].set_title(f'Ch: {c}; gain x alpha: {beta[c]:.4}')
            add_colorbar(im, 'bottom')
            
        noaxis(axs)
            
        if save_fig:
            save_filename = filename[:-20].replace('Had','rec_pinv_exp') + f'_{N}x{N}x{nc}.png'
            plt.savefig(Path(data_folder+save_folder)/save_filename, bbox_inches='tight', dpi=600)
    
del recnet