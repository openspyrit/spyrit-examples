# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:57:59 2023

@author: ducros
"""
#%% to debug needs to run
import collections
collections.Callable = collections.abc.Callable

fig_folder = './figure/'

#%% Load experimental hadamard patterns (after split compensation)
from pathlib import Path
from spyrit.misc.disp import imagesc, add_colorbar, noaxis
import matplotlib.pyplot as plt
from spyrit.misc.walsh_hadamard import walsh_matrix
import numpy as np
import torch

M = 64
N = 512
data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/'
mat_folder = '/Reconstruction/Mat_rc/'

nbin = 20*4
mudark = 106.0
sigdark = 5.0
gain = 2.65

# load
H_exp = np.load(Path(data_folder + mat_folder) / f'hadamard_matrix_{M}_{N}.npy')
H_exp /= H_exp[0,16:500].mean()
   
#%% Init physics operators for experimental patterns
import torch
from spyrit.core.meas import LinearSplit
from spyrit.core.prep import SplitPoisson
from spyrit.core.noise import Poisson
import matplotlib.pyplot as plt

M = 64
N = 512
alpha = 1e1 # in photons/pixels

linop = LinearSplit(H_exp, pinv=True)
noise = Poisson(linop, alpha)
prep  = SplitPoisson(alpha, linop)
prep.set_expe(gain, mudark, sigdark, nbin)

#%% Pinv
from recon_dev import Pinv1Net

save_rec = True
save_fig = True

channel = 10, 55, 100  # to be plotted
Nl, Nh, Nc = 512, 64, 128 # shape of preprocessed data

# raw data
save_tag = True
data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/'
Run = 'RUN0004' 

# Load net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

recnet = Pinv1Net(noise, prep)
recnet.to(device)  

# Reconstruct all channels per batch
n_batch = 8 # a power of two
n_wav = Nc // n_batch


# Load prep data
save_folder = '/Preprocess/'
filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_pos.npy'
prep_pos = np.load(Path(data_folder+save_folder) / filename)

filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_neg.npy'
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
        
        save_filename = f'Had_rec_pinv_dark_mean_{N}x{N}x{nc}.npy'
        np.save(Path(data_folder+save_folder) / save_filename, rec)
        
        save_filename = f'Had_beta_pinv_dark_mean_{N}x{N}x{nc}.npy'
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
        save_filename = f'Had_rec_pinv_dark_mean_{N}x{N}x{nc}.pdf'
        plt.savefig(Path(data_folder+save_folder)/save_filename, bbox_inches='tight', dpi=600)
    
del recnet

#%% Load experimental canonical patterns (after split compensation)
from pathlib import Path
from spyrit.misc.disp import imagesc, add_colorbar, noaxis
import matplotlib.pyplot as plt
from spyrit.misc.walsh_hadamard import walsh_matrix
import numpy as np
import torch

M = 64
N = 512
data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/'
mat_folder = '/Reconstruction/Mat_rc/'

nbin = 20*4
mudark = 106.0
sigdark = 5.0
gain = 2.65

# load
H_exp = np.load(Path(data_folder + mat_folder) / f'canonical_diff_matrix_{M}_{N}.npy')
#H_exp /= H_exp[0,16:500].mean()

#%% Init physics operators for experimental patterns
import torch
from spyrit.core.meas import Linear
from spyrit.core.prep import DirectPoisson
from spyrit.core.noise import Poisson
import matplotlib.pyplot as plt

M = 64
N = 512
alpha = 1e1 # in photons/pixels

linop = Linear(H_exp, pinv=True)
noise = Poisson(linop, alpha)
prep  = DirectPoisson(alpha, linop)

#%% Pinv
from recon_dev import Pinv1Net

save_rec = True
save_fig = True

channel = 10, 55, 100  # to be plotted
Nl, Nh, Nc = 512, 64, 128 # shape of preprocessed data

# raw data
save_tag = True
data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/'
Run_can = 'RUN0006' 
Run_dark = 'RUN0003'  

# Load net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

recnet = Pinv1Net(noise, prep)
recnet.to(device)  

# Reconstruct all channels per batch
n_batch = 8 # a power of two
n_wav = Nc // n_batch


# Load prep data
save_folder = '/Preprocess/'
filename = f'{Run_can}_Can_{Nl}_{Nh}_{Nc}_can.npy'
prep_can = np.load(Path(data_folder+save_folder) / filename)

filename = f'{Run_dark}_Can_{Nl}_{Nh}_{Nc}_dark.npy'
prep_dark = np.load(Path(data_folder+save_folder) / filename)

n_dark = 42
prep_diff = prep_can - prep_dark[:,np.newaxis,n_dark,:] # keep dimension information

# param #2
# background  = (2**15-1)*nbin
# prep_can  = prep_can - background
# prep_dark = prep_dark - background

# spectral dimension comes first
prep_diff = np.moveaxis(prep_diff, -1, 0)
nc, nl, nh = prep_diff.shape
y = torch.from_numpy(prep_diff)

print(f'Loaded: {filename[:-8]}')
print(f'Pos data: range={prep_diff.max() - prep_diff.min()} counts; mean={prep_diff.mean()} counts')  

y = y.to(device)
y = y.view(-1,1,N,M) 

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
        
        save_filename = f'Can_rec_pinv_dark_{n_dark}_{N}x{N}x{nc}.npy'
        np.save(Path(data_folder+save_folder) / save_filename, rec)
        
        save_filename = f'Can_beta_pinv_dark_{n_dark}_{N}x{N}x{nc}.npy'
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
        save_filename = f'Can_rec_pinv_dark_{n_dark}_{N}x{N}x{nc}.pdf'
        plt.savefig(Path(data_folder+save_folder)/save_filename, bbox_inches='tight', dpi=600)    


# Alternative where the mean of the dark stack is substracted
prep_diff = prep_can - prep_dark.mean(1,keepdims=True)
prep_diff = np.moveaxis(prep_diff, -1, 0)
y = torch.from_numpy(prep_diff)

print(f'Loaded: {filename[:-8]}')
print(f'Pos data: range={prep_diff.max() - prep_diff.min()} counts; mean={prep_diff.mean()} counts')  

y = y.to(device)
y = y.view(-1,1,N,M) 

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
        
        save_filename = f'Can_rec_pinv_dark_mean_{N}x{N}x{nc}.npy'
        np.save(Path(data_folder+save_folder) / save_filename, rec)
        
        save_filename = f'Can_beta_pinv_dark_mean_{N}x{N}x{nc}.npy'
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
        save_filename = f'Can_rec_pinv_dark_mean_{N}x{N}x{nc}.pdf'
        plt.savefig(Path(data_folder+save_folder)/save_filename, bbox_inches='tight', dpi=600)
    
del recnet