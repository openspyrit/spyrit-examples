# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:52:22 2023

@author: ducros
"""
#%% to debug needs to run
import collections
collections.Callable = collections.abc.Callable

fig_folder = './figure/'

#%% Load experimental hadamard patterns (after split compensation)
from pathlib import Path
from spyrit.misc.disp import add_colorbar, noaxis
import matplotlib.pyplot as plt
import numpy as np

M = 64
N = 512
data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/'
mat_folder = '/Reconstruction/Mat_rc/'

# load
H_exp = np.load(Path(data_folder + mat_folder) / f'hadamard_matrix_{M}_{N}.npy')
H_exp /= H_exp[0,16:500].mean()
   
#%%
channel = 10, 55, 100       # to be plotted
Nl, Nh, Nc = 512, 64, 1280  # shape of preprocessed data
save_rec = True
save_fig = True
 
# Number of channels. We choose 10 to get 128 channels, which matches the 
# dimension of the hypercubes that were reconstructed for the zebrafish samples.
c_step = 10
lambda_central_list = np.arange(510, 598, 12, dtype=int)  # Central channel in nanometer

lambda_all = np.linspace(500, 608, Nc)

#%% Init physics operators for Hadamard patterns | Alternative
import torch
from spyrit.core.meas  import Linear
from spyrit.core.recon import PseudoInverse

M = 64
N = 512

linop = Linear(torch.from_numpy(H_exp), pinv=True, meas_shape = (1,512)) 
# meas_shape = (1,512) allows to work in 1D, i.e., the forward model
# applies to last dimension only. meas_shape = (512, 1) works as well.

recon = PseudoInverse()

#%% Pinv
# raw data
data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/'
Run = 'RUN0004' 

# save
save_rec = True
save_fig = True
ext_fig = 'pdf'
dpi_fig = 600

# Load net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
linop.to(device) 

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
nc, nl, nh = prep_neg.shape
y = prep_pos - prep_neg #/np.expand_dims(prep_pos[:,:,0], axis=2)
y = torch.from_numpy(y)

print(f'Loaded: {filename[:-8]}')
print(f'Pos data: range={prep_pos.max() - prep_pos.min()} counts; mean={prep_pos.mean()} counts')  

y = y.to(device)
y = y.view(-1,1,N,M) 

# init output
rec = np.zeros((N, N))

for lambda_central in lambda_central_list:
    with torch.no_grad():
        c_central = np.argmin((lambda_all-lambda_central)**2) # Central channel
        m = y[c_central-c_step:c_central+c_step].sum(0, keepdim=True).to(device, torch.float32)
        print(f'reconstructing spectral bin from channels: {c_central-c_step}--{c_central+c_step}')
        rec_gpu = recon(m, linop)
        rec = rec_gpu.cpu().detach().numpy().squeeze()
        rec = np.moveaxis(rec, 0, -1) # spectral channel is now the last axis
        rec = np.flip(rec,0)
    
    if save_rec:
        save_folder = 'Reconstruction/hypercube'  
        Path(data_folder+save_folder).mkdir(parents=True, exist_ok=True)
            
        save_filename = f'had_{lambda_central}_{N}x{N}.npy'
        np.save(Path(data_folder+save_folder) / save_filename, rec)
            
    # Plot
    fig, axs = plt.subplots(1, 1, figsize=(5,5))
    im = axs.imshow(rec)
    axs.set_title(f'{lambda_central} nm')
    add_colorbar(im, 'bottom')
        
    noaxis(axs)
        
    if save_fig:
        save_filename = f'had_{lambda_central}_{N}x{N}.{ext_fig}'
        plt.savefig(Path(data_folder+save_folder)/save_filename, bbox_inches='tight', dpi=dpi_fig)


#%% Load experimental canonical patterns (after split compensation)
from pathlib import Path
from spyrit.misc.disp import add_colorbar, noaxis
import matplotlib.pyplot as plt
import numpy as np
import torch

M = 64
N = 512
data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/'
mat_folder = '/Reconstruction/Mat_rc/'

# load
H_exp = np.load(Path(data_folder + mat_folder) / f'canonical_diff_matrix_{M}_{N}.npy')
#H_exp /= H_exp[0,16:500].mean()

#%% Init physics operators for CANONICAL patterns | Alternative
import torch
from spyrit.core.meas  import Linear
from spyrit.core.recon import PseudoInverse
import matplotlib.pyplot as plt

M = 64
N = 512

linop = Linear(torch.from_numpy(H_exp), pinv=True, meas_shape = (1,512))
recon = PseudoInverse()

#%% Pinv for CANONICAL patterns
channel = 10, 55, 100  # to be plotted
Nl, Nh, Nc = 512, 64, 1280 # shape of preprocessed data

# raw data
data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/'
Run_can = 'RUN0006' 
Run_dark = 'RUN0003'  

# save
save_rec = True
save_fig = True

# Load net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

linop.to(device)  

# Load prep data
save_folder = '/Preprocess/'
filename = f'{Run_can}_Can_{Nl}_{Nh}_{Nc}_can.npy'
prep_can = np.load(Path(data_folder+save_folder) / filename)

filename = f'{Run_dark}_Can_{Nl}_{Nh}_{Nc}_dark.npy'
prep_dark = np.load(Path(data_folder+save_folder) / filename)

# The mean of the dark stack is substracted
prep_diff = prep_can - prep_dark.mean(1,keepdims=True)
prep_diff = np.moveaxis(prep_diff, -1, 0)
y = torch.from_numpy(prep_diff)

print(f'Loaded: {filename[:-8]}')
print(f'Pos data: range={prep_diff.max() - prep_diff.min()} counts; mean={prep_diff.mean()} counts')  

y = y.to(device)
y = y.view(-1,1,N,M) 

# init output
rec = np.zeros((N, N))


for lambda_central in lambda_central_list:
    
    with torch.no_grad():
        c_central = np.argmin((lambda_all-lambda_central)**2) # Central channel
        
        m = y[c_central-c_step:c_central+c_step].sum(0, keepdim=True).to(device, torch.float32)
        print(f'reconstructing spectral bin from channels: {c_central-c_step}--{c_central+c_step}')
        rec_gpu = recon(m, linop)
        rec = rec_gpu.cpu().detach().numpy().squeeze()
                    
        rec = np.moveaxis(rec, 0, -1) # spectral channel is now the last axis
        rec = np.flip(rec,0)
        
        if save_rec:
            save_folder = 'Reconstruction/hypercube'  
            Path(data_folder+save_folder).mkdir(parents=True, exist_ok=True)
            
            save_filename = f'can_{lambda_central}_{N}x{N}.npy'
            np.save(Path(data_folder+save_folder) / save_filename, rec)
            
        # Plot
        fig, axs = plt.subplots(1, 1, figsize=(5,5))
        im = axs.imshow(rec)
        axs.set_title(f'{lambda_central} nm')
        add_colorbar(im, 'bottom')
            
        noaxis(axs)
            
        if save_fig:
            save_filename = f'can_{lambda_central}_{N}x{N}.{ext_fig}'
            plt.savefig(Path(data_folder+save_folder)/save_filename, bbox_inches='tight', dpi=dpi_fig)