# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:02:41 2023

@author: ducros
"""
import numpy as np
from pathlib import Path
from spyrit.misc.color import spectral_colorization # v2.3.4

import sys
sys.path.append('./fonction')
from matrix_tools import bining_line

#%% Load spatio-spectral data
data_folder = Path('./data/2023_02_28_mRFP_DsRed_3D/Preprocess')
data_filename_pos = data_folder / 'RUN0015_Had_512_128_128_pos.npy'
data_filename_neg = data_folder / 'RUN0015_Had_512_128_128_neg.npy'

lambda_min = 538
lambda_max = 660

nbin = 20*4
M_pos = np.load(data_filename_pos) - (2**15-1)*nbin
M_neg = np.load(data_filename_neg) - (2**15-1)*nbin
M_diff = M_pos - M_neg
 
wav = np.linspace(lambda_min, lambda_max, 128)

# to get square images (looks better)
M_pos = bining_line(M_pos, 4)
M_neg = bining_line(M_neg, 4)
M_diff = bining_line(M_diff, 4)

M_pos_color  = spectral_colorization(M_pos, wav, (0,2))
M_neg_color  = spectral_colorization(M_neg, wav, (0,2))
M_diff_color = spectral_colorization(M_diff, wav, (0,2))
    
#%% Plot three spatio-spectral images
import matplotlib.pyplot as plt

offset = 0
ind_plot = [0 + offset ,1 + offset, 2+ offset, 3+ offset]

f, axs = plt.subplots(1,2*len(ind_plot))

for i, ind in enumerate(ind_plot):
    print(ind)
    im = axs[2*i].imshow(M_pos[:,ind,:], cmap='gray') 
    axs[2*i].get_xaxis().set_visible(False)
    axs[2*i].set_title(f'pos: {ind}')
    #
    im = axs[2*i+1].imshow(M_neg[:,ind,:], cmap='gray') 
    axs[2*i+1].get_xaxis().set_visible(False)
    axs[2*i+1].set_title(f'neg: {ind}')

f, axs = plt.subplots(1,2*len(ind_plot))

for i, ind in enumerate(ind_plot):
    im = axs[2*i].imshow(M_pos_color[:,ind,:,:]) 
    axs[2*i].get_xaxis().set_visible(False)
    axs[2*i].set_title(f'pos: {ind}')
    #
    im = axs[2*i+1].imshow(M_neg_color[:,ind,:,:]) 
    axs[2*i+1].get_xaxis().set_visible(False)
    axs[2*i+1].set_title(f'neg: {ind}')
    
f, axs = plt.subplots(1,len(ind_plot))

for i, ind in enumerate(ind_plot):
    im = axs[i].imshow(M_diff_color[:,ind,:,:]) 
    axs[i].get_xaxis().set_visible(False)
    axs[i].set_title(f'diff: {ind}')

#%% Save all spatio-spectral images
save_fig = True

if save_fig:
    save_folder = data_folder / data_filename_pos.name.replace('_Had_512_128_128_pos.npy','')
    save_folder.mkdir(parents=True, exist_ok=True)
    print(f' Save colorized spatio-spectral measurement in {save_folder}')
    
    for ind in range(128):
        # save
        print(f'\r-- channel: {ind}', end="", flush=True)
        plt.imsave(save_folder / f'pos_{ind}.png',M_pos_color[:,ind,:,:])
        plt.imsave(save_folder / f'neg_{ind}.png',M_neg_color[:,ind,:,:])
        plt.imsave(save_folder / f'diff_{ind}.png',M_diff_color[:,ind,:,:])
    
#%% Colorize hypercubes (after spatial reconstruction)
# Plot
save_fig = True

# hypercubes
load_path = './data/2023_02_28_mRFP_DsRed_3D'
recon = 'tikhonet50_div1.5_shift'  # 'pinv' 'tikhonet50_div1.5' 'pinv_shift' 'tikhonet50_div1.5_shift'
Nl, Nh, Nc = 512, 512, 128
z_list = [3, 10, 15] # slice number

# Load hypercubes
T_list = [*range(4, 8), *range(9, 25)] # slice indices, Run0008 corrupted

for z in z_list:
    Run = f'RUN{T_list[z]:04}'
    filename = f'{Run}_rec_{recon}_exp_{Nl}x{Nh}x{Nc}.npy'
    print(f'Colorize hypercube {filename}')
    F = np.load(Path(load_path) / 'Reconstruction/hypercube'/ recon / filename)
    
    # Colorize
    F_color  = spectral_colorization(F, wav, axis=(0,1)) 
    
    # Plot
    ind_plot = [5 ,55, 80, 125]
    f, axs = plt.subplots(1,len(ind_plot))
    
    for i, ind in enumerate(ind_plot):
        im = axs[i].imshow(F_color[:,:,ind]) 
        axs[i].get_xaxis().set_visible(False)
        axs[i].set_title(f'Channel: {ind}')
        
    # Save
    if save_fig:
        save_folder = data_folder.parents[0] / 'Reconstruction' / 'hypercube' /  recon / (Run + '_color')
        save_folder.mkdir(parents=True, exist_ok=True)
                   
        for ind in range(Nc):
            print(f'\r-- Save channel: {ind}', end="", flush=True)
            plt.imsave(save_folder / f'hcube_{ind}.png', F_color[:,:,ind])
        print()