# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:02:41 2023

@author: ducros
"""
import numpy as np
from pathlib import Path
from misc_dev import spectral_colorization

#%% Load spatio-spectral data
data_folder = Path('./data/2023_03_13_2023_03_14_eGFP_DsRed_3D/Preprocess') #_registered
data_filename_pos = data_folder / 'T6_RUN0001_2023_03_14_Had_512_128_128_pos.npy'
data_filename_neg = data_folder / 'T6_RUN0001_2023_03_14_Had_512_128_128_neg.npy'

lambda_min = 488
lambda_max = 610

nbin = 20*4
M_pos = np.load(data_filename_pos) - (2**15-1)*nbin
M_neg = np.load(data_filename_neg) - (2**15-1)*nbin
M_diff = M_pos - M_neg
 
wav = np.linspace(lambda_min, lambda_max, 128)

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
import sys
sys.path.append('./fonction')
from matrix_tools import bining_line

save_fig = False

if save_fig:
    save_folder = data_folder / data_filename_pos.name.replace('_Had_512_128_128_pos.npy','').replace('_2023_03_14','').replace('_2023_03_13','')
    save_folder.mkdir(parents=True, exist_ok=True)
    print(f' Save colorized spatio-spectral measurement in {save_folder}')
    
    # to get square images (looks better)
    M_pos_color = bining_line(M_pos_color, 4)
    M_neg_color = bining_line(M_neg_color, 4)
    M_diff_color = bining_line(M_diff_color, 4)
    
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
recon = 'tikhonet50_div1.5_registered'  # 'pinv', 'pinv_shift', 'pinv_registered', 'tikhonet50_div1.5' 'tikhonet50_div1.5_shift' 
Nl, Nh, Nc = 512, 512, 128
T_list = [3, 9, 15] # slice number

for t in T_list:
    
    if t<6:
        date = '2023_03_13'
        Run = f'RUN{t+1:04}'
    else:
        date = '2023_03_14'
        Run = f'RUN{t-5:04}'
    
    filename = f'T{t}_{Run}_{date}_rec_{recon}_exp_{Nl}x{Nh}x{Nc}.npy'
    
    print(f'Colorize hypercube {filename}')
    F = np.load(Path(data_folder.parents[0]) / 'Reconstruction/hypercube'/ recon / filename)
    
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
        save_folder = data_folder.parents[0] / 'Reconstruction' / 'hypercube' /  recon / (f'T{t}_{Run}_{date}' + '_color')
        save_folder.mkdir(parents=True, exist_ok=True)
                   
        for ind in range(Nc):
            print(f'\r-- Save channel: {ind}', end="", flush=True)
            plt.imsave(save_folder / f'hcube_{ind}.png', F_color[:,:,ind])
        print()