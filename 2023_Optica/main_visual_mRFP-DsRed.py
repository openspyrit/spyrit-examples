# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 19:42:40 2023

@author: ducros
"""

#%% Load unmixed / filtered data
import numpy as np
from pathlib import Path

# Load data
Nl = 512 # number of pixcels along the y dimensions 
Nh = 512 # number of measured Walsh_Hadmard coefficients (correpond to the h dimensions)
Nc = 128 # number of channels

load_path = Path('./data/2023_02_28_mRFP_DsRed_3D')
T_list = range(1,27)    # slice indices

suffix = '_calib_mrfp12_shift' # '', '_calib_mrfp1_shift', '_calib_mrfp12_shift'
recon = 'pinv_shift'  # 'pinv_shift', 'tikhonet50_div1.5_shift'
method_unmix = 'NNLS' # 'NNLS''_UCLS'

member_list = ['DsRed','mRFP','Autofluo'] #['Dsred','mRFP','AF']
filter_list = ['orange filter','red filter']

# unmixed data
folder_unmix = recon + '_' + method_unmix
fullpath = load_path / ('Unmixing' + suffix) / folder_unmix
print(fullpath)  
abond_unmix_4D = np.load(fullpath / 'abondance.npy')
abond_unmix_4D = np.rot90(abond_unmix_4D, k=2, axes=(0, 1))

# filtered data
folder_filter = recon
fullpath = load_path / ('Filtering' + suffix) / folder_filter
print(fullpath)  
abond_filter_4D = np.load(fullpath / 'abondance.npy')
abond_filter_4D = np.rot90(abond_filter_4D, k=2, axes=(0, 1))

#%%
def colorize(im, color, clip_percentile=0.1):
    """
    Helper function to create an RGB image from a single-channel image using a 
    specific color.
    """
    # Check that we just have a 2D image
    if im.ndim > 2 and im.shape[2] != 1:
        raise ValueError('This function expects a single-channel image!')
        
    # Rescale the image according to how we want to display it
    im_scaled = im.astype(np.float32) - np.percentile(im, clip_percentile)
    im_scaled = im_scaled / np.percentile(im_scaled, 100 - clip_percentile)
    print(f'Norm: min={np.percentile(im, clip_percentile)}, max={np.percentile(im_scaled, 100 - clip_percentile)}')
    print(f'New:  min={im_scaled.min()}, max={im_scaled.max()}')
    im_scaled = np.clip(im_scaled, 0, 1)
    
    # Need to make sure we have a channels dimension for the multiplication to work
    im_scaled = np.atleast_3d(im_scaled)
    
    # Reshape the color (here, we assume channels last)
    color = np.asarray(color).reshape((1, 1, -1))
    return im_scaled * color

#%% Visualize in color (could be merged with the SPIM image)
# from https://bioimagebook.github.io/chapters/1-concepts/4-colors/python.html
import matplotlib.pyplot as plt
import os

save_tag = True

z_list = [*range(4, 8), *range(9, 25)] # slice indices, Run0008 corrupted
sat = 0.15 # in percentage

# loop over all z-slices
for z, t in enumerate(z_list):
    
    # Unmixing
    test = abond_unmix_4D[:,:,z,0:3]
    
    # The color we provide gives RGB values in order, between 0 and 1
    im_unmix_1 = colorize(test[..., 0], (1, 0, 1), sat)
    im_unmix_2 = colorize(test[..., 1], (0, 1, 1), sat)
    im_unmix_3 = colorize(test[..., 2], (0.5, 0.5, 0), sat)
    im_unmix = im_unmix_1 + im_unmix_2 #+ im_unmix_3
    #im_unmix /= im_unmix.max()
    
    # Filtering
    test = abond_filter_4D[:,:,z,0:3]
    test = np.clip(test, 0, None)
    
    # The color we provide gives RGB values in order, between 0 and 1
    im_filt_1 = colorize(test[..., 0], (1, 0, 1), sat)
    im_filt_2 = colorize(test[..., 1], (0, 1, 1), sat)
    im_filt = im_filt_1 + im_filt_2 
    
    # plot
    fig, axs = plt.subplots(2, 3, figsize=(3*3,2*3))
    axs[0,0].imshow(im_filt_1)
    axs[0,0].axis(False)
    axs[0,0].set_title(filter_list[1])
    
    axs[0,1].imshow(im_filt_2)
    axs[0,1].axis(False)
    axs[0,1].set_title(filter_list[0])
    
    axs[0,2].imshow(im_filt)
    axs[0,2].axis(False)
    axs[0,2].set_title(filter_list[1] + ' + ' + filter_list[0])
    
    axs[1,0].imshow(im_unmix_1)
    axs[1,0].axis(False)
    axs[1,0].set_title(member_list[0])
    
    axs[1,1].imshow(im_unmix_2)
    axs[1,1].axis(False)
    axs[1,1].set_title(member_list[1])
    
    axs[1,2].imshow(im_unmix)
    axs[1,2].axis(False)
    axs[1,2].set_title(member_list[0] + ' + ' + member_list[1])
    
    if save_tag:
        save_path = load_path / ('Visualisation' + suffix) / folder_unmix
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        filename = f'T{t}.svg'
        print(filename)
        plt.savefig(save_path / filename, bbox_inches='tight', dpi=600)
        plt.close(fig)
        
#%% plot all quantitative maps independently
save_tag = True

# loop over all z-slices
for z, t in enumerate(z_list):
    
    # Unmixing
    test = abond_unmix_4D[:,:,z,0:3]
    im_unmix_all = np.zeros(test.shape + (3,)) # add an RGB dimension
    
    # The color we provide gives RGB values in order, between 0 and 1
    im_unmix_all[...,0,:] = colorize(test[..., 0], (1, 0, 1), sat)
    im_unmix_all[...,1,:] = colorize(test[..., 1], (0, 1, 1), sat)
    im_unmix_all[...,2,:] = colorize(test[..., 2], (0.5, 0.5, 0), sat)
    
    if save_tag:
        save_path = load_path / ('Visualisation' + suffix) / folder_unmix / f'T{t}'
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        for m in range(3):
            plt.imsave(save_path / f'qmap_{m}.png', im_unmix_all[...,m,:])
        plt.imsave(save_path / 'qmap_12.png', np.clip(im_unmix_all[...,0,:] + im_unmix_all[...,1,:], 0, 1))
        plt.imsave(save_path / 'qmap_23.png', np.clip(im_unmix_all[...,1,:] + im_unmix_all[...,2,:], 0, 1))
        plt.imsave(save_path / 'qmap_13.png', np.clip(im_unmix_all[...,0,:] + im_unmix_all[...,2,:], 0, 1))
        plt.imsave(save_path / 'qmap_123.png', np.clip(im_unmix_all[...,0,:] + im_unmix_all[...,1,:] + im_unmix_all[...,2,:], 0, 1))

#%% plot all filter maps fmaps independently

# loop over all z-slices
for z, t in enumerate(z_list):
    
    # Unmixing
    test = abond_filter_4D[:,:,z,:]
    test = np.clip(test, 0, None) # remove negative values
    im_filter_all = np.zeros(test.shape + (3,))
    
    # The color we provide gives RGB values in order, between 0 and 1
    im_filter_all[...,0,:] = colorize(test[..., 0], (1, 0, 1), sat)
    im_filter_all[...,1,:] = colorize(test[..., 1], (0, 1, 1), sat)
    
    if save_tag:
        save_path = load_path / ('Visualisation' + suffix) / folder_unmix / f'T{t}'
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        for m in range(2):
            plt.imsave(save_path / f'fmap_{m}.png', im_filter_all[...,m,:])
        plt.imsave(save_path / 'fmap_12.png', np.clip(im_filter_all[...,0,:] + im_filter_all[...,1,:], 0, 1))


#%% This was an attempt with microplot (not really better that using matplotlb directly)
# from microfilm import microplot
# z = 2
# test = abond_unmix_4D[:,:,z,0:3]
# test = np.moveaxis(test, -1, 0)
# microim = microplot.microshow(images=test, cmaps=['pure_cyan', 'pure_green', 'pure_green'])
# microim = microplot.microshow(images=[test[0], test[1]], cmaps=['pure_cyan', 'YlGn'])