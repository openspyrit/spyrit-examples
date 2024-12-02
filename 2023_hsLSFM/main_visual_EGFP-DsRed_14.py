# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 18:42:52 2023

@author: ducros
"""

#%% Load unmixed / filtered data
import numpy as np
from pathlib import Path

# Load data
Nl = 512 # number of pixcels along the y dimensions 
Nh = 512 # number of measured Walsh_Hadmard coefficients (correpond to the h dimensions)
Nc = 128 # number of channels

load_path = Path('./data/2023_03_13_2023_03_14_eGFP_DsRed_3D')

suffix = '_shift'       # '_shift', '_calib_blind_shift'
recon = 'pinv_shift'   # 'pinv_shift' 'tikhonet50_div1.5_shift'
method_unmix = 'NNLS'               # 'NNLS''_UCLS'

member_list = ['DsRed','EGFP','Autofluo']
filter_list = ['green filter','red filter']

# unmixed data
folder_unmix = recon + '_' + method_unmix
save_path = load_path / ('Unmixing' + suffix) / folder_unmix
print(save_path)  
abond_unmix_4D = np.load(save_path / 'abondance.npy')
abond_unmix_4D = np.rot90(abond_unmix_4D, k=2, axes=(0, 1))

# filtered data
folder_filter = recon
save_path = load_path / ('Filtering' + suffix) / folder_filter
print(save_path)  
abond_filter_4D = np.load(save_path / 'abondance.npy')
abond_filter_4D = np.rot90(abond_filter_4D, k=2, axes=(0, 1))

#%% Visualize in color (could be merged with the SPIM image)
# from https://bioimagebook.github.io/chapters/1-concepts/4-colors/python.html

import matplotlib.pyplot as plt
import os
from spyrit.misc.color import colorize # v2.3.4

save_tag = True

    
z_list = range(1,27)    # slice indices

sat = 0.15              # percentage of saturated pixel
color_unmix_1 = (1, 0, 1)
color_unmix_2 = (0, 1, 1)
color_unmix_3 = (.5, .5, 0)
color_filt_1 = color_unmix_1
color_filt_2 = color_unmix_2

# loop over all z-slices
for z, t in enumerate(z_list):
    
    # Unmixing
    test = abond_unmix_4D[:,:,z,0:3]
    
    # The color we provide gives RGB values in order, between 0 and 1
    im_unmix_1 = colorize(test[..., 0], color_unmix_1, sat)
    im_unmix_2 = colorize(test[..., 1], color_unmix_2, sat)
    im_unmix_3 = colorize(test[..., 2], color_unmix_3, sat)
    im_unmix = im_unmix_1 + im_unmix_2 #+ im_unmix_3
    #im_unmix /= im_unmix.max()
    
    # Filtering
    test = abond_filter_4D[:,:,z,0:3]
    test = np.clip(test, 0, None)
    
    # The color we provide gives RGB values in order, between 0 and 1
    im_filt_1 = colorize(test[..., 1], color_filt_1, sat)
    im_filt_2 = colorize(test[..., 0], color_filt_2, sat)
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
        
        filename = f'T{t:02}_abondance.svg'
        print(filename)
        plt.savefig(save_path / filename, bbox_inches='tight', dpi=600)
        plt.close(fig)
        
#%% plot all quantitative maps independently
save_tag = True

sat = 0.15              # percentage of saturated pixel
color_unmix_1 = (1, 0, 0)
color_unmix_2 = (0, 1, 0)
color_unmix_3 = (.5, .5, 0) # (1, .5, 0.15)
color_filt_1 = color_unmix_1
color_filt_2 = color_unmix_2

# loop over all z-slices
for z, t in enumerate(z_list):
    
    # Unmixing
    test = abond_unmix_4D[:,:,z,0:3]
    im_unmix_all = np.zeros(test.shape + (3,)) # add an RGB dimension
    
    # The color we provide gives RGB values in order, between 0 and 1
    im_unmix_all[...,1,:] = colorize(test[..., 0], color_unmix_1, sat)
    im_unmix_all[...,0,:] = colorize(test[..., 1], color_unmix_2, sat)
    im_unmix_all[...,2,:] = colorize(test[..., 2], color_unmix_3, sat)
    
    if save_tag:
        
        save_path = load_path / ('Visualisation' + suffix) / folder_unmix / 'qmap'
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        for m in range(3):
            plt.imsave(save_path / f'qmap_{m}_T{t:02}.png', im_unmix_all[...,m,:])

        plt.imsave(save_path / f'qmap_01_T{t:02}.png', np.clip(im_unmix_all[...,0,:] + im_unmix_all[...,1,:], 0, 1))
        plt.imsave(save_path / f'qmap_12_T{t:02}.png', np.clip(im_unmix_all[...,1,:] + im_unmix_all[...,2,:], 0, 1))
        plt.imsave(save_path / f'qmap_02_T{t:02}.png', np.clip(im_unmix_all[...,0,:] + im_unmix_all[...,2,:], 0, 1))
        plt.imsave(save_path / f'qmap_012_T{t:02}.png', np.clip(im_unmix_all[...,0,:] + im_unmix_all[...,1,:] + im_unmix_all[...,2,:], 0, 1))

#%% plot all filter maps fmaps independently

save_tag = True

color_filt_1 = color_unmix_1
color_filt_2 = color_unmix_2

# loop over all z-slices
for z, t in enumerate(z_list):
    
    # Unmixing
    test = abond_filter_4D[:,:,z,:]
    test = np.clip(test, 0, None) # remove negative values
    im_filter_all = np.zeros(test.shape + (3,))
    
    # The color we provide gives RGB values in order, between 0 and 1
    im_filter_all[...,0,:] = colorize(test[..., 0], color_filt_2, sat)
    im_filter_all[...,1,:] = colorize(test[..., 1], color_filt_1, sat)
    
    if save_tag:
        save_path = load_path / ('Visualisation' + suffix) / folder_unmix / 'fmap'
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        for m in range(2):
            plt.imsave(save_path / f'fmap_{m}_T{t:02}.png', im_filter_all[...,m,:])
        plt.imsave(save_path / f'fmap_01_T{t:02}.png', np.clip(im_filter_all[...,0,:] + im_filter_all[...,1,:], 0, 1))

#%% This was an attempt with microplot (not really better that using matplotlb directly)
# from microfilm import microplot
# z = 2
# test = abond_unmix_4D[:,:,z,0:3]
# test = np.moveaxis(test, -1, 0)
# microim = microplot.microshow(images=test, cmaps=['pure_cyan', 'pure_green', 'pure_green'])
# microim = microplot.microshow(images=[test[0], test[1]], cmaps=['pure_cyan', 'YlGn'])

#%% Upload to pilot-warehouse
from pathlib import Path 
import girder_client as gc

# api Rest url of the warehouse
url='https://pilot-warehouse.creatis.insa-lyon.fr/api/v1'


# Generate the warehouse client
gc = gc.GirderClient(apiUrl=url)

# Authentification
txt_file = open(Path('C:/Users/ducros/.apikey/pilot-warehouse.txt'), 'r', encoding='utf8')
apiKey = txt_file.read()
gc.authenticate(apiKey=apiKey)  # Authentication to the warehouse

folder = load_path  / ('Visualisation' + suffix)

folderId = '6708d7990e9f151150f3c100' # /2023_03_13_2023_03_14_eGFP_DsRed_3D/
gc.upload(str(folder), folderId, reuseExisting=True)

#%%

