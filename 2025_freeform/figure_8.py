# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 16:16:15 2025

@author: ducros
"""

#%%
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt

h = 128  # image size hxh

mask_type = None        # 'skew' or None to load a PNG
mask_filename = 'cat_roi.png'  # only if mask_type is not 'skew'

norm = 1e4 # normalization factor


fig_folder = Path('figures')
plot_tag = True 

# Plot options  
fs = 18                 # Font size
lw = 2 # line width
dpi_fig = 300

save_tag = True
ext = 'png'
plt.rcParams['text.usetex'] = True

#%%
# Load experimental data
# ====================================================================
import json
import ast

data_folder =  Path(r".\data\2025-09-25_freeform_publication") 


if mask_type == 'skew':
    data_title = [
    r'obj_StarSector_source_white_LED_Walsh_im_128x128_ti_1ms_zoom_x1',    
    r'obj_StarSector_source_white_LED_hadam1d_skew_16384_im_128x128_ti_2ms_zoom_x1',
    r'obj_StarSector_source_white_LED_hadam2d_skew_32768_im_128x128_ti_1ms_zoom_x1',
    r'obj_StarSector_source_white_LED_smatrix_skew_8191_im_128x128_ti_4ms_zoom_x1', 
    r'obj_StarSector_source_white_LED_raster_skew_8192_im_128x128_ti_4ms_zoom_x1',
    #r'obj_StarSector_source_white_LED_Raster_im_128x128_ti_2ms_zoom_x1',
    ]
else:
    data_title = [
    r'obj_StarSector_source_white_LED_Walsh_im_128x128_ti_1ms_zoom_x1',
    r'obj_StarSector_source_white_LED_hadam1d_cat_8192_im_128x128_ti_4ms_zoom_x1',
    r'obj_StarSector_source_white_LED_hadam2d_cat_32768_im_128x128_ti_1ms_zoom_x1',
    r'obj_StarSector_source_white_LED_smatrix_cat_4095_im_128x128_ti_8ms_zoom_x1',
    r'obj_StarSector_source_white_LED_raster_cat_4096_im_128x128_ti_8ms_zoom_x1',
    #r'obj_StarSector_source_white_LED_Raster_im_128x128_ti_2ms_zoom_x1',
    ]

def load_spihim(data_folder, data_title):
      
    suffix = {"data": "_spectraldata.npz", "metadata": "_metadata.json"}
    
    # Spectral data in numpy
    exp_data = [
        np.load(data_folder / title / (title + suffix["data"]))["spectral_data"]
        for title in data_title
        ]

    # Metadata
    patterns = [[] for _ in range(len(data_title))]
    wavelengths = [[] for _ in range(len(data_title))]
    
    for ii, title in enumerate(data_title):
        
        file = open(data_folder / title / (title + suffix["metadata"]), "r")
        json_metadata = json.load(file)[4]
        file.close()
        
        # Pattern order
        # replace "np.int32(" with an empty string and ")" with an empty string
        tmp = json_metadata["patterns"]
        tmp = tmp.replace("np.int32(", "").replace(")", "")
        patterns[ii] = ast.literal_eval(tmp)
        
        # Wavelength
        wavelengths[ii] = ast.literal_eval(json_metadata["wavelengths"])
        
    return exp_data, wavelengths, patterns

data_exp, wavelength, patterns = load_spihim(data_folder, data_title)

# take only the first repetition
for j in range(len(data_exp)):
    data_exp[j] = data_exp[j][:data_exp[j].shape[0]//2, :]
    
#%% Plot total spectra
i_list = [0, 4, 2, 1, 3]
method_list = ['FH2', 'RS', 'MH2', 'H1', 'S1']

# lambda_central_list = [512, 1900, 1900, 1900] # no signal below 15 and above 2038
# nc_list = [1, 1, 15, 60]

lambda_central_list = [515, 515, 1800, 1800] # no signal below 15 and above 2038
nc_list = [16, 3, 16, 3]

if plot_tag:
    
    plt.figure()
    for h, i in enumerate(i_list):
        if i == 4:
            mult = 50
            plt.plot(mult*data_exp[i].sum(axis=0), 
                     label=method_list[h]+f' (x{mult})')
        else:
            plt.plot(data_exp[i].sum(axis=0), 
                     label=method_list[h])
    #plt.title('Total spectrum', fontsize=fs)
    plt.ylabel('Total spectrum (in counts)', fontsize=fs)
    plt.xlabel('spectral channels', fontsize=fs)   
    
    plt.ylim(bottom=0)
    
    plt.axvline(x=lambda_central_list[0], linestyle='--', color='k')
    plt.fill_betweenx(np.arange(0,2048), 
                      lambda_central_list[1]-nc_list[1]+1, 
                      lambda_central_list[1]+nc_list[1], 
                      alpha=.15, 
                      color = 'k',
                      transform=plt.gca().get_xaxis_transform())
    plt.axvline(x=lambda_central_list[2], linestyle='--', color='k')
    plt.fill_betweenx(np.arange(0,2048), 
                      lambda_central_list[3]-nc_list[3]+1, 
                      lambda_central_list[3]+nc_list[3], 
                      alpha=.15, 
                      color = 'k',
                      transform=plt.gca().get_xaxis_transform())
    plt.legend(loc='upper center', fontsize=fs-2)
    plt.gca().tick_params(axis='both', which='major', labelsize=fs-2)
    plt.tight_layout()

if False:
    plt.savefig(fig_folder / ('figure_8.'+ext), bbox_inches='tight', dpi=dpi_fig)
    
#%% Plot maximum spectrum
i_list = [0, 4, 2, 1, 3]
method_list = ['FH2', 'RS', 'MH2', 'H1', 'S1']


if plot_tag:   
    plt.figure()
    for h, i in enumerate(i_list):
        
        kmax = np.argmax(data_exp[i][:,300:700].sum(axis=1))
        
        print(kmax)
        
        if i == 4:
            mult = 10
            plt.plot(mult*data_exp[i][kmax], 
                     label=method_list[h]+f' (x{mult}), '+f'$k={kmax}$',
                     linewidth=lw)
        else:
            plt.plot(data_exp[i][kmax], 
                     label=method_list[h]+f', $k={kmax}$',
                     linewidth=lw)
            
    #plt.title('Largest spectrum', fontsize=fs)
    plt.ylabel(r'Largest Spectrum (in counts)', fontsize=fs)
    plt.xlabel(r'Spectral Channels', fontsize=fs)
    plt.ylim(bottom=0)
    
    # Only for pdf figure (paper), not for png (presentation)
    if ext=='pdf':
        plt.axvline(x=lambda_central_list[0], linestyle='--', color='k', linewidth=lw)
        plt.fill_betweenx(np.arange(0,2048), 
                          lambda_central_list[1]-nc_list[1]+1, 
                          lambda_central_list[1]+nc_list[1], 
                          alpha=.15, 
                          color = 'k',
                          transform=plt.gca().get_xaxis_transform())
        plt.axvline(x=lambda_central_list[2], linestyle='--', color='k', linewidth=lw)
        plt.fill_betweenx(np.arange(0,2048), 
                          lambda_central_list[3]-nc_list[3]+1, 
                          lambda_central_list[3]+nc_list[3], 
                          alpha=.15, 
                          color = 'k',
                          transform=plt.gca().get_xaxis_transform())
    
    plt.legend(loc='upper right', fontsize=fs-2)
    plt.gca().tick_params(axis='both', which='major', labelsize=fs-2)
    plt.tight_layout()

    if save_tag:
        plt.savefig(fig_folder / ('figure_8a.' + ext), 
                    transparent=True,
                    bbox_inches='tight', 
                    dpi=dpi_fig)

#%% Plot minimum spectrum
i_list = [0, 4, 2, 1, 3]
method_list = ['FH2', 'RS', 'MH2', 'H1', 'S1']


if plot_tag:   
    plt.figure()
    for h, i in enumerate(i_list):
        
        kmax = np.argmin(data_exp[i][:,300:700].sum(axis=1))
        
        print(kmax)
        
        #if i == 4:
        if i in [0,4,2,1]:
            mult = 10
            plt.plot(mult*data_exp[i][kmax], 
                     label=method_list[h]+f' (x{mult}), '+f'$k={kmax}$',
                     linewidth=lw)
        else:
            plt.plot(data_exp[i][kmax], 
                     label=method_list[h]+f', $k={kmax}$',
                     linewidth=lw)
            
    #plt.title('Largest spectrum', fontsize=fs)
    plt.ylabel(r'Lowest Spectrum (in counts)', fontsize=fs)
    plt.xlabel(r'Spectral Channels', fontsize=fs)
    plt.ylim(bottom=0)
    
    # Only for pdf figure (paper), not for png (presentation)
    if ext=='pdf':
        plt.axvline(x=lambda_central_list[0], linestyle='--', color='k', linewidth=lw)
        plt.fill_betweenx(np.arange(0,2048), 
                          lambda_central_list[1]-nc_list[1]+1, 
                          lambda_central_list[1]+nc_list[1], 
                          alpha=.15, 
                          color = 'k',
                          transform=plt.gca().get_xaxis_transform())
        plt.axvline(x=lambda_central_list[2], linestyle='--', color='k', linewidth=lw)
        plt.fill_betweenx(np.arange(0,2048), 
                          lambda_central_list[3]-nc_list[3]+1, 
                          lambda_central_list[3]+nc_list[3], 
                          alpha=.15, 
                          color = 'k',
                          transform=plt.gca().get_xaxis_transform())
        
    plt.legend(loc='upper right', fontsize=fs-2)
    plt.gca().tick_params(axis='both', which='major', labelsize=fs-2)
    plt.tight_layout()

    if save_tag:
        plt.savefig(fig_folder / ('figure_8b.' + ext), 
                    transparent=True,
                    bbox_inches='tight', 
                    dpi=dpi_fig)