# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 18:09:06 2023

@author: ducros
"""

#%% 1. Register raw measurements (i.e., before reconstruction)
# Load EGFP / DsRed raw data 
# from pathlib import Path
# import numpy as np
# import cv2

# Nl = 512 # number of pixcels along the y dimensions 
# Nh = 128 # number of measured patterns
# Nc = 128 # number of channels

# t_pix = 6.0 # maximum shift in pixel

# data_folder = Path('./data/2023_03_13_2023_03_14_eGFP_DsRed_3D/')
# prep_folder = Path('Preprocess')

# nbin = 20*4
# T_list = range(1,27)    # slice indices

# prep_raw = np.zeros((Nl,Nh,Nc))
# prep_reg = np.zeros((Nl,Nh,Nc))

#%% 1. Register raw measurements (i.e., before reconstruction)
#  /!\ NOT working as expected. Problem at the border with extrapolation.

# save_folder = data_folder / Path(prep_folder.name + '_registered_constant')
# save_folder.mkdir(parents=True)
    
# for t in T_list:
#     if t<6:
#         #Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/data_2023_03_13/'
#         date = '2023_03_13'
#         Run = f'RUN{t+1:04}'
#     else:
#         #Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/data_2023_03_14/'
#         date = '2023_03_14'
#         Run = f'RUN{t-5:04}'

#     # Load prep data (pos)
#     filename = f'T{t}_{Run}_{date}_Had_{Nl}_{Nh}_{Nc}_pos.npy'
#     print(filename)
#     prep_raw = np.load(data_folder / prep_folder / filename)
#     #prep_reg[:] = prep_raw.mean()
    
#     # Spectral registration (spatial shift that varies linearly with wavelength)
#     # cv2.BORDER_REPLICATE not working as expected ??
#     for l in range(Nc):
#         print(f'\r-- Registration channel: {l}', end="", flush=True)
#         # register (pos)
#         tx = l * t_pix / Nc
#         translation_matrix = np.float32([[1,0,0], [0,1,tx] ])
#         prep_reg[:,:,l] = cv2.warpAffine(prep_raw[:,:,l], translation_matrix,
#                                          (Nh,Nl), cv2.INTER_CUBIC, cv2.BORDER_REPLICATE)
        
    
#     # save (pos)
#     print()
#     print(f'Saving {save_folder / filename}')
#     np.save(save_folder / filename, prep_reg)
    
#     # Load prep data (neg)
#     filename = f'T{t}_{Run}_{date}_Had_{Nl}_{Nh}_{Nc}_neg.npy'
#     print(filename)
#     prep_raw = np.load(data_folder / prep_folder / filename)
    
#     # Spectral registration (spatial shift that varies linearly with wavelength)
#     for l in range(Nc):
#         print(f'\r-- Registration channel: {l}', end="", flush=True)
#         # register (neg)
#         tx = l * t_pix / Nc
#         translation_matrix = np.float32([[1,0,0], [0,1,tx] ])
#         prep_reg[:,:,l] = cv2.warpAffine(prep_raw[:,:,l], translation_matrix,
#                                          (Nh,Nl), cv2.INTER_CUBIC, cv2.BORDER_REPLICATE)
    
#     # save (neg)
#     print()
#     print(f'Saving {save_folder / filename}')
#     np.save(save_folder / filename, prep_reg)

#%% 2. Load EGFP / DsRed hypercube

from pathlib import Path
import numpy as np
import cv2

Nl = 512 # number of pixcels along the y dimensions 
Nh = 512 # number of measured Walsh_Hadmard coefficients (correpond to the h dimensions)
Nc = 128 # number of channels

t_pix = 6.0 # maximum shift in pixel

T_list = range(1,27)    # slice indices
load_path = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D'
recon = 'tikhonet50_div1.5'  # 'pinv' 'tikhonet50_div1.5'

load_folder = Path(load_path) / 'Reconstruction/hypercube' / recon 
save_folder = Path(load_path) / 'Reconstruction/hypercube' / (recon + '_shift')

# init
xyl_raw = np.zeros((Nl,Nh,Nc))
xyl_reg = np.zeros((Nl,Nh,Nc))

#%% 2. Register hypercube (i.e., after reconstruction)

save_folder.mkdir(parents=True)

for t in T_list:
    if t<6:
        date = '2023_03_13'
        Run = f'RUN{t+1:04}'
    else:
        date = '2023_03_14'
        Run = f'RUN{t-5:04}'
    
    # Load
    filename = f'T{t}_{Run}_{date}_rec_{recon}_exp_{Nl}x{Nh}x{Nc}.npy'
    print(filename)
    
    xyl_raw = np.load(load_folder / filename)
    
    # Spectral registration (spatial shift that varies linearly with wavelength)
    for l in range(Nc):
        
        # register
        tx = l * t_pix / Nc
        translation_matrix = np.float32([[1,0,0], [0,1,tx] ])
        xyl_reg[:,:,l] = cv2.warpAffine(xyl_raw[:,:,l], translation_matrix, (Nl,Nh))
        
    # save
    recon_shift = recon + '_shift'
    filename = f'T{t}_{Run}_{date}_rec_{recon_shift}_exp_{Nl}x{Nh}x{Nc}.npy'
    np.save(save_folder / filename, xyl_reg)


#%% 3. Load data DsRed / mRFP

from pathlib import Path
import numpy as np
import cv2

# Load data
Nl = 512 # number of pixcels along the y dimensions 
Nh = 512 # number of measured Walsh_Hadmard coefficients (correpond to the h dimensions)
Nc = 128 # number of channels

t_pix = 6.0 # maximum shift in pixel


T_list = [*range(4, 8), *range(9, 25)] # slice indices, Run0008 corrupted
load_path = './data/2023_02_28_mRFP_DsRed_3D'
recon = 'tikhonet50_div1.5'  # 'pinv' 'tikhonet50_div1.5'


load_folder = Path(load_path) / 'Reconstruction/hypercube' / recon 
save_folder = Path(load_path) / 'Reconstruction/hypercube' / (recon + '_shift')

# Init
xyl_raw = np.zeros((Nl,Nh,Nc))
xyl_reg = np.zeros((Nl,Nh,Nc))

#%% 3. Register hypercube (i.e., after reconstruction)
save_folder.mkdir(parents=True)

# loop over z-slices
for t in T_list:
    Run = f'RUN{t:04}'

    # Load
    filename = f'{Run}_rec_{recon}_exp_{Nl}x{Nh}x{Nc}.npy'
    print(filename)
    
    xyl_raw = np.load(load_folder / filename)
    
    # Spectral registration (spatial shift that varies linearly with wavelength)
    for l in range(Nc):
        
        # register
        tx = l * t_pix / Nc
        translation_matrix = np.float32([[1,0,0], [0,1,tx] ])
        xyl_reg[:,:,l] = cv2.warpAffine(xyl_raw[:,:,l], translation_matrix, (Nl,Nh))
        
    # save
    recon_shift = recon + '_shift'
    filename = f'{Run}_rec_{recon_shift}_exp_{Nl}x{Nh}x{Nc}.npy'
    np.save(save_folder / filename, xyl_reg)
