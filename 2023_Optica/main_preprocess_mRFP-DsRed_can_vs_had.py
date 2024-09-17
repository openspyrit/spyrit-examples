# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:38:33 2023

@author: ducros
"""
#%%
import numpy as np
#from PIL import Image
import sys
sys.path.append('./fonction')
#from load_data import Select_data
#from matrix_tools import bining_colonne, bining_line
from load_data import load_pattern_pos_neg, load_data_pos_neg
from pathlib import Path


#%% Hadamard patterns  /!\ DATA NOT IN THE WAREHOUSE YET (hard drive only)
import matplotlib.pyplot as plt
from spyrit.misc.disp import add_colorbar

save_tag = True
data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/'
Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/'
Run = 'RUN0001' 
save_folder = '/Reconstruction/Mat_rc/'

# Binning is chosen such that:
# 56 - 2104 = 2048 rows, hence 512 rows after x4 binning
# 20 = 128 spectral channels 
H_pos, H_neg = load_pattern_pos_neg(Dir,Run,4)

norm = H_pos[0,16:500].mean()
H_pos = np.flip(H_pos,1).copy() # copy() required to remove negatives strides
H_neg = np.flip(H_neg,1).copy() # copy() required to remove negatives strides
H_pos /= norm
H_neg /= norm

print(f'Hadamard pattern normalization factor: {norm}')

f, axs = plt.subplots(3, 1)
axs[0].set_title('Positive measurement patterns')
im = axs[0].imshow(H_pos, cmap='gray') 
add_colorbar(im)
axs[0].get_xaxis().set_visible(False)

axs[1].set_title('Negative measurement patterns')
im = axs[1].imshow(H_neg, cmap='gray') 
add_colorbar(im)
axs[1].get_xaxis().set_visible(False)

axs[2].set_title('Sum')
im = axs[2].imshow(H_pos + H_neg, cmap='gray') 
add_colorbar(im)
axs[2].get_xaxis().set_visible(False)

# Save
Nl, Nh, = H_pos.shape
if save_tag:

    Path(data_folder+save_folder).mkdir(parents=True, exist_ok=True)
    # data
    filename = f'hadamard_matrix_{Nl}_{Nh}_pos.npy'
    np.save(Path(data_folder+save_folder) / filename, H_pos)
    filename = f'hadamard_matrix_{Nl}_{Nh}_neg.npy'
    np.save(Path(data_folder+save_folder) / filename, H_neg)
    filename = f'hadamard_matrix_{Nl}_{Nh}.npy'
    np.save(Path(data_folder+save_folder) / filename, H_pos - H_neg)
    # figure
    filename = f'hadamard_matrix_{Nl}_{Nh}.pdf'
    plt.savefig(Path(data_folder+save_folder)/filename, bbox_inches='tight', dpi=600)
    

# Check the patterns
# filename = f'motifs_Hadamard_{Nl}_{Nh}.npy'
# H_2 = np.load(Path(data_folder+save_folder) / filename)
# H_2 /= H_2[0,16:500].mean()
# H_2 = np.flip(H_2,1).copy() # copy() required to remove negatives strides

# H = H_pos - H_neg
# H = np.flip(H,1)
# H /= H[0,16:500].mean()
# H = np.flip(H,1)
# print(f'error: {np.linalg.norm(H-H_2)/np.linalg.norm(H_2)}')
    
#%% mRFp + DsRed sample /!\ DATA NOT IN THE WAREHOUSE YET (hard drive only)

Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/'
Run = 'RUN0004' 
save_folder = '/Preprocess/'
n_channel = 2


# Binning is chosen such that:
# 56 - 2104 = 2048 rows, hence 512 rows after x4 binning
# 20 = 128 spectral channels 
stack_pos, stack_neg = load_data_pos_neg(Dir, Run, 56, 2104, 4, n_channel)

# Save
if save_tag:

    Path(data_folder+save_folder).mkdir(parents=True, exist_ok=True)
    
    Nl, Nh, Nc = stack_pos.shape
    filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_pos.npy'
    np.save(Path(data_folder+save_folder) / filename, stack_pos)
    
    filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_neg.npy'
    np.save(Path(data_folder+save_folder) / filename, stack_neg)
    
    print('-- Preprocessed measurements saved')
    
#%% Check the prep data (pos neg should math with seb's prep)
# data_folder = './data/2023_02_28_mRFP_DsRed_3D/'
# Run = 'RUN0006' 
# Nl, Nh, Nc = 512, 128, 128

# filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}.npy'
# prep = np.load(Path(data_folder+save_folder) / filename)

# filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_pos.npy'
# prep_pos = np.load(Path(data_folder+save_folder) / filename)

# filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_neg.npy'
# prep_neg =  np.load(Path(data_folder+save_folder) / filename)

# print(f'error: {np.linalg.norm(prep_pos-prep_neg-prep)/np.linalg.norm(prep)}')


#%% Canonical (pushbroom) patterns  /!\ DATA NOT IN THE WAREHOUSE YET
import matplotlib.pyplot as plt
from spyrit.misc.disp import add_colorbar
from load_data import load_pattern

save_tag = True
data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/'
Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/'
save_folder = '/Reconstruction/Mat_rc/'
Run_can = 'RUN0002'
Run_dark = 'RUN0001'

# Binning is chosen such that:
# 56 - 2104 = 2048 rows, hence 512 rows after x4 binning
# 20 = 128 spectral channels 
H_can  = load_pattern(Dir, Run_can,  4) # Seb used 750,1000, not the default

# Dark as first negative of Hadamard patterns
_, H_neg = load_pattern_pos_neg(Dir,Run_dark,4)
H_dark = H_neg[0,:].reshape((1,-1))

H_can  = np.flip(H_can,1).copy() # copy() required to remove negatives strides
H_dark = np.flip(H_dark,1).copy() # copy() required to remove negatives strides

H = H_can - H_dark

norm = H[0,16:500].max()
H /= norm
H_can /= norm
H_dark /= norm

print(f'Canonical pattern normalization factor: {norm}')

f, axs = plt.subplots(3, 1)
axs[0].set_title('Canonical patterns')
im = axs[0].imshow(H_can, cmap='gray') 
add_colorbar(im)
axs[0].get_xaxis().set_visible(False)

axs[1].set_title('Dark patterns')
axs[1].plot(H_dark.squeeze())
#axs[1].get_xaxis().set_visible(False)

axs[2].set_title('Difference')
im = axs[2].imshow(H, cmap='gray') 
add_colorbar(im)
axs[2].get_xaxis().set_visible(False)

# Save
Nl, Nh, = H.shape
if save_tag:

    Path(data_folder+save_folder).mkdir(parents=True, exist_ok=True)
    # data
    filename = f'canonical_matrix_{Nl}_{Nh}_pos.npy'
    np.save(Path(data_folder+save_folder) / filename, H_can)
    filename = f'dark_matrix_{Nl}_{Nh}_neg.npy'
    np.save(Path(data_folder+save_folder) / filename, H_dark)
    filename = f'canonical_diff_matrix_{Nl}_{Nh}.npy'
    np.save(Path(data_folder+save_folder) / filename, H)
    # figure
    filename = f'canonical_matrix_{Nl}_{Nh}.pdf'
    plt.savefig(Path(data_folder+save_folder)/filename, bbox_inches='tight', dpi=600)
    
# svd
U, S_can, Vh = np.linalg.svd(H_can) #full_matrices=True
U, S, Vh = np.linalg.svd(H) #full_matrices=True

if save_tag:
    fig = plt.figure()
    plt.plot(S_can)
    plt.plot(S)
    plt.yscale('log')
    plt.legend(['Raw', 'Diff'])
    plt.title('Sigular values of measurement matrix')
    #
    filename = f'canonical_svd_{Nl}_{Nh}.pdf'
    plt.savefig(Path(data_folder+save_folder)/filename, bbox_inches='tight', dpi=600)

#%% mRFp + DsRed sample /!\ DATA NOT IN THE WAREHOUSE YET
from load_data import load_data

Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/'
Run_can = 'RUN0006' 
Run_dark = 'RUN0003' 
save_folder = '/Preprocess/'
save_tag = True
n_channel = 2

# Binning is chosen such that:
# 56 - 2104 = 2048 rows, hence 512 rows after x4 binning
# 20 = 128 spectral channels 
stack_can  = load_data(Dir, Run_can, 56, 2104, 4, n_channel)
stack_dark = load_data(Dir, Run_dark, 56, 2104, 4, n_channel)

# Save
if save_tag:

    Path(data_folder+save_folder).mkdir(parents=True, exist_ok=True)
    
    Nl, Nh, Nc = stack_can.shape

    filename = f'{Run_can}_Can_{Nl}_{Nh}_{Nc}_can.npy'
    np.save(Path(data_folder+save_folder) / filename, stack_can)
    filename = f'{Run_dark}_Can_{Nl}_{Nh}_{Nc}_dark.npy'
    np.save(Path(data_folder+save_folder) / filename, stack_dark)     
    
    print('-- Preprocessed measurements saved')