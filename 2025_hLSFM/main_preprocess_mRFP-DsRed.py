# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 10:30:58 2023

@author: ducros
"""
#%%
import numpy as np
from PIL import Image
import sys
sys.path.append('./fonction')
from fonction.load_data import Select_data, load_pattern_pos_neg, load_data_pos_neg
from pathlib import Path


#%% Acquisition patterns
import matplotlib.pyplot as plt
from spyrit.misc.disp import add_colorbar

save_tag = True
data_folder = './data/2023_02_28_mRFP_DsRed_3D/'
Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/data_2023_02_28/'
Run = 'RUN0002' 
save_folder = 'Reconstruction/Mat_rc/'
save_path = Path(data_folder) / save_folder
save_path.mkdir(parents=True, exist_ok=True)


# Binning is chosen such that:
# 56 - 2104 = 2048 rows, hence 512 rows after x4 binning
# 20 = 128 spectral channels 
H_pos, H_neg = load_pattern_pos_neg(Dir,Run,4)

norm = H_pos[0,16:500].mean()
H_pos = np.flip(H_pos,1).copy() # copy() required to remove negatives strides
H_neg = np.flip(H_neg,1).copy() # copy() required to remove negatives strides
H_pos /= norm
H_neg /= norm

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
    # data
    filename = f'measurement_matrix_{Nl}_{Nh}_pos.npy'
    np.save(save_path/ filename, H_pos)
    filename = f'measurement_matrix_{Nl}_{Nh}_neg.npy'
    np.save(save_path / filename, H_neg)
    filename = f'measurement_matrix_{Nl}_{Nh}.npy'
    np.save(save_path / filename, H_pos - H_neg)
    # figure
    filename = f'measurement_matrix_{Nl}_{Nh}.png'
    plt.savefig(save_path/filename, bbox_inches='tight', dpi=600)
    

# Check the patterns
filename = f'motifs_Hadamard_{Nl}_{Nh}.npy'
H_2 = np.load(save_path / filename)
H_2 /= H_2[0,16:500].mean()
H_2 = np.flip(H_2,1).copy() # copy() required to remove negatives strides

H = H_pos - H_neg
H = np.flip(H,1)
H /= H[0,16:500].mean()
H = np.flip(H,1)
print(f'error: {np.linalg.norm(H-H_2)/np.linalg.norm(H_2)}')
    
#%% mRFp + DsRed sample /!\ DATA NOT IN THE WAREHOUSE YET (hard drive only)
pilot_data_folder = './data/2023_02_28_mRFP_DsRed_3D/'

# raw data downloaded from pilot and saved locally
Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/data_2023_02_28/'
save_folder = 'Preprocess'
save_path = Path(pilot_data_folder+ save_folder)
save_path.mkdir(parents=True, exist_ok=True)

# N.B.: Run0008 seems to be corrupted!
T_list = range(4,25)    # slice indices

for t in T_list:

    if t == 8:
        print(f"-- Skipping corrupted slice: {t}")
        continue

    print(f'-- Slice: {t}')
    Run = f'RUN{t:04}'
    
    # Binning is chosen such that:
    # 56 - 2104 = 2048 rows, hence 512 rows after x4 binning
    # 20 = 128 spectral channels 
    stack_pos, stack_neg = load_data_pos_neg(Dir,Run,56,2104,4,20)
    
    # Save
    Nl, Nh, Nc = stack_pos.shape
    filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_pos.npy'
    np.save(save_path / filename, stack_pos)
    
    filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_neg.npy'
    np.save(save_path / filename, stack_neg)
    
    print('-- Preprocessed measurements saved')
    
#%% Check the prep data (pos neg should math with seb's prep)
data_folder = './data/2023_02_28_mRFP_DsRed_3D/'
Run = 'RUN0006' 
Nl, Nh, Nc = 512, 128, 128

filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}.npy'
prep = np.load(Path(data_folder+save_folder) / filename)

filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_pos.npy'
prep_pos = np.load(Path(data_folder+save_folder) / filename)

filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_neg.npy'
prep_neg =  np.load(Path(data_folder+save_folder) / filename)

print(f'error: {np.linalg.norm(prep_pos-prep_neg-prep)/np.linalg.norm(prep)}')