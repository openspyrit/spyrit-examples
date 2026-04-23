# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:22:16 2023

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

data_folder = './data/2023_02_28_mRFP_DsRed_3D/'
Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/data_2023_02_28/'
Run = 'RUN0002' 
save_folder = Dir

# Binning is chosen such that:
# 56 - 2104 = 2048 rows, hence 512 rows after x4 binning
# 20 = 128 spectral channels 
H_pos, H_neg = load_pattern_pos_neg(Dir,Run,4)

H_pos = np.flip(H_pos,1).copy() # copy() required to remove negatives strides
H_neg = np.flip(H_neg,1).copy() # copy() required to remove negatives strides
norm = H_pos[0,16:500].mean()
H_pos /= norm
H_neg /= norm


f, axs = plt.subplots(3, 1)
axs[0].set_title('Positive measurement patterns')
im = axs[0].imshow(H_pos, cmap='gray') 
add_colorbar(im, 'bottom')
axs[0].get_xaxis().set_visible(False)

axs[1].set_title('Negative measurement patterns')
im = axs[1].imshow(H_neg, cmap='gray') 
add_colorbar(im, 'bottom')
axs[1].get_xaxis().set_visible(False)

axs[2].set_title('Sum')
im = axs[2].imshow(H_pos + H_neg, cmap='gray') 
add_colorbar(im, 'bottom')
axs[2].get_xaxis().set_visible(False)

# Save
# Nl, Nh, Nc = stack_pos.shape
# Ns = int(Run[-1])-1
# filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_pos.npy'
# np.save(Path(data_folder+save_folder) / filename, stack_pos)
# filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_neg.npy'
# np.save(Path(data_folder+save_folder) / filename, stack_neg)

#%% eGFP + DsRed sample /!\ DATA NOT IN THE WAREHOUSE (hard drive only)
pilot_data_folder = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D/'
raw_data_folder = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D/Raw_data_chSPSIM_and_SPIM/'
save_folder = 'Preprocess'
save_path = Path(pilot_data_folder) / save_folder
save_path.mkdir(parents=True, exist_ok=True)

T_list = range(1,27)    # slice indices

for t in T_list:
    
    print(f'-- Slice: {t}')
    
    if t<6:
        date = '2023_03_13'
        Run = f'RUN{t+1:04}'
    else:
        date = '2023_03_14'
        Run = f'RUN{t-5:04}'
    
    
    # Binning is chosen such that:
    # 56 - 2104 = 2048 rows, hence 512 rows after x4 binning
    # 20 = 128 spectral channels 
    
    Dir = raw_data_folder + 'data_' + date + '/'
    stack_pos, stack_neg = load_data_pos_neg(Dir,Run,56,2104,4,20)
    
    # Save


    Nl, Nh, Nc = stack_pos.shape
    filename = f'T{t}_{Run}_{date}_Had_{Nl}_{Nh}_{Nc}_pos.npy'
    np.save(save_path / filename, stack_pos)
    
    filename = f'T{t}_{Run}_{date}_Had_{Nl}_{Nh}_{Nc}_neg.npy'
    np.save(save_path / filename, stack_neg)
    
    print('-- Preprocessed measurements saved')
    
#%% Check the prep data (pos neg should math with seb's prep)
#Run = 'RUN0004' 
#Ns = int(Run[-1])-1
#save_folder = '/Preprocess/'
#Nl, Nh, Nc = 512, 128, 128

#filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}.npy'
#prep = np.load(Path(data_folder+save_folder) / filename)

#filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_pos.npy'
#prep_pos = np.load(Path(data_folder+save_folder) / filename)

#filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_neg.npy'
#prep_neg =  np.load(Path(data_folder+save_folder) / filename)

#print(f'error: {np.linalg.norm(prep_pos-prep_neg-prep)/np.linalg.norm(prep)}')