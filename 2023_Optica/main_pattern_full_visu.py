# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 08:56:58 2023

@author: ducros
"""
#%%
import numpy as np
from PIL import Image
import sys
sys.path.append('./fonction')
from load_data import Select_data
from matrix_tools import bining_colonne, bining_line
from pathlib import Path


def load_pattern_full_pos_neg(Dir, Run, c_bin=1, l_bin=1):
    
    Path_files, list_files = Select_data(Dir,Run)
    Nh = len(list_files)//2
    Nl, Nc = np.rot90(np.array(Image.open(Path_files+list_files[0]))).shape
    
    print(f'Found {Nh} patterns of size {Nl}x{Nc}')
    
    pat_pos = np.zeros((Nh,Nl//l_bin,Nc//c_bin))
    pat_neg = np.zeros((Nh,Nl//l_bin,Nc//c_bin))
    
    for i in range(0,2*Nh,2):
        
        print(Path_files+list_files[i])
        print(Path_files+list_files[i+1])    
        
        tmp = np.float_(np.rot90(np.array(Image.open(Path_files+list_files[i])))) 
        tmp = bining_colonne(tmp, c_bin)
        tmp = bining_colonne(tmp.T, l_bin)
        pat_pos[i//2] = tmp.T
        
        tmp = np.float_(np.rot90(np.array(Image.open(Path_files+list_files[i+1]))))
        tmp = bining_colonne(tmp, c_bin)
        tmp = bining_colonne(tmp.T, l_bin)
        pat_neg[i//2] = tmp.T
    
    return pat_pos, pat_neg

#%% Acquisition patterns
import matplotlib.pyplot as plt
from spyrit.misc.disp import add_colorbar

save_fig = True
save_folder = Path(r'D:\Creatis\Communication\Journal\40_hspim\image_source\patterns_full')

data_folder = './data/2023_02_28_mRFP_DsRed_3D/'
Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/data_2023_02_28/'
Run = 'RUN0002' 

# Binning
H_pos, H_neg = load_pattern_full_pos_neg(Dir,Run,8,8)

H_pos = np.flip(H_pos,1).copy() # copy() required to remove negatives strides
H_neg = np.flip(H_neg,1).copy() # copy() required to remove negatives strides
norm = H_pos[0,16:500].mean()
H_pos = H_pos / norm
H_neg = H_neg / norm

H_pos = np.flip(H_pos, axis=1)
H_neg = np.flip(H_neg, axis=1)

H_pos = (H_pos - 0.8*H_pos.min())/(1.2*H_pos.max() - 0.8*H_pos.min())
H_neg = (H_neg - 0.8*H_neg.min())/(1.2*H_neg.max() - 0.8*H_neg.min())

# Normalize to adjust contrast
# H_pos_min = 0.8*H_pos.min(keepdims=True, axis=0)
# H_pos_max = 1.2*H_pos.max(keepdims=True, axis=0)
# H_pos = (H_pos - H_pos_min)/(H_pos_max - H_pos_min)

# H_neg_min = 0.8*H_neg.min(keepdims=True, axis=0)
# H_neg_max = 1.2*H_neg.max(keepdims=True, axis=0)
# H_neg = (H_neg - H_neg_min)/(H_neg_max - H_neg_min)

# plot
ind = 4
f, axs = plt.subplots(1, 3)
axs[0].set_title('Positive measurement patterns')
im = axs[0].imshow(H_pos[ind], cmap='gray') 
add_colorbar(im, 'bottom')
axs[0].get_xaxis().set_visible(False)

axs[1].set_title('Negative measurement patterns')
im = axs[1].imshow(H_neg[ind], cmap='gray') 
add_colorbar(im, 'bottom')
axs[1].get_xaxis().set_visible(False)

axs[2].set_title('Sum')
im = axs[2].imshow(H_pos[ind] + H_neg[ind], cmap='gray') 
add_colorbar(im, 'bottom')
axs[2].get_xaxis().set_visible(False)

# Save
if save_fig:
    save_folder.mkdir(parents=True, exist_ok=True)
    for ind in range(128):
        plt.imsave(save_folder / f'pat_pos_{ind}.png',H_pos[ind,:,:], cmap='Greys')
        plt.imsave(save_folder / f'pat_neg_{ind}.png',H_neg[ind,:,:], cmap='Greys')