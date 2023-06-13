# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 18:17:29 2023

@author: ducros
"""
#%% Load and save experimental patterns
import numpy as np
from pathlib import Path

M = 128
N = 512
mat_folder = r'.\data\2023_03_13_2023_03_14_eGFP_DsRed_3D\Reconstruction\Mat_rc'
save_folder = './pattern/'

# load
H_exp = np.load(Path(mat_folder) / f'motifs_Hadamard_{M}_{N}.npy')
H_exp = np.flip(H_exp,1).copy() # copy() required to remove negatives strides
H_exp /= H_exp[0,16:500].mean()

H_exp2 = H_exp/H_exp[0,:]

# save
#np.save(Path(save_folder) / 'patterns_2023_03_13.npy', H_exp)

#%% Load experimental measurement matrix (after split compensation)
from pathlib import Path
from spyrit.misc.disp import add_colorbar
import matplotlib.pyplot as plt
from spyrit.misc.walsh_hadamard import walsh_matrix

M,N = H_exp.shape
H_tar = walsh_matrix(N)
H_tar = H_tar[:M]

#%% plot 2D
f, axs = plt.subplots(2, 1)
axs[0].set_title('Target measurement patterns')
im = axs[0].imshow(H_tar, cmap='gray') 
add_colorbar(im, 'bottom')
axs[0].get_xaxis().set_visible(False)

axs[1].set_title('Experimental measurement patterns')
im = axs[1].imshow(H_exp, cmap='gray') 
add_colorbar(im, 'bottom')
axs[1].get_xaxis().set_visible(False)

#%% plot 1d
fig_folder = './figure/'

ind = [[0,1,2],[6,12,46]]

f, axs = plt.subplots(2, 3, figsize=(15,7))
for ax, ix in zip(axs, ind):
    for a,i in zip(ax, ix):
        print(i)
        print(a)
        a.plot(H_tar[i,:])
        a.plot(H_exp[i,:])
        a.plot(H_exp2[i,:])
        a.set_title(f'Patterns #{i}')
        
plt.savefig(Path(fig_folder) / 'patterns_profiles', bbox_inches='tight', dpi=600)