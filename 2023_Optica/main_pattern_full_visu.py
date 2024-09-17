# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 08:56:58 2023

@author: ducros
"""
#%% Illuminations patterns (Nx, Ny, K)
import numpy as np
import sys
sys.path.append('./fonction')
from load_data import load_pattern_full_pos_neg
from pathlib import Path

import matplotlib.pyplot as plt
from spyrit.misc.disp import add_colorbar

save_fig = True
ext_fig = 'pdf'
dpi_fig = 600

save_folder = Path('./pattern/')

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
        
#%% Profiles / acquisition matrix (Nx, K)
import matplotlib.pyplot as plt
from spyrit.misc.disp import add_colorbar
from load_data import load_pattern_pos_neg

save_fig = True
ext_fig = 'pdf'
dpi_fig = 1200

data_folder = './data/2023_02_28_mRFP_DsRed_3D/'
Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/data_2023_02_28/'
Run = 'RUN0002' 

# Binning is chosen such that:
# 56 - 2104 = 2048 rows, hence 512 rows after x4 binning
# 20 = 128 spectral channels 
H_pos, H_neg = load_pattern_pos_neg(Dir,Run,4)

H_pos = np.flip(H_pos,1).copy() # copy() required to remove negatives strides
H_neg = np.flip(H_neg,1).copy() # copy() required to remove negatives strides
norm = H_pos[0,16:500].mean()
H_pos /= norm
H_neg /= norm
H_diff = H_pos - H_neg
H_diff /= H_diff[0,16:500].mean()
H_sum = H_pos + H_neg
H_sum /= H_sum[0,16:500].mean()

# plot
f, axs = plt.subplots(4, 1)

axs[0].set_ylabel('Pos')
im = axs[0].imshow(H_pos, cmap='gray') 
axs[0].tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)
add_colorbar(im)

axs[1].set_ylabel('Neg')
im = axs[1].imshow(H_neg, cmap='gray') 
axs[1].tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)
add_colorbar(im)

axs[2].set_ylabel('Diff')
im = axs[2].imshow(H_diff, cmap='gray') 
axs[2].tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)
add_colorbar(im)

axs[3].set_ylabel('Sum')
im = axs[3].imshow(H_sum, cmap='gray') 
axs[3].tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)
add_colorbar(im)

if save_fig:
    save_filename = f'measurement_matrix_actual.{ext_fig}'
    plt.savefig(save_folder/save_filename, bbox_inches='tight', dpi=dpi_fig)


# Save
# Nl, Nh, Nc = stack_pos.shape
# Ns = int(Run[-1])-1
# filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_pos.npy'
# np.save(Path(data_folder+save_folder) / filename, stack_pos)
# filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_neg.npy'
# np.save(Path(data_folder+save_folder) / filename, stack_neg)

#%% Profiles / acquisition matrix (Nx, K)
from spyrit.misc.walsh_hadamard import walsh_matrix
import matplotlib.pyplot as plt
from spyrit.misc.disp import add_colorbar
from load_data import load_pattern_pos_neg

save_fig = True
ext_fig = 'pdf'
dpi_fig = 1200


H = walsh_matrix(512)
H = H[:128]
H_pos = np.where(H > 0, H, 0)
H_neg = np.where(H < 0, -H, 0)
H_sum = H_pos + H_neg

# plot
f, axs = plt.subplots(4, 1)

axs[0].set_ylabel('Pos')
im = axs[0].imshow(H_pos, cmap='gray') 
axs[0].tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)
add_colorbar(im)

axs[1].set_ylabel('Neg')
im = axs[1].imshow(H_neg, cmap='gray') 
axs[1].tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)
add_colorbar(im)

axs[2].set_ylabel('Diff')
im = axs[2].imshow(H, cmap='gray') 
axs[2].tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)
add_colorbar(im)

axs[3].set_ylabel('Sum')
im = axs[3].imshow(H_sum, cmap='gray') 
axs[3].tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)
add_colorbar(im)

if save_fig:
    save_filename = f'measurement_matrix_target.{ext_fig}'
    plt.savefig(save_folder/save_filename, bbox_inches='tight', dpi=dpi_fig)