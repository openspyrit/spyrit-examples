# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 09:04:19 2025

@author: ducros
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 20:22:44 2025

@author: ducros
"""

#%%
import torch
import numpy as np

from pathlib import Path


mask_type = None         # 'skew' or None to load a PNG
mask_filename = 'cat_roi.png'  # only if mask_type is not 'skew'
image_filename = 'dog.png'

pattern_folder = Path('figures/presentation/')


#%%

h=16

from PIL import Image

mask = Image.open(pattern_folder/mask_filename)
mask_resized = mask.resize((h,h), Image.NEAREST)
mask_resized = np.array(mask_resized)[:,:,0]


#%%
from matplotlib.pyplot import imsave
def just_save(A, folder, name, imsize=None):
    
    folder_full = Path(folder) / name
    Path(folder_full).mkdir(parents=True, exist_ok=True)
    
    
    if imsize is None:
        for i in range(A.shape[0]):
            # save
            filename = f'{name}_{i}.png'
            imsave(folder_full/filename, A[i], cmap='gray', vmin=0, vmax=255)
            
    else:
        for i in range(A.shape[0]):
            # save
            filename = f'{name}_{i}.png'
            imsave(folder_full/filename, 
                   np.array(Image.fromarray(A[i]).resize(imsize, Image.NEAREST)),
                   cmap='gray', 
                   vmin=0, vmax=255)
        
    return folder_full

#%% Define mask
from matplotlib.pyplot import imread
from spyrit.misc.disp import imagesc

mask = imread(pattern_folder/mask_filename)
mask = mask[:,:,0]
mask = mask.astype(np.bool)
ind_array = np.where(mask == True)
ind_array = (torch.from_numpy(ind_array[0]),torch.from_numpy(ind_array[1]))

mask_name = mask_filename[:-8]
    
N_pixel = len(ind_array[0])

# shape and plot
print(mask.shape)
imagesc(mask)


#
mask_resized = mask_resized.astype(np.bool)
ind_array = np.where(mask_resized == True)
ind_array = (torch.from_numpy(ind_array[0]),torch.from_numpy(ind_array[1]))
N_pixel = len(ind_array[0])

# remove last pixel, if necessary
ind_pix = -1
if N_pixel==65:
    mask_resized[ind_array[0][ind_pix], ind_array[1][ind_pix]] = False
    ind_array = np.where(mask_resized == True)
    ind_array = (torch.from_numpy(ind_array[0]),torch.from_numpy(ind_array[1]))
    N_pixel = len(ind_array[0])

ind_array = np.where(mask_resized == True)
ind_array = (torch.from_numpy(ind_array[0]),torch.from_numpy(ind_array[1]))
N_pixel = len(ind_array[0])


mask_name = mask_filename[:-8]
    
# shape and plot
print(mask.shape)
imagesc(mask_resized)

#%% Hadamard matrix 1D
from spyrit.core.torch import walsh_matrix
from spyrit.core.meas import FreeformLinearSplit

print('== Hadamard 1D ==')

# Hadamard
H = walsh_matrix(N_pixel)

#--- Simulation
meas_1d = FreeformLinearSplit(H,
                    meas_shape = (h,h), 
                    index_mask = torch.stack(ind_array),
                    )

# DMD pattern
A_hadam1d = meas_1d.unvectorize(meas_1d.A)

# 8-bit numpy array with max set to 255
A_hadam1d = A_hadam1d.numpy().astype(np.uint8)
A_hadam1d = A_hadam1d*255
print(A_hadam1d.shape)

# plot
m = 0
imagesc(A_hadam1d[m], f'{m}-th of H1')
    
# save as numpy array
filename = f'hadam1d_{mask_name}_{A_hadam1d.shape[0]}_{h}x{h}'
full_folder = just_save(A_hadam1d, pattern_folder, filename)

# save larger images (looks nicer when imported in powerpoint)
filename = f'hadam1d_{mask_name}_{A_hadam1d.shape[0]}_{4*h}x{4*h}'
full_folder = just_save(A_hadam1d, pattern_folder, filename, imsize=(4*h,4*h))

#%% 2D Hadamard
#Simulate noisy measurements
from spyrit.core.meas import HadamSplit2d

print('== Hadamard 2D masked ==')

#--- Simulation
meas_2d = HadamSplit2d(h)
A_hadam2d = meas_2d.unvectorize(meas_2d.A)
A_hadam2d = A_hadam2d.numpy().astype(np.uint8)
A_hadam2d = A_hadam2d*255

print(A_hadam2d.shape)

# plot
m = 0
imagesc(A_hadam2d[m], f'{m}-th of H2')

# save as numpy array
filename = f'hadam2d_full_{A_hadam2d.shape[0]}_{h}x{h}'
full_folder = just_save(A_hadam2d, pattern_folder, filename)

# save larger images (looks nicer when imported in powerpoint)
filename = f'hadam2d_full_{A_hadam2d.shape[0]}_{4*h}x{4*h}'
full_folder = just_save(A_hadam2d, pattern_folder, filename, imsize=(4*h,4*h))

#%% Raster scan
from spyrit.core.meas import FreeformLinear


H = torch.eye(N_pixel)
meas_rs = FreeformLinear(H,
                meas_shape = (h,h), 
                index_mask = torch.stack(ind_array),
                #dtype = x.dtype, # Check why default dtype not working here ???
                )

# DMD pattern
A_rs = meas_rs.unvectorize(meas_rs.H)

# 8-bit numpy array with max set to 255
A_rs = A_rs.numpy().astype(np.uint8)
A_rs = A_rs*255
print(A_rs.shape)

# plot
m = 0
imagesc(A_rs[m], f'{m}-th of H1')
    
# save as numpy array
filename = f'raster_{mask_name}_{A_rs.shape[0]}_{h}x{h}'
full_folder = just_save(A_rs, pattern_folder, filename)

# save larger images (looks nicer when imported in powerpoint)
filename = f'raster_{mask_name}_{A_rs.shape[0]}_{4*h}x{4*h}'
full_folder = just_save(A_rs, pattern_folder, filename, imsize=(4*h,4*h))

#%% 2D Hadamard masked
#Simulate noisy measurements
from spyrit.core.meas import HadamSplit2d

print('== Hadamard 2D masked ==')

#--- Simulation
meas_2d = HadamSplit2d(h)
A_hadam2d = meas_2d.unvectorize(meas_2d.A)
A_hadam2d = A_hadam2d*mask_resized
A_hadam2d = A_hadam2d.numpy().astype(np.uint8)
A_hadam2d = A_hadam2d*255

print(A_hadam2d.shape)

# plot
m = 0
imagesc(A_hadam2d[m], f'{m}-th of H2')

# save as numpy array
filename = f'hadam2d_{mask_name}_{A_hadam2d.shape[0]}_{h}x{h}'
full_folder = just_save(A_hadam2d, pattern_folder, filename)

# save larger images (looks nicer when imported in powerpoint)
filename = f'hadam2d_{mask_name}_{A_hadam2d.shape[0]}_{4*h}x{4*h}'
full_folder = just_save(A_hadam2d, pattern_folder, filename, imsize=(4*h,4*h))