# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 18:13:57 2025

@author: ducros
"""

# -*- coding: utf-8 -*-

#%%
import torch
import numpy as np

from pathlib import Path

h = 128  # image size hxh

mask_type = None         # 'skew' or None to load a PNG
mask_filename = 'cat_roi.png'  # only if mask_type is not 'skew'

pattern_folder = Path('patterns')

#%%
def ravel_and_save(A, folder, name):
    
    folder_full = Path(folder) / name
    Path(folder_full).mkdir(parents=True, exist_ok=True)
    
    for i in range(A.shape[0]):
        # ravel
        vec = A[i].ravel()
        # save
        filename = f'{name}_{i}.npy'
        np.save(folder_full/filename, vec)
        
    return folder_full

#%% Define mask
from matplotlib.pyplot import imread
from spyrit.misc.disp import imagesc

# Build skew mask
if mask_type == 'skew':
    start = 45
    mask_width = 64 # a power of two
    f1,l1 = start,start + mask_width
    
    base = np.arange(f1,l1)
    ind = base
    for _ in range(h-1):
        base = base + h-1
        ind = np.append(ind, base)
        
    ind_array = np.unravel_index(ind, (h,h))
    ind_array = tuple(torch.from_numpy(array) for array in ind_array)
    
    mask = torch.zeros((h,h), dtype=torch.bool)
    mask[ind_array] = 1
    
    mask_name = mask_type
    
# Or load image
else:
    mask = imread(mask_filename)
    mask = mask[:,:,0]
    mask = mask.astype(np.bool)
    ind_array = np.where(mask == True)
    ind_array = (torch.from_numpy(ind_array[0]),torch.from_numpy(ind_array[1]))
    
    mask_name = mask_filename[:-8]
    
N_pixel = len(ind_array[0])

# shape and plot
print(mask.shape)
imagesc(mask)

#%% Raster Scan
from spyrit.core.meas import FreeformLinear
import numpy as np

print('== Raster Scan ==')

# Identity
H = torch.eye(N_pixel)

# meas operator
meas_1d = FreeformLinear(H,
                meas_shape = (h,h), 
                index_mask = torch.stack(ind_array),
                dtype = torch.bool, # Check why default dtype not working here ???
                )

# DMD pattern
A_raster = meas_1d.unvectorize(meas_1d.H)

# 8-bit numpy array with max set to 255
A_raster = A_raster.numpy().astype(np.uint8)
A_raster = A_raster*255
print(A_raster.shape)

# plot
m = 0
imagesc(A_raster[m], f'{m}-th of raster')
    
# save as numpy array
filename =  f'raster_{mask_name}_{A_raster.shape[0]}_{h}x{h}'
#np.save(pattern_folder / filename, A_raster)
full_folder = ravel_and_save(A_raster, pattern_folder, filename)

filename = full_folder / f'raster_{mask_name}_mask_{h}x{h}.npy'
np.save(filename, mask)

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
# np.save(pattern_folder / filename, A_hadam1d)
full_folder = ravel_and_save(A_hadam1d, pattern_folder, filename)

filename = full_folder / f'hadam1d_{mask_name}_mask_{h}x{h}.npy'
np.save(filename, mask)

#%% Arbitrary shape -- S matrix
from spyrit.misc.walsh_hadamard import walsh_S_matrix
from spyrit.core.meas import FreeformLinear

print('== S-matrix 1D ==')

# S matrix
K = N_pixel-1
H = torch.from_numpy(walsh_S_matrix(K))

# ROI indices
ind_array_0_S = ind_array[0][:-1]
ind_array_1_S = ind_array[1][:-1]
mask_smatrix = np.zeros_like(mask)
mask_smatrix[ind_array_0_S,ind_array_1_S] = True
imagesc(mask_smatrix, 'mask (S-matrix)')

#--- Simulation
meas_1d = FreeformLinear(H, 
                    meas_shape = (h,h), 
                    index_mask = torch.stack((ind_array_0_S, ind_array_1_S)),
                    dtype = torch.uint8, # Check why default dtype not working here ???
                    )

# DMD pattern
A_smatrix = meas_1d.unvectorize(meas_1d.H)
A_smatrix = A_smatrix.numpy()
A_smatrix = A_smatrix*255
print(A_smatrix.shape)

# plot
m = 0
imagesc(A_smatrix[m], f'{m}-th of S-matrix')
    
# save as numpy array
filename = f'smatrix_{mask_name}_{A_smatrix.shape[0]}_{h}x{h}'
# np.save(pattern_folder/filename, A_smatrix)
full_folder = ravel_and_save(A_smatrix, pattern_folder, filename)

filename = full_folder / f'smatrix_{mask_name}_mask_{h}x{h}.npy'
np.save(filename, mask_smatrix)

#%% 2D Hadamard masked
#Simulate noisy measurements
from spyrit.core.meas import HadamSplit2d

print('== Hadamard 2D masked ==')

#--- Simulation
meas_2d = HadamSplit2d(h)
A_hadam2d = meas_2d.unvectorize(meas_2d.A)
A_hadam2d = A_hadam2d*mask
A_hadam2d = A_hadam2d.numpy().astype(np.uint8)
A_hadam2d = A_hadam2d*255

print(A_hadam2d.shape)

# plot
m = 0
imagesc(A_hadam2d[m], f'{m}-th of H2')

# save as numpy array
filename = f'hadam2d_{mask_name}_{A_hadam2d.shape[0]}_{h}x{h}'
# np.save(pattern_folder / filename, A_hadam2d)
full_folder = ravel_and_save(A_hadam2d, pattern_folder, filename)

filename = full_folder / f'hadam2d_{mask_name}_mask_{h}x{h}.npy'
np.save(filename, mask_smatrix)