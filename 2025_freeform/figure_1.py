# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 20:22:44 2025

@author: ducros
"""

#%%
import torch
import numpy as np

from pathlib import Path

h = 4  # image size hxh

mask_type = None         # 'skew' or None to load a PNG
mask_filename = f'roi_{h}x{h}.png'  # only if mask_type is not 'skew'
image_filename = f'dog.png'

pattern_folder = Path('figures/figure_1/')


#%%
from matplotlib.pyplot import imread
from matplotlib.pyplot import imsave

img = imread(pattern_folder/image_filename)

# 10% noise
img_noise_10 = img + img.max() * 0.1 * np.random.standard_normal(size=(img.shape[0],img.shape[1]))
imsave(pattern_folder/(image_filename[:-4]+'_n10.png'), img_noise_10, cmap='gray')

# 20% noise
img_noise_20 = img + img.max() * 0.2 * np.random.standard_normal(size=(img.shape[0],img.shape[1]))
imsave(pattern_folder/(image_filename[:-4]+'_n20.png'), img_noise_20, cmap='gray')

# 50% noise
img_noise_50 = img + img.max() * 0.5 * np.random.standard_normal(size=(img.shape[0],img.shape[1]))
imsave(pattern_folder/(image_filename[:-4]+'_n50.png'), img_noise_50, cmap='gray')


from PIL import Image

mask = Image.open(pattern_folder/mask_filename)
mask_resized = mask.resize((img.shape[0],img.shape[1]), Image.NEAREST)
mask_resized = np.array(mask_resized)[:,:,0]

# no noise
img_0 = img.deepcopy
img_0[mask_resized==0] = img.min()
imsave(pattern_folder/(image_filename[:-4]+'_roi.png'), img_0, cmap='gray')

# 10% noise
img_noise_10[mask_resized==0] = img_noise_10.min()
imsave(pattern_folder/(image_filename[:-4]+'_n10_roi.png'), img_noise_10, cmap='gray')

# 20% noise
img_noise_20[mask_resized==0] = img_noise_20.min()
imsave(pattern_folder/(image_filename[:-4]+'_n20_roi.png'), img_noise_20, cmap='gray')

# 50% noise
img_noise_50[mask_resized==0] = img_noise_50.min()
imsave(pattern_folder/(image_filename[:-4]+'_n50_roi.png'), img_noise_50, cmap='gray')

#%%
from matplotlib.pyplot import imsave
def just_save(A, folder, name):
    
    folder_full = Path(folder) / name
    Path(folder_full).mkdir(parents=True, exist_ok=True)
    
    for i in range(A.shape[0]):
        # save
        filename = f'{name}_{i}.png'
        imsave(folder_full/filename, A[i])
        
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
full_folder = just_save(A_hadam1d, pattern_folder, filename)

filename = full_folder / f'hadam1d_{mask_name}_mask_{h}x{h}.npy'
np.save(filename, mask)

#%% Arbitrary shape -- S matrix
# from spyrit.misc.walsh_hadamard import walsh_S_matrix
# from spyrit.core.meas import FreeformLinear

# print('== S-matrix 1D ==')

# # S matrix
# K = N_pixel-1
# H = torch.from_numpy(walsh_S_matrix(K))

# # ROI indices
# ind_array_0_S = ind_array[0][:-1]
# ind_array_1_S = ind_array[1][:-1]
# mask_smatrix = np.zeros_like(mask)
# mask_smatrix[ind_array_0_S,ind_array_1_S] = True
# imagesc(mask_smatrix, 'mask (S-matrix)')

# #--- Simulation
# meas_1d = FreeformLinear(H, 
#                     meas_shape = (h,h), 
#                     index_mask = torch.stack((ind_array_0_S, ind_array_1_S)),
#                     dtype = torch.uint8, # Check why default dtype not working here ???
#                     )

# # DMD pattern
# A_smatrix = meas_1d.unvectorize(meas_1d.H)
# A_smatrix = A_smatrix.numpy()
# A_smatrix = A_smatrix*255
# print(A_smatrix.shape)

# # plot
# m = 0
# imagesc(A_smatrix[m], f'{m}-th of S-matrix')
    
# # save as numpy array
# filename = f'smatrix_{mask_name}_{A_smatrix.shape[0]}_{h}x{h}'
# # np.save(pattern_folder/filename, A_smatrix)
# full_folder = just_save(A_smatrix, pattern_folder, filename)

# filename = full_folder / f'smatrix_{mask_name}_mask_{h}x{h}.npy'
# np.save(filename, mask_smatrix)

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
# np.save(pattern_folder / filename, A_hadam2d)
full_folder = just_save(A_hadam2d, pattern_folder, filename)

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
full_folder = just_save(A_hadam2d, pattern_folder, filename)