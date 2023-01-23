# -*- coding: utf-8 -*-
import numpy as np
from spyrit.misc.statistics import Cov2Var
from spyrit.misc.sampling import img2mask
from matplotlib import pyplot as plt
from spyrit.misc.disp import noaxis

from pathlib import Path

#%%
M = [4096, 1024, 512] #[1024, 4095, 4096]
N = 128

title_Tag = False

C = np.load('../../stat/ILSVRC2012_v10102019/Cov_8_128x128.npy')
save_root = Path('./recon_128/')

#%% Energy/variance subsampling
Ord = Cov2Var(C)
mask_0 = img2mask(Ord, M[0])
mask_1 = img2mask(Ord, M[1])
mask_2 = img2mask(Ord, M[2])

#%% Square subsampling
Ord = np.zeros((N,N))
K = int(M[0]**.5)
M_act = K**2
Ord[:K,:K]= -np.arange(-M_act,0).reshape(K,K)
mask_3 = img2mask(Ord, M[0])

Ord = np.zeros((N,N))
K = int(M[1]**.5)
M_act = K**2
Ord[:K,:K]= -np.arange(-M_act,0).reshape(K,K)
mask_4 = img2mask(Ord, M[1])

Ord = np.zeros((N,N))
K = int(M[2]**.5)
M_act = K**2
Ord[:K,:K]= -np.arange(-M_act,0).reshape(K,K)
mask_5 = img2mask(Ord, M[2])

#%%
fig , axs = plt.subplots(1,1)
im = axs.imshow(mask_0, cmap='gray')
if title_Tag:
    axs.set_title(f"M = {M[0]}")
noaxis(axs)

full_path = save_root / ( f'mask_var_M_{M[0]}'+ '.png')
fig.savefig(full_path, bbox_inches='tight')

#%%
fig , axs = plt.subplots(1,1)
im = axs.imshow(mask_1, cmap='gray')
if title_Tag:
    axs.set_title(f"M = {M[1]}")
noaxis(axs)

full_path = save_root / ( f'mask_var_M_{M[1]}'+ '.png')
fig.savefig(full_path, bbox_inches='tight')

#%%
fig , axs = plt.subplots(1,1)
im = axs.imshow(mask_2, cmap='gray')
if title_Tag:
    axs.set_title(f"M = {M[2]}")
noaxis(axs)

full_path = save_root / ( f'mask_var_M_{M[2]}'+ '.png')
fig.savefig(full_path, bbox_inches='tight')

#%%
fig , axs = plt.subplots(1,1)
im = axs.imshow(mask_3, cmap='gray')
if title_Tag:
    axs.set_title(f"M = {M[0]}")
noaxis(axs)

full_path = save_root / ( f'mask_rect_M_{M[0]}'+ '.png')
fig.savefig(full_path, bbox_inches='tight')

#%%
fig , axs = plt.subplots(1,1)
im = axs.imshow(mask_4, cmap='gray')
if title_Tag:
    axs.set_title(f"M = {M[1]}")
noaxis(axs)

full_path = save_root / ( f'mask_rect_M_{M[1]}'+ '.png')
fig.savefig(full_path, bbox_inches='tight')

#%%
fig , axs = plt.subplots(1,1)
im = axs.imshow(mask_5, cmap='gray')
if title_Tag:
    axs.set_title(f"M = {M[2]}")
noaxis(axs)

full_path = save_root / ( f'mask_rect_M_{M[2]}'+ '.png')
fig.savefig(full_path, bbox_inches='tight')