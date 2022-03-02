# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 15:56:20 2022

@author: ducros
"""

import numpy as np
import spyrit.misc.walsh_hadamard as wh
from matplotlib import pyplot as plt


img_size = 64

H = wh.walsh2_matrix(img_size)
Mean = np.load('./stats/Average_64x64.npy')
Cov  = np.load('./stats/Cov_64x64.npy')

H_expe = np.load('../../spas/data/H.npy')
Mean_expe = np.load('../../spas/data/Average_64x64.npy')
Cov_expe  = np.load('../../spas/data/Cov_64x64.npy')

#%%
err_Cov  = np.linalg.norm(Cov_expe - Cov)
err_Mean = np.linalg.norm(Mean_expe - Mean)
err_H    = np.linalg.norm(H_expe - H)

np.where(np.isinf(Cov)==True)
np.where(np.isinf(Cov_expe)==True)
np.where(np.isinf(Cov_expe-Cov)==True)

plt.imshow(Cov_expe - Cov)

#%% Check permutation
_, acquisition_metadata, _, _ = read_metadata('./data/zoom_x1_starsector/zoom_x1_starsector_metadata.json')

ind = np.array(acquisition_metadata.patterns)[::2]//2
P_ind = permutation_from_ind(ind+1).T
P_cov = Permutation_Matrix(Cov2Var(Cov))

print(np.sum(abs(P_ind - P_cov).ravel()))