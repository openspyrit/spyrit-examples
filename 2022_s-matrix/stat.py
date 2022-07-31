# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 18:19:18 2022

@author: ducros
"""

#%% compute statistics
import spyrit.misc.statistics as st
st.stat_fwalsh_S_stl10()

#%% load and visualize mean
import numpy as np
import matplotlib.pyplot as plt
A = np.load('./stats/Average_64x64.npy')

f, ax = plt.subplots(1, 1, figsize=(12,8),  dpi= 100)
im = ax.imshow(A) 

#%% load and visualize cov
C = np.load('./stats/Cov_64x64.npy')

f, ax = plt.subplots(1, 1, figsize=(12,8),  dpi= 100)
im = ax.imshow(C) 