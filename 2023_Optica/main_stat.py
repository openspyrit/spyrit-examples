# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 10:17:43 2023

@author: ducros
"""
import torch
import numpy as np
from statistics_dev import data_loaders_ImageNet, stat_walsh1, stat_1
from pathlib import Path
from spyrit.misc.walsh_hadamard import walsh_matrix

#%% compute covariance matrices

data_folder = '../../data/ILSVRC2012_v10102019'
#dataloaders = data_loaders_ImageNet(data_folder, img_size=32, batch_size=4096)
dataloaders = data_loaders_ImageNet(data_folder, img_size=512, batch_size=1024)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#mean = mean_walsh1(dataloaders['train'], device)
#cov = cov_walsh1(dataloaders['train'], mean, device)

stat_1(dataloaders['train'], device, Path('./stat/'), n_loop=1)
stat_walsh1(dataloaders['train'], device, Path('./stat/'), n_loop=1)

#%% Compares covariance matrices (image domain vs Hadamard domain)
N = 512

stat_folder = './stat/'
trans_type = ''

cov_file   = f'Cov_1_{N}x{N}.npy'
mean_file   = f'Average_1_{N}x{N}.npy'
mean_dir  = np.load(Path(stat_folder) / mean_file)
sigma_dir = np.load(Path(stat_folder) / cov_file)

cov_file   = f'Cov_walsh1_{N}x{N}.npy'
mean_file   = f'Average_walsh1_{N}x{N}.npy'
mean_walsh  = np.load(Path(stat_folder) / mean_file)
sigma_walsh = np.load(Path(stat_folder) / cov_file)


H = walsh_matrix(N)

diff = np.linalg.norm(mean_walsh - H @ mean_dir) / np.linalg.norm(mean_walsh)
print(f'Mean difference: {diff}')

diff = np.linalg.norm(sigma_walsh - H @ sigma_dir @ H.T) / np.linalg.norm(sigma_walsh)
print(f'Covariance difference: {diff}')