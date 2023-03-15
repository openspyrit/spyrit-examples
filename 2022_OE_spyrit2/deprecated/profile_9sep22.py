# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:34:25 2023

@author: ducros

git checkout towards_v2
git checkout -b tmp-v13 49ea3d9a3793686ebb0ec8c41ed414b286198c26

"""
# -*- coding: utf-8 -*-

from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np
import torch
from spyrit.learning.model_Had_DCAN import Permutation_Matrix, Weight_Decay_Loss
from spyrit.learning.nets import train_model, Train_par, save_net
from spyrit.restructured.Updated_Had_Dcan import *
from spyrit.misc.statistics import Cov2Var, data_loaders_ImageNet, data_loaders_stl10
import spyrit.misc.walsh_hadamard as wh


alpha = 10.0 #ph/pixel max
W = 64
M = W**2 // 8 # subsampled by factor 8
B = 10

# init reconstrcution networks
cov_file = '../../stat/ILSVRC2012_v10102019/Cov_8_64x64.npy'
#cov_file = '../../../stat/stl10/Cov_64x64.npy'
Cov = np.load(cov_file)

Ord = Cov2Var(Cov)
Perm = Permutation_Matrix(Ord)
H =  wh.walsh2_matrix(W)
Hperm = Perm @ H
Pmat = Hperm[:M,:]

FO = Split_Forward_operator_ft_had(Pmat, Perm, W, W)
Noi = Bruit_Poisson_approx_Gauss(10.0, FO)
Prep = Split_diag_poisson_preprocess(10.0, M, W**2)

Denoi = Unet()
Cov_perm = Perm @ Cov @ Perm.T
DC = Generalized_Orthogonal_Tikhonov(sigma_prior = Cov_perm, 
                                     M = M, 
                                     N = W**2)
model = DC2_Net(Noi, Prep, DC, Denoi)

# A batch of images
dataloaders = data_loaders_stl10('../../data', img_size=W, batch_size=100)  
x, _ = next(iter(dataloaders['val']))

# use GPU, if available
#device = "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)
x = x.to(device)

# warm-up
y = model(x) 

with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory = True) as prof:
    with record_function("model_inference"):
        y = model(x)   # another reconstruction, from the ground-truth image

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))