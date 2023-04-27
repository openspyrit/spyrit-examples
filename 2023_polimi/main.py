#%% -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:41:56 2023

@author: ducros
"""
import pickle

import torch
import numpy as np
from spyrit.core.prep import DirectPoisson
from spyrit.core.recon import PinvNet
from spyrit.core.meas import Linear
from spyrit.core.noise import NoNoise, Poisson
from spyrit.misc.statistics import data_loaders_stl10
from spyrit.misc.disp import imagesc
 
h = 32
M = 32*32 // 2
B = 10

# A batch of images
dataloaders = data_loaders_stl10('../../data', img_size=h, batch_size=10)  
x, _ = next(iter(dataloaders['train']))
# N.B.: no view here compared to previous example


# load patterns
filepath = "D:/Creatis/Projets/1_En cours/2023_polimi/Dataset/"
filename = "measurement_matrix_undersampled"
filename = "measurement_matrix_full_sampled"
with open(filepath + filename + '.pickle','rb') as f:
    [H] = pickle.load(f)
    
H = H[:M,:]
    
# Operators
alpha = 1000.0
meas_op = Linear(H, pinv=True) 
meas_op.h, meas_op.w = h, h

#noise = NoNoise(meas_op)        # noiseless
#prep = DirectPoisson(1.0, meas_op)

noise = Poisson(meas_op, alpha) # poisson noise
prep = DirectPoisson(alpha, meas_op)

pinv_net = PinvNet(noise, prep)

# use GPU, if available
#device = "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pinv_net = pinv_net.to(device)
x = x.to(device)

# measurements and images
y = pinv_net.acquire(x)
z = pinv_net.reconstruct(y)
#z = pinv_net(x)

# reshape
x_plot = x.view(-1,h,h).cpu().numpy() 
z_plot = z.view(-1,h,h).cpu().numpy()
z_plot[0,0,0] = 0.0

# plot
imagesc(x_plot[0,:,:], 'Ground-truth image')
imagesc(z_plot[0,:,:], f'Reconstructed image ({device})')

#%%
x_np = x.view(-1,h*h)[0,:].cpu().numpy()
y_np = y.view(-1,M)[0,:].cpu().numpy()
x_rec, _, _, _ = np.linalg.lstsq(H, y_np)

x_rec = np.reshape(x_rec, (32, 32))
x_rec = (x_rec/alpha - 0.5)*2
x_rec[0,0] = 0.0

imagesc(x_rec, 'Reconstructed image (numpy)')