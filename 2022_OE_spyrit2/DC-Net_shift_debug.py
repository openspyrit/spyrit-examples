# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 19:14:12 2022

@author: ducros
"""

# -*- coding: utf-8 -*-

#%%
import torch
import numpy as np
from torchvision import datasets, transforms
from pathlib import Path
import spyrit.misc.walsh_hadamard as wh

# from spyrit.misc.statistics import stat_walsh_stl10
from spyrit.misc.statistics import *
from spyrit.misc.disp import *

from spyrit.restructured.Updated_Had_Dcan import * 

#%% User-defined parameters
img_size = 64 # image size
M = 512    # number of measurements
N0 = 50    # Image intensity (in photons)
bs = 1 # Batch size

data_root = Path('data/')
stat_root = Path('models_online/') 
average_file = stat_root / ('Average_{}x{}'.format(img_size,img_size)+'.npy')
cov_file = stat_root / ('Cov_{}x{}'.format(img_size,img_size)+'.npy')    

#%% A batch of STL-10 test images
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Torch device: {device}')

torch.manual_seed(7)

transform = transforms.Compose(
    [transforms.functional.to_grayscale,
     transforms.Resize((img_size, img_size)),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

testset = \
    torchvision.datasets.STL10(root=data_root, split='test',download=False, transform=transform)
testloader =  torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)

#%% Plot an image
inputs, _ = next(iter(testloader))
b,c,h,w = inputs.shape

x = inputs.view(b*c,w*h)
x_0 = torch.zeros_like(x)

i_im = 0
img = x[i_im,:]
img = img.numpy();
imagesc(np.reshape(img,(h,w)))

#%% Acquisition mateix, permutation, and statistics
# Init DC-Net
Mean = np.load(average_file)
Cov  = np.load(cov_file)

H =  wh.walsh2_matrix(img_size)
Ord = Cov2Var(Cov)
Perm = Permutation_Matrix(Ord)
Hperm = Perm@H
Pmat = Hperm[:M,:]

#%%
Forward = Forward_operator_shift_had(Pmat, Perm)
Noise = Bruit_Poisson_approx_Gauss(N0, Forward)
Prep = Preprocess_shift_poisson(N0, M, img_size**2)
DC = Generalized_Orthogonal_Tikhonov(sigma_prior = Cov, M = M, N = img_size**2)
Denoi = ConvNet()

model = DC_Net(Noise, Prep, DC, Denoi)

# Load trained DC-Net
#load_net(title, model, device)
model.eval()              # Mandantory when batchNorm is used

#%% Reconstruct with network
outputs = model(inputs)
img = outputs[i_im,0,:,:]
img = torch2numpy(img);
imagesc(np.reshape(img,(h,w)))

#%% MMSE reconstruction from network
outputs = model.forward_mmse(inputs)
img = outputs[i_im,0,:,:]
img = torch2numpy(img);
imagesc(np.reshape(img,(h,w)))