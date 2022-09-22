# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:20:25 2022

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
from spyrit.learning.model_Had_DCAN import Permutation_Matrix
from spyrit.misc.statistics import *
from spyrit.misc.disp import *
from spyrit.learning.nets import load_net

from spyrit.restructured.Updated_Had_Dcan import * 
from spyrit.misc.disp import imagesc, add_colorbar, noaxis

#%% User-defined parameters
img_size = 64 # image size
CR = 4
M = 4096//CR    # number of measurements
N0 = 100    # Image intensity (in photons)
bs = 8 # Batch size

i_im = 4     # image index
i_noi = 2    # for 0 and 1 the 'single' offset outperforms the 'sum' offset

data_root = Path('data/')
stat_root = Path('data_online/') 
average_file = stat_root / ('Average_{}x{}'.format(img_size,img_size)+'.npy')
cov_file = stat_root / ('Cov_{}x{}'.format(img_size,img_size)+'.npy')    

title = f'./model_exp/DC_Net_Conv_net_N0_10_N_64_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07'

#%% A batch of STL-10 test images
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Torch device: {device}')

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

img_true = x[i_im,:].numpy().reshape((h,w))
imagesc(img_true)

#%% Acquisition matrix, permutation, and statistics

# Init DC-Net
Mean = np.load(average_file)
Cov  = np.load(cov_file)
H =  wh.walsh2_matrix(img_size)

# 1. low-freq order
Ord = np.zeros((64,64))
Ord[0:int(M**0.5),0:int(M**0.5)] = 1

# 2. high-energy order
Ord = Cov2Var(Cov)

# permutation
Perm = Permutation_Matrix(Ord)
Hperm = Perm@H
Pmat = Hperm[:M,:]

#%% Pinv_Net
F_split = Split_Forward_operator_ft_had(Pmat, Perm, img_size, img_size)
torch.manual_seed(i_noi)    # for reproducibility
M_split = Bruit_Poisson_approx_Gauss(N0, F_split)
P_split = Split_diag_poisson_preprocess(N0, M, img_size**2)


DC = Pinv_orthogonal()
Denoi = ConvNet()
model_split = Pinv_Net(M_split, P_split, DC, Denoi)
model_split.eval() # Mandantory when batchNorm is used in Denoi, facultatory otherwise

#%%
nonoise_split = F_split(x)
noiseless_split = F_split(N0*(x+1)/2)

torch.manual_seed(i_noi)    # for reproducibility
raw_split = M_split(x)
y_split = P_split(raw_split, F_split)

#%% Pinv reconstruction from network
raw_split = raw_split.view((bs,2*M))
outputs = model_split.reconstruct_meas2im(raw_split)
img_split = outputs[:,0,:,:]
img_split = img_split.detach().cpu().numpy().reshape((-1,h,w))

fig , axs = plt.subplots(1,3)
#
im = axs[0].imshow(img_split[0,:,:], cmap='gray')
add_colorbar(im)
#
im = axs[1].imshow(img_split[1,:,:], cmap='gray')
add_colorbar(im)
#
im = axs[2].imshow(img_split[2,:,:], cmap='gray')
add_colorbar(im)

#%% Another pinv reconstruction from network
outputs = model_split.forward_meas2im(inputs)
img_split = outputs[:,0,:,:]
img_split = img_split.detach().cpu().numpy().reshape((-1,h,w))

fig , axs = plt.subplots(1,3)
#
im = axs[0].imshow(img_split[0,:,:], cmap='gray')
add_colorbar(im)
#
im = axs[1].imshow(img_split[1,:,:], cmap='gray')
add_colorbar(im)
#
im = axs[2].imshow(img_split[2,:,:], cmap='gray')
add_colorbar(im)