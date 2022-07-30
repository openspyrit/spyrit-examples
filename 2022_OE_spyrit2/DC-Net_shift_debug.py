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
from spyrit.learning.model_Had_DCAN import Permutation_Matrix, meas2img
from spyrit.misc.statistics import *
from spyrit.misc.disp import *
from spyrit.learning.nets import load_net

from spyrit.restructured.Updated_Had_Dcan import * 

#%% User-defined parameters
img_size = 64 # image size
M = 512    # number of measurements
N0 = 10    # Image intensity (in photons)
bs = 8 # Batch size

data_root = Path('data/')
stat_root = Path('models_online/') 
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

i_im = 0
img = x[i_im,:]
imagesc(np.reshape(img.numpy(),(h,w)))

#%% Acquisition matrix, permutation, and statistics

# Init DC-Net
Mean = np.load(average_file)
Cov  = np.load(cov_file)

H =  wh.walsh2_matrix(img_size)
Ord = Cov2Var(Cov)
Perm = Permutation_Matrix(Ord)
Hperm = Perm@H
Pmat = Hperm[:M,:]

#%% Pinv_Net
Forward = Forward_operator_shift_had(Pmat, Perm)

torch.manual_seed(0)    # for reproducibility
Noise = Bruit_Poisson_approx_Gauss(N0, Forward)

Prep = Preprocess_shift_poisson(N0, M, img_size**2)
DC = Pinv_orthogonal()
Denoi = ConvNet()

model = Pinv_Net(Noise, Prep, DC, Denoi)

# Load trained DC-Net
#load_net(title, model, device)
model.eval()              # Mandantory when batchNorm is used


#%%
noiseless = Forward(x)
nonoise   = Prep(Forward(N0*(x+1)/2), Forward) # image in N0 x [0,1]^N with N0*(x+1)/2
raw = Noise(x)
y   = Prep(raw, Forward)

plt.figure()
plt.plot(noiseless[:,:].T)
plt.title("noiseless data")

plt.figure()
plt.plot(raw[:,:].T)
plt.title("raw data")

plt.figure()
plt.plot(nonoise[:,:].T)
plt.title("prep no noise data")

plt.figure()
plt.plot(y[:,:].T)
plt.title("prep noise data")

#%% Plot
n_im = meas2img(noiseless[i_im,:].numpy(), Ord)
y_im = meas2img(y[i_im,:].numpy(), Ord)
diff = y_im - n_im

fig , axs = plt.subplots(1,3)
#
im0 = axs[0].imshow(n_im, cmap='gray')
fig.colorbar(im0, ax=axs[0])
axs[0].set_title(f"Noiseless")
#
im1 = axs[1].imshow(y_im, cmap='gray')
fig.colorbar(im1, ax=axs[1])
axs[1].set_title(f"Noisy, N0 = {N0}")
#
im2 = axs[2].imshow(diff, cmap='gray')
fig.colorbar(im2, ax=axs[2])
axs[2].set_title("Diff")

#%% DC_Net
# DC2 = Generalized_Orthogonal_Tikhonov(sigma_prior = Cov, M = M, N = img_size**2)
# model2 = DC_Net(Noise, Prep, DC2, Denoi)

# model2.eval()              # Mandantory when batchNorm is used

#%% Reconstruct with network
# outputs = model(inputs)
# img = outputs[i_im,0,:,:]
# img = torch2numpy(img);
# imagesc(np.reshape(img,(h,w)))

#%% Pinv reconstruction from network
outputs = model.forward_mmse(inputs)
img = outputs[i_im,0,:,:]
img = torch2numpy(img);
imagesc(np.reshape(img,(h,w)))
plt.title(f'P-Inv ; N0 = {N0}')

#%% MMSE reconstruction from network
# outputs = model2.forward_mmse(inputs)
# img = outputs[i_im,0,:,:]
# img = torch2numpy(img);
# imagesc(np.reshape(img,(h,w)))
# plt.title(f'MMSE ; N0 = {N0}')
