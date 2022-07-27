# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:09:38 2022

@author: ducros
"""

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
from spyrit.learning.nets import load_net

from spyrit.restructured.Updated_Had_Dcan import * 

#%% User-defined parameters
img_size = 64 # image size
M = 512    # number of measurements
N0 = 10    # Image intensity (in photons)
bs = 8 # Batch size

i_im = 3     # image index
i_noi = 0    # noise sample index

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

img_true = x[i_im,:].numpy().reshape((h,w))
imagesc(img_true)

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
F_shift = Forward_operator_shift_had(Pmat, Perm)
F_pos   = Forward_operator_pos(Pmat, Perm)

torch.manual_seed(i_noi)    # for reproducibility
M_shift = Bruit_Poisson_approx_Gauss(N0, F_shift)
M_pos   = Bruit_Poisson_approx_Gauss(N0, F_pos)


P_shift = Preprocess_shift_poisson(N0, M, img_size**2)
P_pos   = Preprocess_pos_poisson(N0, M, img_size**2)


DC = Pinv_orthogonal()
Denoi = ConvNet()

model_shift = Pinv_Net(M_shift, P_shift, DC, Denoi)
model_pos   = Pinv_Net(M_pos, P_pos, DC, Denoi)

# Load trained DC-Net
#load_net(title, model, device)
model_shift.eval()              # Mandantory when batchNorm is used
model_pos.eval()              # Mandantory when batchNorm is used


#%%
nonoise_shift = F_shift(x)
nonoise_pos   = F_pos(x)

noiseless_shift = F_shift(N0*(x+1)/2) # image in N0 x [0,1]^N with N0*(x+1)/2

raw_shift = M_shift(x)
raw_pos = raw_shift[:,1:]   # same dataset for both  #M_pos(x)

y_shift = P_shift(raw_shift, F_shift)
y_pos   = P_pos(raw_pos, F_pos)

#%%
plt.figure()
plt.plot(nonoise_shift[:,:].T)
plt.title("noiseless data")

plt.figure()
plt.plot(nonoise_pos[:,:].T)
plt.title("noiseless data")

plt.figure()
plt.plot(y_shift[:,:].T)
plt.title("raw shift")

plt.figure()
plt.plot(y_pos[:,:].T)
plt.title("raw pos")


#%% Pinv reconstruction from network
outputs = model_pos.reconstruct_mmse(raw_pos.view((bs,1,M)),64,64)
img_pos = outputs[i_im,0,:,:]
img_pos.detach().cpu().numpy().reshape(h,w)

outputs = model_shift.reconstruct_mmse(raw_shift.view((bs,1,M+1)),64,64)
img_shift = outputs[i_im,0,:,:]
img_shift = img_shift.detach().cpu().numpy().reshape(h,w)

outputs = model_shift.reconstruct_mmse(noiseless_shift.view((bs,1,M+1)),64,64)
img_nonoise = outputs[i_im,0,:,:]
img_nonoise = img_nonoise.detach().cpu().numpy().reshape(h,w)

#img_shift[0,0] = 0

fig , axs = plt.subplots(2,3)
#
im0 = axs[0,0].imshow(img_shift, cmap='gray')
fig.colorbar(im0, ax=axs[0,0])
axs[0,0].set_title(f'P-Inv shift patterns; N0 = {N0}')
#
im1 = axs[0,1].imshow(img_pos, cmap='gray')
fig.colorbar(im1, ax=axs[0,1])
axs[0,1].set_title(f"P-Inv pos patterns, N0 = {N0}")
#
im2 = axs[0,2].imshow(img_nonoise, cmap='gray')
fig.colorbar(im2, ax=axs[0,2])
axs[0,2].set_title("No noise")
#
im3 = axs[1,0].imshow(img_shift-img_nonoise, cmap='gray')
fig.colorbar(im3, ax=axs[1,0])
axs[1,0].set_title(f'P-Inv shift patterns; N0 = {N0}')
#
im4 = axs[1,1].imshow(img_pos-img_nonoise, cmap='gray')
fig.colorbar(im4, ax=axs[1,1])
axs[1,1].set_title(f"P-Inv pos patterns, N0 = {N0}")
#
im5 = axs[1,2].imshow(img_true, cmap='gray')
fig.colorbar(im5, ax=axs[1,2])
axs[1,2].set_title(f"P-Inv pos patterns, N0 = {N0}")