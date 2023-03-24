# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 12:12:42 2021

@author: ducros
"""

#%%
from __future__ import print_function, division
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path

import spyrit.misc.walsh_hadamard as wh
from spyrit.learning.model_Had_DCAN import *
from spyrit.learning.nets import *
from spyrit.misc.metrics import psnr, psnr_, batch_psnr

#%%
#- Acquisition
img_size = 64 # image size
batch_size = 256
M = 512  #number of measurements

#- Model and data paths
data_root = Path('./data/')
stats_root = Path('./stats_walsh/')

#- Plot options
plt.rcParams['pdf.fonttype'] = 42   # Save plot using type 1 font
plt.rcParams['text.usetex'] = True  # Latex
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)

transform = torchvision.transforms.Compose(
    [torchvision.transforms.functional.to_grayscale,
     torchvision.transforms.Resize((img_size, img_size)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize([0.5], [0.5])])

trainset = \
    torchvision.datasets.STL10(root=data_root, split='train+unlabeled',download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)

testset = \
    torchvision.datasets.STL10(root=data_root, split='test',download=True, transform=transform)
testloader =  torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)

dataloaders = {'train':trainloader, 'val':testloader}

#%%
inputs, _ = next(iter(dataloaders['val']))
b,c,h,w = inputs.shape

Cov = np.load(stats_root / Path("Cov_{}x{}.npy".format(img_size, img_size)))
Mean = np.load(stats_root / Path("Average_{}x{}.npy".format(img_size, img_size)))
H =  wh.walsh2_matrix(img_size)/img_size

#%% Load STL-10 Images and simulate measurements
M = 64*64//4
Ord = Cov2Var(Cov)
model_root = './models/'
epo = [0,5,10, 40]#, 55]
ind = [9,10,11]

#- free net
model_param = 'free_N_64_M_1024_epo_40_lr_0.001_sss_20_sdr_0.2_bs_256_reg_1e-07'
model = compNet(img_size, M, Mean, Cov, variant=3, H=H, Ord=Ord)

model = model.to(device)
inputs = inputs.to(device)
meas = model.forward_acquire(inputs, b, c, h, w)    # with pos/neg coefficients


#%% Plot
f, axs = plt.subplots(len(ind), len(epo)+1, figsize=(10,8),  dpi= 100)

#-- Ground-truth  
for i_ind,v_ind in enumerate(ind): 
    
    img = inputs[v_ind, 0, :, :].cpu().detach().numpy()
    axs[i_ind, 0].imshow(img, cmap='gray')
    axs[i_ind, 0].set_title("Ground-truth")
    axs[i_ind, 0].get_xaxis().set_visible(False)
    axs[i_ind, 0].get_yaxis().set_visible(False)
    axs[i_ind, 0].axis('off')

#-- Reconstrcution       
for i_epo, v_epo in enumerate(epo):   
    
    title  = 'model_epoch_{}'.format(v_epo)
    load_net((model_root + 'NET_' + model_param) / Path(title), model, device)
    recon_net  = model.forward_reconstruct(meas, b, c, h, w)
    
    for i_ind,v_ind in enumerate(ind): 
        #-- Net
        img = inputs[v_ind, 0, :, :].cpu().detach().numpy()
        rec_net = recon_net[v_ind, 0, :, :].cpu().detach().numpy()
        #-- Plot   
        axs[i_ind, i_epo+1].imshow(rec_net, cmap='gray')
        axs[i_ind, i_epo+1].set_title(f"${v_epo}$ epoch: ${psnr_(img,rec_net):.2f}$ dB")
        axs[i_ind, i_epo+1].get_xaxis().set_visible(False)
        axs[i_ind, i_epo+1].get_yaxis().set_visible(False)
        axs[i_ind, i_epo+1].axis('off')
        

f.subplots_adjust(wspace=0, hspace=0)
#plt.suptitle(model_param)
plt.savefig("train.pdf", bbox_inches=0)

#%%
# Load training history
train_path = model_root/Path('TRAIN_'+ model_param +'.pkl')
train_net  = read_param(train_path)


# #plt.rcParams.update({'font.size': 20})

# Plot
fig, ax = plt.subplots(figsize=(10,6))
ax.set_xlabel('Time (epochs)')
ax.set_ylabel('Loss (MSE)')
ax.plot(train_net.val_loss,'g--', linewidth=4)
ax.plot(train_net.train_loss,'r-.', linewidth=4)
#ax.plot(train_net_free.val_loss,'m', linewidth=4)
#ax.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
plt.grid(True)

ax.legend(('validation', 'training'),  loc='upper right')
#fig.savefig('loss_test.pdf', dpi=fig.dpi, bbox_inches='tight')# pad_inches=0.1)