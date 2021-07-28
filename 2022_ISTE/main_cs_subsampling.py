# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 18:04:38 2021

@author: ducros
"""

#%%
from __future__ import print_function, division
import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
import spyrit.misc.walsh_hadamard as wh
from spyrit.learning.model_Had_DCAN import *
import skimage.transform as skt
#import skimage.data as skd
from spyrit.misc.metrics import psnr, psnr_, batch_psnr
#%%
from scipy.sparse.linalg import aslinearoperator
import pylops

#mu = 1.5
def TV(y, H, img_size, mu = 0.15, lamda = [0.1, 0.1], niter = 20, niterinner = 10):
    ny = img_size;
    nx = img_size;
    A = aslinearoperator(H);
    H_p = pylops.LinearOperator(A)
    Dop = \
        [pylops.FirstDerivative(ny * nx, dims=(ny, nx), dir=0, edge=False,
                                kind='backward', dtype=np.float64),
         pylops.FirstDerivative(ny * nx, dims=(ny, nx), dir=1, edge=False,
                                kind='backward', dtype=np.float64)]
    xinv, niter = \
    pylops.optimization.sparsity.SplitBregman(H_p, Dop, y.flatten(),
                                              niter, niterinner,
                                              mu=mu, epsRL1s=lamda,
                                              tol=1e-4, tau=1., show=False,
                                              **dict(iter_lim=5, damp=1e-4))
    return xinv;

#%%
#- Acquisition
img_size = 64 # image size
batch_size = 512
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

transform = transforms.Compose(
    [transforms.functional.to_grayscale,
     transforms.Resize((img_size, img_size)),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

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

#%% STL-10 Images
inputs = inputs.to(device)


#%% Recon from Walsh-ordered 2D
ind = [104,107]
M = 1024

#- mask
np.random.seed(0)
ord1 = np.random.rand(h,w)
ord2 = np.ones((h,h))
n_sub = math.ceil(M**0.5)
ord2[:,n_sub:] = 0
ord2[n_sub:,:] = 0
ord3 = Cov2Var(Cov)
Ord = [ord1, ord2, ord3]
Ord_label = ["Random", "Natural", "Energy"]


#%% Plot
f, axs = plt.subplots(len(ind)+1, len(Ord)+1, figsize=(2*(len(Ord)+1),2*(len(ind)+1)+1),  dpi= 100)


for Ord_i, Ord_v  in enumerate(Ord):
    
    H_sub = subsample(H, Ord_v, M)
    mask = img2mask(Ord_v, M)

    for i_ind,v_ind in enumerate(ind):
        
        img = inputs[v_ind, 0, :, :].cpu().detach().numpy()
        had = wh.walsh2(img)
        had = had*mask
        m = img2meas(had, Ord_v)
         
        #-- Plot mask
        if i_ind==0:
            axs[0, 0].axis('off')
            axs[0, Ord_i+1].imshow(mask, cmap='gray')
            axs[0, Ord_i+1].set_title(f"{Ord_label[Ord_i]}")
            axs[0, Ord_i+1].get_xaxis().set_visible(False)
            axs[0, Ord_i+1].get_yaxis().set_visible(False)
        
        #-- Recon
        rec_l2 = wh.iwalsh2(had)
        rec_tv = TV(m[:M], H_sub, img_size, mu = 1e-2, niter = 20).reshape(img_size,img_size)     
        rec_tv /= img_size # CHECK THIS OUT
        
        #- Plot recon   
        axs[i_ind+1, 0].imshow(img, cmap='gray')
        #if i_ind==0:
        axs[i_ind+1, 0].set_title("Ground-truth")
        axs[i_ind+1, 0].get_xaxis().set_visible(False)
        axs[i_ind+1, 0].get_yaxis().set_visible(False)
        # 
        axs[i_ind+1,  Ord_i+1].imshow(rec_tv, cmap='gray')
        axs[i_ind+1,  Ord_i+1].set_title(f"${psnr_(img,rec_tv):.2f}$ dB")
        axs[i_ind+1,  Ord_i+1].get_xaxis().set_visible(False)
        axs[i_ind+1,  Ord_i+1].get_yaxis().set_visible(False)


f.subplots_adjust(wspace=0, hspace=0)
plt.savefig("cs_subsample.pdf", bbox_inches=0)