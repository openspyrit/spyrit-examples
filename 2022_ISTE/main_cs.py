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
#import spyrit.misc.walsh_hadamard as wh
from spyrit.learning.model_Had_DCAN import *
import skimage.transform as skt
import skimage.data as skd
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
ind = [72,73]
M = [2048, 1024, 512,256] 
sig = [1,]#[0.1, 0.25, 2, 16] 
#eps = np.random.standard_normal((M,))

#image = skd.shepp_logan_phantom().astype(np.float32, copy=False)


#%% Plot
f, axs = plt.subplots(4, 5, figsize=(10,10),  dpi= 100)

np.random.seed(0)
Ord = np.random.rand(h,w)
Ord = Cov2Var(Cov)

for i_M, v_M  in enumerate(M):
    H_sub = subsample(H, Ord, v_M)
    mask = img2mask(Ord, v_M)

    for i_ind,v_ind in enumerate(ind): 
        #-- Meas
        if i_ind==0:
             img = skt.resize(skd.shepp_logan_phantom().astype(np.float32, copy=False),(img_size,img_size))  
        else:
             img = inputs[v_ind, 0, :, :].cpu().detach().numpy()
        had = wh.walsh2(img)
        had = had*mask
        m = img2meas(had, Ord)
        
        #-- Recon
        rec_l2 = wh.iwalsh2(had)
        rec_tv = TV(m[:v_M], H_sub, img_size, mu = 1e-2, niter = 20).reshape(img_size,img_size)     
        
        #- Plot   
        axs[2*i_ind, 0].imshow(img, cmap='gray')
        axs[2*i_ind, 0].set_title("Ground-truth")
        axs[2*i_ind, 0].get_xaxis().set_visible(False)
        axs[2*i_ind, 0].get_yaxis().set_visible(False)
        axs[2*i_ind+1, 0].axis('off')
        # 
        axs[2*i_ind,  i_M+1].imshow(rec_l2, cmap='gray')
        axs[2*i_ind+1,i_M+1].imshow(rec_tv, cmap='gray')
        axs[2*i_ind,  i_M+1].set_title(f"L2: $M={v_M}$")
        axs[2*i_ind+1,i_M+1].set_title(f"CS: $M={v_M}$")
        #
        axs[2*i_ind,i_M+1].get_xaxis().set_visible(False)
        axs[2*i_ind,i_M+1].get_yaxis().set_visible(False)
        axs[2*i_ind+1,i_M+1].get_xaxis().set_visible(False)
        axs[2*i_ind+1,i_M+1].get_yaxis().set_visible(False)

f.subplots_adjust(wspace=0, hspace=0)
plt.savefig("cs.pdf", bbox_inches=0)