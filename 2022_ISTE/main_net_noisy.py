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
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
#import spyrit.misc.walsh_hadamard as wh
from spyrit.learning.model_Had_DCAN import *
from spyrit.learning.nets import *
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

#%% STL-10 Images and network
N0 = 10 # mean of the maximun number of photons
sig = 0.0 # range in prct of the maximun number of photons

M = 64*64//4
Ord = Cov2Var(Cov)
inputs = inputs.to(device)


#%% Load and recon
# noiseless
model_root = './models/' 
title_free  = 'NET_c0mp_N_64_M_1024_epo_40_lr_0.001_sss_20_sdr_0.2_bs_256_reg_1e-07'

# 50 photons
model_root = './models/'
title_50  = 'NET_c0mp_N0_50.0_sig_0.0_N_64_M_1024_epo_40_lr_0.001_sss_20_sdr_0.2_bs_256_reg_1e-07'

# 10 photons
model_root = './models/'
title_10  = 'NET_c0mp_N0_10.0_sig_0.0_N_64_M_1024_epo_40_lr_0.001_sss_20_sdr_0.2_bs_256_reg_1e-07'

# 2 photons
model_root = './models/'
title_2  = 'NET_c0mp_N0_2.0_sig_0.0_N_64_M_1024_epo_40_lr_0.001_sss_20_sdr_0.2_bs_256_reg_1e-07'


#%% Recon from Walsh-ordered 2D
ind = 72
ph = [50, 10, 2] # nb of photons

#%% Plot
f, axs = plt.subplots(3, 5, figsize=(10,7),  dpi= 100)

img = inputs[ind, 0, :, :].cpu().detach().numpy()

for ph_i, ph_v in enumerate(ph):
    
    #-- Meas
    model = noiCompNet(img_size, M, Mean, Cov, variant=0, N0=ph_v, sig=sig, H=H, Ord=Ord)
    model = model.to(device)
    torch.manual_seed(0)    # for reproducibility
    #torch.seed()           # for random measurements
    meas = model.forward_acquire(inputs, b, c, h, w)    # with pos/neg coefficients
    
    #-- Recon
    load_net(model_root / Path(title_free), model, device)
    recon_free = model.forward_reconstruct(meas, b, c, h, w).cpu().detach().numpy()
    load_net(model_root / Path(title_50), model, device)
    recon_50 = model.forward_reconstruct(meas, b, c, h, w).cpu().detach().numpy()
    load_net(model_root / Path(title_10), model, device)
    recon_10 = model.forward_reconstruct(meas, b, c, h, w).cpu().detach().numpy()
    load_net(model_root / Path(title_2), model, device)
    recon_2 = model.forward_reconstruct(meas, b, c, h, w).cpu().detach().numpy()
    #
    rec_free  = recon_free[ind, 0, :, :]
    rec_50 = recon_50[ind, 0, :, :]
    rec_10 = recon_10[ind, 0, :, :]
    rec_2 = recon_2[ind, 0, :, :]
    
    #- Plot   
    axs[ph_i, 0].imshow(img, cmap='gray')
    axs[ph_i, 0].set_title("Ground-truth")
    axs[ph_i, 1].imshow(rec_free, cmap='gray')
    axs[ph_i, 1].set_title(f"NET no noise: ${psnr_(img,rec_free):.2f}$ dB")
    axs[ph_i, 2].imshow(rec_50, cmap='gray')
    axs[ph_i, 2].set_title(f"50 ph: ${psnr_(img,rec_50):.2f}$ dB")
    axs[ph_i, 3].imshow(rec_10, cmap='gray')
    axs[ph_i, 3].set_title(f"10 ph: ${psnr_(img,rec_10):.2f}$ dB")
    axs[ph_i, 4].imshow(rec_2, cmap='gray')
    axs[ph_i, 4].set_title(f"2 ph: ${psnr_(img,rec_2):.2f}$ dB")

    
# remove axes
for ax in iter(axs.flatten()):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')

# row labels
rows = ['{} photons'.format(row) for row in ph]
for ax, row in zip(axs[:,0], rows):
    ax.set_ylabel(row,  size='large')#, rotation=0,)
    ax.get_yaxis().set_visible(True)
    ax.axis('on')
    #
    #ax.xaxis.set_visible(False)
    plt.setp(ax.spines.values(), visible=False)  # make spines (the box) invisible
    ax.tick_params(left=False, labelleft=False)  # remove ticks and labels for the left axis
    ax.patch.set_visible(False) #remove background patch (only needed for non-white background)
    

f.subplots_adjust(wspace=0, hspace=0)
#plt.suptitle(f"Measurement with ${N0}$ photons $\pm {sig}$")
plt.savefig("net_noise.pdf", bbox_inches=0)