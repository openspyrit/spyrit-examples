# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:52:17 2025

@author: ducros
"""

#%%
from typing import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from spyrit.core.meas import HadamSplit2d
from spyrit.core.noise import Gaussian, Poisson
from spyrit.core.prep import Unsplit, UnsplitRescale, Rerange
from spyrit.core.inverse import PseudoInverse
from spyrit.core.recon import PinvNet, LearnedPGD, DCNet
from spyrit.core.nnet import Unet
from spyrit.external.drunet import DRUNet
from spyrit.core.train import load_net
from utility_dpgd import DualPGD, load_model


import timeit
import time

#%%

h = 128
n_batch = 64

# image
x = torch.empty(n_batch, 1, h, h).uniform_(0, 1)

#device = torch.device("cuda")
device =   torch.device("cpu")
print("Using device:", device)

x = x.to(device)

# %% HadamSplit2d, Subsampling x4, Poisson noise, pseudo-inverse

# Low-frequency sampling map with shape (64, 64)
sampling_map = torch.ones((h, h))
sampling_map[:, h // 2 :] = 0
sampling_map[h // 2 :, :] = 0


meas_op = HadamSplit2d(h, h**2//4,
                       noise_model=Poisson(100),
                       order=sampling_map, 
                       device=device) # reshape_output=True

prep_op = UnsplitRescale(100)   # (y+ - y-)/alpha 
rerange = Rerange((0, 1), (-1, 1))

y = meas_op(x)
m = prep_op(y)

# HadamSplit2d has a fast_pinv() method to get the pseudo inverse solution.
x_rec = meas_op.measure(x)  # shape is (4, 64, 64)



#%% Measure
number = 100

# Synchronization and warm-up
# see https://discuss.pytorch.org/t/why-time-time-in-python-is-inaccurte/94274/2
def recon(x):
    meas_op.measure(x)
    torch.cuda.synchronize()
    
recon(x) # warm-up that actually sends to GPU

# time
with torch.no_grad():
    torch.cuda.synchronize()
    t = timeit.timeit(lambda: recon(x), number=number)
print(f"Measure ({number}x): {t:.3f} seconds")

del t
torch.cuda.empty_cache()

#%% measure_H

number = 100
 # actually sends to GPU

# Synchronization and warm-up
# see https://discuss.pytorch.org/t/why-time-time-in-python-is-inaccurte/94274/2
def recon(x):
    meas_op.measure_H(x)
    torch.cuda.synchronize()
    
recon(x) # warm-up that actually sends to GPU

# time
with torch.no_grad():
    torch.cuda.synchronize()
    t = timeit.timeit(lambda: recon(x), number=number)
print(f"measure_H ({number}x): {t:.3f} seconds")

del t
torch.cuda.empty_cache()

#%% adjoint_H
number = 100

# Synchronization and warm-up
# see https://discuss.pytorch.org/t/why-time-time-in-python-is-inaccurte/94274/2
def recon(m):
    meas_op.adjoint_H(m)
    torch.cuda.synchronize()
    
recon(m) # warm-up that actually sends to GPU

# time
with torch.no_grad():
    torch.cuda.synchronize()
    t = timeit.timeit(lambda: recon(m), number=number)
print(f"adjoint_H ({number}x): {t:.3f} seconds")

del t
torch.cuda.empty_cache()

#%% Pinv
number = 100

# Synchronization and warm-up
# see https://discuss.pytorch.org/t/why-time-time-in-python-is-inaccurte/94274/2
def recon(m):
    meas_op.fast_pinv(m)
    torch.cuda.synchronize()
    
recon(m) # warm-up that actually sends to GPU

# time
with torch.no_grad():
    torch.cuda.synchronize()
    t = timeit.timeit(lambda: recon(m), number=number)
print(f"Pinv ({number}x): {t:.3f} seconds")

del t
torch.cuda.empty_cache()

#%% PinvNet
number = 10
model_folder_full = Path.cwd() / Path("model/")

#model_name = "pinv-net_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07_retrained_light.pth"
denoiser = OrderedDict(
    {"rerange": rerange, "denoi": Unet(), "rerange_inv": rerange.inverse()}
)
denoiser = nn.Sequential(denoiser)
# this function loads the model into the '.denoi' key present in the second
# argument. It fails if it does not find the '.denoi' key.
#load_net(model_folder_full / model_name, denoiser, device, False)

# Init
recnet = PinvNet(meas_op, prep_op, denoiser, device=device)
recnet.eval()

# Synchronization and warm-up
# see https://discuss.pytorch.org/t/why-time-time-in-python-is-inaccurte/94274/2
def recon(y):
    recnet.reconstruct(y)
    torch.cuda.synchronize()
    
recon(y) # warm-up that actually sends to GPU

# time
with torch.no_grad():
    torch.cuda.synchronize()
    t = timeit.timeit(lambda: recon(y), number=number)
    #t = timeit.timeit(lambda: recnet.reconstruct(y), number=number)
print(f"Pinv-net ({number}x): {t:.3f} seconds")

del recnet
del denoiser
torch.cuda.empty_cache()

#%% LearnedPGD
number = 10 # 9 is much longer?!

denoiser = OrderedDict(
    {"rerange": rerange, "denoi": Unet(), "rerange_inv": rerange.inverse()}
)
denoiser = nn.Sequential(denoiser)
# this function loads the model into the '.denoi' key present in the second
# argument. It fails if it does not find the '.denoi' key.
#train.load_net(model_folder_full / model_name, denoiser, device, False)

# Initialize network
recnet = LearnedPGD(meas_op, prep_op, denoiser, step_decay=0.9)
recnet.eval()
recnet = recnet.to(device)


# Synchronization and warm-uup
# see https://discuss.pytorch.org/t/why-time-time-in-python-is-inaccurte/94274/2
def recon(y):
    recnet.reconstruct(y)
    torch.cuda.synchronize()
    
recon(y) # warm-up that actually sends to GPU

# time
with torch.no_grad():
    torch.cuda.synchronize()
    t = timeit.timeit(lambda: recon(y), number=number)
print(f"LPGD-net ({number}x): {t:.3f} seconds")

del recnet
del denoiser
torch.cuda.empty_cache()


#%% LearnedPGD / alternative with time
# import time

# number = 20 # 9 is much longer?!

# denoiser = OrderedDict(
#     {"rerange": rerange, "denoi": Unet(), "rerange_inv": rerange.inverse()}
# )
# denoiser = nn.Sequential(denoiser)
# # this function loads the model into the '.denoi' key present in the second
# # argument. It fails if it does not find the '.denoi' key.
# #train.load_net(model_folder_full / model_name, denoiser, device, False)

# # Initialize network
# recnet = LearnedPGD(meas_op, prep_op, denoiser, step_decay=0.9)
# recnet.eval()
# recnet = recnet.to(device)

# torch.cuda.synchronize()
# start_time = time.perf_counter()
# with torch.no_grad():
#     for _ in range(number):
#         recnet.reconstruct(y)
    
# torch.cuda.synchronize()
# end_time = time.perf_counter()   
# t = end_time - start_time 

# print(f"LPGD-net ({number}x): {t:.3f} seconds")

# del recnet
# del denoiser
# torch.cuda.empty_cache()

#%% DCNet
number = 10
denoiser = OrderedDict(
    {"rerange": rerange, "denoi": Unet(), "rerange_inv": rerange.inverse()}
)
denoiser = nn.Sequential(denoiser)

# Load covariance prior
Cov = torch.empty(h**2, h**2).uniform_(0, 1) 

# Init
recnet = DCNet(meas_op, prep_op, Cov, denoiser, device=device)
recnet.eval()


# Synchronization and warm-uup
# see https://discuss.pytorch.org/t/why-time-time-in-python-is-inaccurte/94274/2
def recon(y):
    recnet.reconstruct(y)
    torch.cuda.synchronize()
    
recon(y) # warm-up that actually sends to GPU

# time
with torch.no_grad():
    torch.cuda.synchronize()
    t = timeit.timeit(lambda: recon(y), number=number)
print(f"DC-net ({number}x): {t:.3f} seconds")

del Cov
del recnet
del denoiser
torch.cuda.empty_cache()

#%% Pinv-PnP
number = 10

denoiser = OrderedDict(
    {
        # No rerange() needed with normalize=False
        #"rerange": rerange,
        "denoi": DRUNet(normalize=False),
        # No rerange.inverse() here as DRUNet works for images in [0,1] 
        #"rerange_inv": rerange.inverse(), 
    }
)
denoiser = nn.Sequential(denoiser)

# Initialize network
recnet = PinvNet(meas_op, prep_op, denoiser, device=device)
recnet.eval()

# Synchronization and warm-uup
# see https://discuss.pytorch.org/t/why-time-time-in-python-is-inaccurte/94274/2
def recon(y):
    recnet.reconstruct(y)
    torch.cuda.synchronize()
    
recon(y) # warm-up that actually sends to GPU

# time
with torch.no_grad():
    torch.cuda.synchronize()
    t = timeit.timeit(lambda: recon(y), number=number)
print(f"Pinv-PnP ({number}x): {t:.3f} seconds")

del recnet
del denoiser
torch.cuda.empty_cache()

#%% DPGD-PnP
number = 10

# load denoiser
n_channel, n_feature, n_layer = 1, 100, 20
model_name = "DFBNet_l1_patchsize=50_varnoise0.1_feat_100_layers_20.pth"
denoiser = load_model(
    pth=(model_folder_full / model_name).as_posix(),
    n_ch=n_channel,
    features=n_feature,
    num_of_layers=n_layer,
)

denoiser.module.update_lip((1, 50, 50))
denoiser.eval()

# Reconstruction hyperparameters
gamma = 1 / h**2
max_iter = 101
mu = 3500
crit_norm = 1e-4

# Init
recnet = DualPGD(meas_op, prep_op, denoiser, gamma, mu, max_iter, crit_norm)
recnet = recnet.to(device)

# Synchronization and warm-uup
# see https://discuss.pytorch.org/t/why-time-time-in-python-is-inaccurte/94274/2
def recon(y):
    recnet.reconstruct(y)
    torch.cuda.synchronize()
    
recon(y) # warm-up that actually sends to GPU

# time
with torch.no_grad():
    torch.cuda.synchronize()
    t = timeit.timeit(lambda: recon(y), number=number)
print(f"DPGD-PnP ({number}x): {t:.3f} seconds")

del recnet
del denoiser
torch.cuda.empty_cache()