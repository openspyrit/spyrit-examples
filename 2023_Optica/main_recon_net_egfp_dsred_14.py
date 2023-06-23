# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:05:36 2023

@author: ducros
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 18:06:44 2023

Reconstructions with data simulated using experimental measurements 

@author: ducros
"""
#%% to debug needs to run
import collections
collections.Callable = collections.abc.Callable

fig_folder = './figure/'

#%% Load experimental measurement matrix (after split compensation)
from pathlib import Path
from spyrit.misc.disp import imagesc, add_colorbar
import matplotlib.pyplot as plt
from spyrit.misc.walsh_hadamard import walsh_matrix
import numpy as np

M = 128
N = 512
data_folder = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D/'
mat_folder = '/Reconstruction/Mat_rc/'

# load
H_exp = np.load(Path(data_folder + mat_folder) / f'motifs_Hadamard_{M}_{N}.npy')
H_exp /= H_exp[0,16:500].mean()
H_exp = H_exp #/ H_exp[0,:]
H_tar = walsh_matrix(N)
H_tar = H_tar[:M]

# plot
f, axs = plt.subplots(2, 1)
axs[0].set_title('Target measurement patterns')
im = axs[0].imshow(H_tar, cmap='gray') 
add_colorbar(im, 'bottom')
axs[0].get_xaxis().set_visible(False)

axs[1].set_title('Experimental measurement patterns')
im = axs[1].imshow(np.flip(H_exp,1).copy(), cmap='gray') 
add_colorbar(im, 'bottom')
axs[1].get_xaxis().set_visible(False)

#plt.savefig(Path(fig_folder) / 'patterns', bbox_inches='tight', dpi=600)

#%% Simulate measuments
import torch
from recon_dev import DC1Net, Pinv1Net, Tikho1Net
from statistics_dev import data_loaders_ImageNet
from spyrit.core.meas import LinearSplit
from spyrit.core.prep import SplitPoisson
from spyrit.core.noise import Poisson
import matplotlib.pyplot as plt
from meas_dev import Hadam1Split

M = 128
N = 512
alpha = 1e0 # in photons/pixels
b = 0

linop_tar = Hadam1Split(M,N)
noise_tar = Poisson(linop_tar, alpha)
prep_tar  = SplitPoisson(alpha, linop_tar)

linop_exp = LinearSplit(H_exp, pinv=True)
noise_exp = Poisson(linop_exp, alpha)
prep_exp = SplitPoisson(alpha, linop_exp)
    
#%% Load prep data
save_tag = False
data_folder = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D/'
data_subfolder = 'data_2023_03_14/'
Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/'
Run = 'RUN0002' 
Ns = int(Run[-1])+5
save_folder = '/Preprocess/'
Nl, Nh, Nc = 512, 128, 128

channel = 100 # 10, 55, 100

save_folder = '/Preprocess/'
filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_pos.npy'
prep_pos = np.load(Path(data_folder+save_folder) / filename)

filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_neg.npy'
prep_neg =  np.load(Path(data_folder+save_folder) / filename)

# spectral dimension comes first
prep_pos = np.moveaxis(prep_pos, -1, 0)
prep_neg = np.moveaxis(prep_neg, -1, 0)

print(f'Pos data: range={prep_pos.max() - prep_pos.min()} counts; mean={prep_pos.mean()} counts')

# param #1
# background  = prep_pos.min()
# gain = 1e6

# param #2
background  = 0
gain = 1e0

prep_pos = gain*(prep_pos - background)
prep_neg = gain*(prep_neg - background)


nc, nl, nh = prep_neg.shape
y_exp = np.zeros((nc, nl, 2*nh))
y_exp[:,:,::2]  = prep_pos #/np.expand_dims(prep_pos[:,:,0], axis=2)
y_exp[:,:,1::2] = prep_neg #/np.expand_dims(prep_pos[:,:,0], axis=2)
y_exp = torch.from_numpy(y_exp)

y_exp = y_exp[channel,:,:].to(torch.float32)
m_exp = prep_exp(y_exp)

fig, axs = plt.subplots(1, 2)#, figsize=(15,7))
fig.suptitle(fr'M = {M}, N = {N}')

im = axs[0].imshow(y_exp[:,:].cpu())
axs[0].set_title('Meas (raw)')
plt.colorbar(im, ax=axs[0])

im = axs[1].imshow(m_exp[:,:].cpu())
axs[1].set_title('Meas (after prep)')
plt.colorbar(im, ax=axs[1])

if save_tag:
    save_folder = Path(fig_folder) / Path(data_subfolder)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_folder / f'T{Ns}_{Run}_measurements', bbox_inches='tight', dpi=600)

#%% Pseudo inverse + unet
from spyrit.core.nnet import Unet
from spyrit.core.train import load_net

save_tag = True
y = y_exp
net_prefix = 'pinv-net_unet_imagenet_ph_10'
net_suffix = 'N_512_M_128_epo_30_lr_0.001_sss_10_sdr_0.5_bs_20_reg_1e-07'

#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
y = y.to(device)
y = y.view(-1,1,N,2*M) 

# Reconstruction using target patterns
pinvnet_tar = Pinv1Net(noise_tar, prep_tar, Unet())

# Load trained DC-Net
title = './model/' + net_prefix + '_Hadam1_' + net_suffix
load_net(title, pinvnet_tar, device, False)
pinvnet_tar.to(device)  # Mandantory when batchNorm is used
pinvnet_tar.eval()

# Reconstruction using target patterns ----------------------------------------
# version bidouille
#fact_tar = 2e-2
#x_pinv = pinvnet_tar.reconstruct(y*fact_tar)

x_pinv = pinvnet_tar.reconstruct_expe(y)
x_pinv = x_pinv.view(-1,N,N).detach()

# Plot
fig, axs = plt.subplots(1, 2, figsize=(10,5))
fig.suptitle(f'Spectral channel: {channel}')

im = axs[0].imshow(x_pinv[0,:,:].cpu())
axs[0].set_title(f'Pinv, target patterns')
plt.colorbar(im, ax=axs[0])

# Reconstruction using target patterns
del pinvnet_tar
pinvnet_exp = Pinv1Net(noise_exp, prep_exp, Unet())

# Load trained DC-Net
title = './model/' + net_prefix + '_exp_' + net_suffix
load_net(title, pinvnet_exp, device, False)
pinvnet_exp.eval()                    # Mandantory when batchNorm is used   

pinvnet_exp.to(device)  # Mandantory when batchNorm is used
pinvnet_exp.eval()

# Reconstruction using experimental patterns ----------------------------------
# version bidouille
# fact_exp = 1e-4
# x_pinv = pinvnet_exp.reconstruct(y*fact_exp)

x_pinv = pinvnet_exp.reconstruct_expe(y)

x_pinv = x_pinv.view(-1,N,N).detach()

# Plot
im = axs[1].imshow(x_pinv[0,:,:].cpu())
axs[1].set_title(f'Pinv, experimental patterns')
plt.colorbar(im, ax=axs[1])
            
if save_tag:
    save_folder = Path(fig_folder) / Path(data_subfolder)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_folder / f'T{Ns}_{Run}_c{channel}_pinvnet.png', bbox_inches='tight', dpi=600)
