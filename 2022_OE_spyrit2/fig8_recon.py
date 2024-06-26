# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 08:48:35 2022

This scripts reconstructs the images in Figure 8

NB (15-Sep-22): to debug needs to run
import collections
collections.Callable = collections.abc.Callable

"""

#%%
from pathlib import Path
import collections
collections.Callable = collections.abc.Callable

import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from spyrit.misc.statistics import Cov2Var
from spyrit.core.noise import Poisson 
from spyrit.core.meas import HadamSplit
from spyrit.core.prep import SplitPoisson
from spyrit.core.recon import DCNet, PinvNet
from spyrit.core.train import load_net
from spyrit.core.nnet import Unet
from spyrit.misc.sampling import reorder, Permutation_Matrix
from spyrit.misc.disp import add_colorbar, noaxis

from spas import read_metadata, spectral_slicing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Torch device: {device}')

#%% user-defined
# used for acquisition
N_acq = 64

# for reconstruction
N_rec = 128  # 128 or 64
M_list = [4096, 1024, 512] # for N_rec = 128
#M_list = [4095, 1024, 512] # for N_rec = 64

N0 = 10     # Check if we used 10 in the paper
stat_folder_rec = Path('./stat/') # Path('../../stat/ILSVRC2012_v10102019/')

net_arch    = 'dc-net'      # ['dc-net','pinv-net']
net_denoi   = 'unet'        # ['unet', 'cnn']
net_data    = 'imagenet'    # 'imagenet'
bs = 256
 
save_root = Path('./recon/')

#%% covariance matrix and network filnames
if N_rec==64:
    cov_rec_file= stat_folder_rec/ ('Cov_{}x{}'.format(N_rec, N_rec)+'.npy')
elif N_rec==128:
    cov_rec_file= stat_folder_rec/ ('Cov_8_{}x{}'.format(N_rec, N_rec)+'.npy')
    
#%% Networks
for M in M_list:

    if (N_rec == 128) and (M == 4096):
        net_order   = 'rect'
    else:
        net_order   = 'var'

    net_suffix  = f'N0_{N0}_N_{N_rec}_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_{bs}_reg_1e-07_light'
    
    #% Init and load trained network
    # Covariance in hadamard domain
    Cov_rec = np.load(cov_rec_file)
    
    # Sampling order
    if net_order == 'rect':
        Ord_rec = np.ones((N_rec, N_rec))
        n_sub = math.ceil(M**0.5)
        Ord_rec[:,n_sub:] = 0
        Ord_rec[n_sub:,:] = 0
        
    elif net_order == 'var':
        Ord_rec = Cov2Var(Cov_rec)
        
    # Init network  
    meas_op = HadamSplit(M, N_rec, torch.from_numpy(Ord_rec))
    noise = Poisson(meas_op, N0) # could be replaced by anything here as we just need to recon
    prep  = SplitPoisson(N0, meas_op)    
    denoi = Unet()
    model = DCNet(noise, prep, torch.from_numpy(Cov_rec), denoi)
    
    pinet = PinvNet(noise, prep)
    
    # Load trained DC-Net
    net_title = f'{net_arch}_{net_denoi}_{net_data}_{net_order}_{net_suffix}'
    title = './model/' + net_title
    load_net(title, model, device, strict = False)
    model.eval()                    # Mandantory when batchNorm is used
    
    model_pinv = PinvNet(noise, prep)
    model_pinv.to(device)
    
    model.prep.set_expe()
    model.to(device)
    
    #% Load expe data and unsplit
    data_root = Path('data/')

    data_file_prefix_list = ['zoom_x12_usaf_group5',
                             'zoom_x12_starsector',
                             'tomato_slice_2_zoomx2',
                             'tomato_slice_2_zoomx12',
                             ]
       
    
    #% Load data
    for data_file_prefix in data_file_prefix_list:
        
        print(Path(data_file_prefix) / data_file_prefix)
        
        # meta data
        meta_path = data_root / Path(data_file_prefix) / (data_file_prefix + '_metadata.json')
        _, acquisition_parameters, _, _ = read_metadata(meta_path)
        wavelengths = acquisition_parameters.wavelengths 
        
        # data
        full_path = data_root / Path(data_file_prefix) / (data_file_prefix + '_spectraldata.npz')
        raw = np.load(full_path)
        meas= raw['spectral_data']
        
        # reorder measurements to match with the reconstruction order
        Ord_acq = -np.array(acquisition_parameters.patterns)[::2]//2   # pattern order
        Ord_acq = np.reshape(Ord_acq, (N_acq,N_acq))                   # sampling map
        
        Perm_rec = Permutation_Matrix(Ord_rec)    # from natural order to reconstrcution order 
        Perm_acq = Permutation_Matrix(Ord_acq).T  # from acquisition to natural order
        meas = reorder(meas, Perm_acq, Perm_rec)
        
        #% Reconstruct a single spectral slice from full reconstruction
        wav_min = 579 
        wav_max = 579.1
        wav_num = 1
        meas_slice, wavelengths_slice, _ = spectral_slicing(meas.T, 
                                                        wavelengths, 
                                                        wav_min, 
                                                        wav_max, 
                                                        wav_num)
        with torch.no_grad():
            m = torch.Tensor(meas_slice[:2*M,:]).to(device)
            rec_gpu = model.reconstruct_expe(m)
            rec = rec_gpu.cpu().detach().numpy().squeeze()
            
        #% Plot or save 
        # rotate
        #rec = np.rot90(rec,2)
        
        fig , axs = plt.subplots(1,1)
        im = axs.imshow(rec, cmap='gray')
        noaxis(axs)
        add_colorbar(im, 'bottom')
        
        full_path = save_root / (data_file_prefix + '_' + f'{M}_{N_rec}' + '.pdf')
        fig.savefig(full_path, bbox_inches='tight')
        
        
        #% pseudo inverse
        if M==4096:
            rec_pinv_gpu = model_pinv.reconstruct_expe(m)
            rec_pinv = rec_pinv_gpu.cpu().detach().numpy().squeeze()
        
            fig , axs = plt.subplots(1,1)
            im = axs.imshow(rec_pinv, cmap='gray')
            noaxis(axs)
            add_colorbar(im, 'bottom')
            
            full_path = save_root / (data_file_prefix + '_' + f'pinv_{N_rec}' + '.pdf')
            fig.savefig(full_path, bbox_inches='tight', dpi=600)
# %%
