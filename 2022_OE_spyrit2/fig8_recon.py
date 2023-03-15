# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 08:48:35 2022

This scripts reconstructs the images in Figure 8

NB (15-Sep-22): to debug needs to run
import collections
collections.Callable = collections.abc.Callable

"""

#%%
import torch
import numpy as np
import math
from matplotlib import pyplot as plt
from pathlib import Path
# get debug in spyder
import collections
collections.Callable = collections.abc.Callable

from spyrit.misc.statistics import Cov2Var
from spyrit.core.noise import Poisson 
from spyrit.core.meas import HadamSplit
from spyrit.core.prep import SplitPoisson
from spyrit.core.recon import DCNet, PinvNet
from spyrit.core.train import load_net
from spyrit.core.nnet import Unet
from spyrit.misc.sampling import reorder
from spyrit.misc.disp import add_colorbar, noaxis


from spas import read_metadata, spectral_slicing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Torch device: {device}')

#%% user-defined

# for reconstruction
N_rec = 128  # 128 or 64
M_list = [4096] # [4096, 1024, 512]   for N_rec = 64 
N0 = 10     # Check if we used 10 in the paper
stat_folder_rec = Path('../../stat/ILSVRC2012_v10102019/')

# used for acquisition
N_acq = 64
stat_folder_acq =  Path('./data_online/') 

net_arch    = 'dc-net'      # ['dc-net','pinv-net']
net_denoi   = 'unet'        # ['unet', 'cnn']
net_data    = 'imagenet'    # 'imagenet' 
save_root = Path('./recon_128/')

#%% covariance matrix and network filnames
stat_folder_rec = Path('../../stat/ILSVRC2012_v10102019/')

if N_rec==64:
    cov_rec_file= stat_folder_rec/ ('Cov_{}x{}'.format(N_rec, N_rec)+'.npy')
elif N_rec==128:
    cov_rec_file= stat_folder_rec/ ('Cov_8_{}x{}'.format(N_rec, N_rec)+'.npy')
    
bs = 256
#  
stat_folder_acq =  Path('./data_online/') 
cov_acq_file= stat_folder_acq / ('Cov_{}x{}'.format(N_acq, N_acq)+'.npy')


#%% Networks
for M in M_list:
    
    if (N_rec == 128) and (M == 4096):
        net_order   = 'rect'
    else:
        net_order   = 'var'

    net_suffix  = f'N0_{N0}_N_{N_rec}_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_{bs}_reg_1e-07_light'
    net_folder= f'{net_arch}_{net_denoi}_{net_data}/'
    
    #%% Init and load trained network
    # Covariance in hadamard domain
    Cov_rec = np.load(cov_rec_file)
    Cov_acq = np.load(cov_acq_file)
    Ord_acq = Cov2Var(Cov_acq)
    
    # Sampling order
    if net_order == 'rect':
        Ord_rec = np.ones((N_rec, N_rec))
        n_sub = math.ceil(M**0.5)
        Ord_rec[:,n_sub:] = 0
        Ord_rec[n_sub:,:] = 0
        
    elif net_order == 'var':
        Ord_rec = Cov2Var(Cov_rec)
        
    # Init network  
    meas = HadamSplit(M, N_rec, Ord_rec)
    noise = Poisson(meas, N0) # could be replaced by anything here as we just need to recon
    prep  = SplitPoisson(N0, M, N_rec**2)    
    denoi = Unet()
    model = DCNet(noise, prep, Cov_rec, denoi)
    
    pinet = PinvNet(noise, prep)
    
    # Load trained DC-Net
    net_title = f'{net_arch}_{net_denoi}_{net_data}_{net_order}_{net_suffix}'
    title = './model_v2/' + net_folder + net_title
    load_net(title, model, device, strict = False)
    model.eval()                    # Mandantory when batchNorm is used
    
    model_pinv = PinvNet(noise, prep)
    model_pinv.to(device)
    
    model.prep.set_expe()
    model.to(device)
    
    #%% Load expe data and unsplit
    data_root = Path('data_online/')
    data_folder_list = [Path('usaf_x12/'),
                        Path('star_sector_x12/'),
                        Path('tomato_slice_x2/'),
                        Path('tomato_slice_x12'),
                        ]
    
    data_file_prefix_list = ['zoom_x12_usaf_group5',
                             'zoom_x12_starsector',
                             'tomato_slice_2_zoomx2',
                             'tomato_slice_2_zoomx12',
                             ]
       
    
    #%% Load data
    for data_folder,data_file_prefix in zip(data_folder_list,data_file_prefix_list):
        
        print(data_folder / data_file_prefix)
        
        # meta data
        meta_path = data_root / data_folder / (data_file_prefix + '_metadata.json')
        _, acquisition_parameters, _, _ = read_metadata(meta_path)
        wavelengths = acquisition_parameters.wavelengths 
        
        # data
        full_path = data_root / data_folder / (data_file_prefix + '_spectraldata.npz')
        raw = np.load(full_path)
        meas= raw['spectral_data']
        
        # reorder measurements to match with reconstruction 
        meas = reorder(meas, Ord_acq, Ord_rec)
        
        #%% Reconstruct a single spectral slice from full reconstruction
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
            
        #%% Plot or save 
        # rotate
        #rec = np.rot90(rec,2)
        
        fig , axs = plt.subplots(1,1)
        im = axs.imshow(rec, cmap='gray')
        noaxis(axs)
        add_colorbar(im, 'bottom')
        
        full_path = save_root / (data_folder.name + '_' + f'{M}_{N_rec}' + '.pdf')
        fig.savefig(full_path, bbox_inches='tight')
        
        
        #%% pseudo inverse
        if M==4096:
            rec_pinv_gpu = model_pinv.reconstruct_expe(m)
            rec_pinv = rec_pinv_gpu.cpu().detach().numpy().squeeze()
        
            fig , axs = plt.subplots(1,1)
            im = axs.imshow(rec_pinv, cmap='gray')
            noaxis(axs)
            add_colorbar(im, 'bottom')
            
            full_path = save_root / (data_folder.name + '_' + f'pinv_{N_rec}' + '.pdf')
            fig.savefig(full_path, bbox_inches='tight', dpi=600)