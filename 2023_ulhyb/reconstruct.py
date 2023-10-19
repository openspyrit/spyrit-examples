# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:41:19 2022

This scripts generated the figures in the Appendix of the paper

NB (15-Sep-22): to debug needs to run
import collections
collections.Callable = collections.abc.Callable

"""

#%%
import torch
import numpy as np
import math

from matplotlib import pyplot as plt

from spyrit.misc.statistics import Cov2Var
from spyrit.core.noise import Poisson 
from spyrit.core.meas import HadamSplit
from spyrit.core.prep import SplitPoisson
from spyrit.core.recon import DCNet
from spyrit.core.train import load_net
from spyrit.core.nnet import Unet
from spyrit.misc.sampling import reorder, Permutation_Matrix

from spas import read_metadata, plot_color, spectral_binning, spectral_slicing

from pathlib import Path

import torch.nn as nn # delete after tests

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Torch device: {device}')

# get debug in spyder
import collections
collections.Callable = collections.abc.Callable

#%% user-defined
# used for acquisition
N_acq = 32

# for reconstruction
N_rec = 64  
M_list = [1024] #[4096, 1024, 512]

N0 = 50     # Check if we used 10 in the paper
net_arch    = 'dc-net'      # ['dc-net','pinv-net']
net_denoi   = 'unet'        # ['unet', 'cnn']

# net trained using stl10
net_data    = 'stl10'    # 'imagenet' 'stl10'
net_folder = './model_stl10/reprod/'
stat_folder_rec = Path('./stat_stl10/')
cov_rec_file = stat_folder_rec/ ('Cov_{}x{}'.format(N_rec, N_rec)+'.npy') 
bs = 1024

# net trained using imagenet
# net_data    = 'imagenet'    # 'imagenet' 'stl10'
# net_folder = './model/'
# stat_folder_rec = Path('./stat/')
# cov_rec_file = stat_folder_rec/ ('Cov_8_{}x{}'.format(N_rec, N_rec)+'.npy') 
# bs = 256

save_root = Path('./recon/')

#%% Networks
for M in M_list:
    
    if (N_rec == 128) and (M == 4096):
        net_order   = 'rect'
    else:
        net_order   = 'var'
    
    net_order   = 'rect'
    net_suffix  = f'N0_{N0}_N_{N_rec}_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_{bs}_reg_1e-07_light'
    
    #net_order   = 'rect'
    #net_suffix  = f'N0_{N0}_N_{N_rec}_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_{bs}_reg_1e-07_light'
    
    #%% Init and load trained network   
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
    meas = HadamSplit(M, N_rec, Ord_rec)
    noise = Poisson(meas, N0) # could be replaced by anything here as we just need to recon
    prep  = SplitPoisson(N0, meas)
    
    # No postprocessing
    # net_title = f'N0_{N0}_N_{N_rec}_M_{M}_tikho_{stat_folder_rec.name}_no_net'
    # model = DCNet(noise, prep, Cov_rec)
    
    # Unet post-processing
    denoi = Unet()
    model = DCNet(noise, prep, Cov_rec, denoi)
    net_title = f'{net_arch}_{net_denoi}_{net_data}_{net_order}_{net_suffix}'
    title = net_folder + net_title
    load_net(title, model, device, True)
    model.eval()                    # Mandantory when batchNorm is used  
    
    # bypass the convolutional layers
    #model.denoi = nn.Identity()
    
    #%% List expe data
    data_root = Path('data/')
    data_file_prefix_list = ['test',
                             ]
    
    #%% Load data
    for data_file_prefix in data_file_prefix_list:
        
        print(data_file_prefix)
        
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
        
        #%% Reconstruct all spectral channels
        # init
        n_batch = 32 # a power of two
        n_wav = wavelengths.shape[0]//n_batch
        rec_net = np.zeros((wavelengths.shape[0],N_rec, N_rec))
        
        # Net
        model.prep.set_expe()
        model.to(device)
        
        with torch.no_grad():
            for b in range(n_batch):       
    
                ind = range(b*n_wav, (b+1)*n_wav)            
                m = torch.Tensor(meas[:2*M,ind].T).to(device)
                rec_net_gpu = model.reconstruct_expe(m)
                rec_net[ind,:,:] = rec_net_gpu.cpu().detach().numpy().squeeze()
        
        too_long = ['Bitten_Apple_t_5min-im_64x64_Zoom_x1_ti_10ms_tc_0.5ms',
                    'color_checker_full_FOV_64x64_Zoom_x1_ti_15ms_tc_0.2ms']
        
        # Save
        if save_root:
            save_root.mkdir(parents=True, exist_ok=True)
            
            if data_file_prefix not in too_long:
                save_prefix = data_file_prefix
            elif data_file_prefix == too_long[0]:
                save_prefix = 'Bitten_Apple'
            elif data_file_prefix == too_long[1]:
                save_prefix = 'color_checker'
            full_path = save_root / (save_prefix + '_all_' + net_title + '.npy')
            np.save(full_path, rec_net)
        
        #%% Pick-up a few spectral slice from full reconstruction
        wav_min = 530 
        wav_max = 730
        wav_num = 4
        
        rec_net = rec_net.reshape((wavelengths.shape[0],-1))
        rec_slice, wavelengths_slice, _ = spectral_slicing(
            rec_net, wavelengths, wav_min, wav_max, wav_num)
        rec_slice = rec_slice.reshape((wavelengths_slice.shape[0],N_rec,N_rec))
        
        # rotate 180°
        if data_file_prefix not in ['zoom_x2_usaf_group2','zoom_x12_usaf_group5']:
            rec_slice = np.rot90(rec_slice,2,(1,2))
        
        # either save
        if save_root is not False: 
            # filenames
            if data_file_prefix not in too_long:
                save_prefix = data_file_prefix
            elif data_file_prefix == too_long[0]:
                save_prefix = 'Bitten_Apple'
            elif data_file_prefix == too_long[1]:
                save_prefix = 'color_checker'  
            full_path = save_root / (save_prefix + '_slice_' + net_title + '.pdf')
             
            # plot, save, and close
            plot_color(rec_slice, wavelengths_slice, fontsize=7, filename=full_path)
            plt.close()
        # or plot    
        else:
            plot_color(rec_slice, wavelengths_slice, fontsize=7)
            
        #%% Reconstrcution of spectrally-binned measurements
        meas_bin, wavelengths_bin, _ = spectral_binning(
            meas.T, wavelengths, wav_min, wav_max, wav_num)
        
        with torch.no_grad():
            m = torch.Tensor(meas_bin[:2*M,:]).to(device)
            rec_net_gpu = model.reconstruct_expe(m)
            rec_net = rec_net_gpu.cpu().detach().numpy().squeeze()
            
        #%% Plot or save 
        # rotate 180°
        if data_file_prefix not in ['zoom_x12_usaf_group5','zoom_x12_usaf_group5']:
            rec_net = np.rot90(rec_net,2,(1,2))
        
        # either save
        if save_root is not False:
            
            # filenames
            if data_file_prefix not in too_long:
                save_prefix = data_file_prefix
            elif data_file_prefix == too_long[0]:
                save_prefix = 'Bitten_Apple'
            elif data_file_prefix == too_long[1]:
                save_prefix = 'color_checker'        
            full_path = save_root / (save_prefix + '_bin_' + net_title + '.pdf')
            
            # plot, save, and close
            plot_color(rec_net, wavelengths_bin, fontsize=7, filename=full_path)
            plt.close()
       
        # or plot    
        else:   
            plot_color(rec_net, wavelengths_bin, fontsize=7)