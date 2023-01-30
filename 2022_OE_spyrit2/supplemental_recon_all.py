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

from spyrit.core.Acquisition import Acquisition_Poisson_approx_Gauss
from spyrit.core.Forward_Operator import Forward_operator_Split_ft_had
from spyrit.core.Preprocess import Preprocess_Split_diag_poisson
from spyrit.core.Data_Consistency import Generalized_Orthogonal_Tikhonov
from spyrit.core.training import load_net
from spyrit.core.neural_network import Unet
from spyrit.core.reconstruction import DC2_Net

from spyrit.misc.statistics import Cov2Var
from spyrit.misc.sampling import Permutation_Matrix 
import spyrit.misc.walsh_hadamard as wh

from spas import read_metadata, plot_color, spectral_binning, spectral_slicing

from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Torch device: {device}')

# get debug in spyder
import collections
collections.Callable = collections.abc.Callable

#%% user-defined
# for reconstruction
N_rec = 128  
M_list = [4096, 1024, 512] # [4095, 1024, 512]   for N_rec = 64 
N0 = 10     # Check if we used 10 in the paper
stat_folder_rec = Path('../../stat/ILSVRC2012_v10102019/') # works for for N = 64 only !!

# used for acquisition
N_acq = 64
stat_folder_acq =  Path('./data_online/') 

net_arch    = 'dc-net'      # ['dc-net','pinv-net']
net_denoi   = 'unet'        # ['unet', 'cnn']
net_data    = 'imagenet'    # 'imagenet' 
save_root = Path('./recon_128/')

#%% covariance matrix and network filnames
stat_folder_rec = Path('../../stat/ILSVRC2012_v10102019/')
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

    net_suffix  = f'N0_{N0}_N_{N_rec}_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_{bs}_reg_1e-07'
    net_folder= f'{net_arch}_{net_denoi}_{net_data}/'
    
    #%% Init and load trained network
    H =  wh.walsh2_matrix(N_rec)
    
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
    Perm_rec = Permutation_Matrix(Ord_rec)
    Hperm = Perm_rec @ H
    Pmat = Hperm[:M,:]
    
    #    
    Forward = Forward_operator_Split_ft_had(Pmat, Perm_rec, N_rec, N_rec)
    Noise = Acquisition_Poisson_approx_Gauss(N0, Forward)
    Prep = Preprocess_Split_diag_poisson(N0, M, N_rec**2)
    
    Denoi = Unet()
    Cov_perm = Perm_rec @ Cov_rec @ Perm_rec.T
    DC = Generalized_Orthogonal_Tikhonov(sigma_prior = Cov_perm, 
                                         M = M, N = N_rec**2)
    model = DC2_Net(Noise, Prep, DC, Denoi)
    
    # Load trained DC-Net
    net_title = f'{net_arch}_{net_denoi}_{net_data}_{net_order}_{net_suffix}'
    title = './model_v2/' + net_folder + net_title
    load_net(title, model, device)
    model.eval()                    # Mandantory when batchNorm is used  
    
    #%% List expe data
    data_root = Path('data_online/')
    data_folder_list = [Path('Bitten_Apple_t_5min-im_64x64_Zoom_x1_ti_10ms_tc_0.5ms/'),
                        # Path('color_checker_full_FOV_64x64_Zoom_x1_ti_15ms_tc_0.2ms/'),
                        # Path('green_Tree_leaf_ti_20ms_zoomx4_objx40/'),
                        # Path('half_SiemensStar_with_ThorLabsName/'),
                        # Path('Thorlabs_box_ti_10ms_without_telecentric/'),
                        # Path('usaf_x12/'), 
                        # Path('usaf_x2/'), 
                        # Path('star_sector_x2/'), 
                        # Path('star_sector_x12/'),
                        # Path('star_sector_linear'),
                        # Path('tomato_slice_x2/'),
                        # Path('tomato_slice_x12'),
                        # Path('cat/'),
                        # Path('cat_linear/'),
                        Path('horse/')
                        ]
     
    data_file_prefix_list = ['Bitten_Apple_t_5min-im_64x64_Zoom_x1_ti_10ms_tc_0.5ms',
                             # 'color_checker_full_FOV_64x64_Zoom_x1_ti_15ms_tc_0.2ms',
                             # 'green_Tree_leaf_ti_20ms_zoomx4_objx40',
                             # 'half_SiemensStar_with_ThorLabsName',
                             # 'Thorlabs_box_ti_10ms_without_telecentric',
                             # 'zoom_x12_usaf_group5',
                             # 'zoom_x2_usaf_group2',
                             # 'zoom_x2_starsector',
                             # 'zoom_x12_starsector',
                             # 'SeimensStar_whiteLamp_linear_color_filter',
                             # 'tomato_slice_2_zoomx2',
                             # 'tomato_slice_2_zoomx12',
                             # 'Cat_whiteLamp',
                             # 'Cat_LinearColoredFilter',
                             'Horse_whiteLamp'
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
        
        # Order used for acquisistion
        Perm_acq = Permutation_Matrix(Ord_acq)
        
        # zero filling when reconstrcution res is higher than acquisition res
        if N_rec > N_acq:
            
            # Natural order measurements (N_acq resolution)
            Perm_raw = np.zeros((2*N_acq**2,2*N_acq**2))
            Perm_raw[::2,::2] = Perm_acq.T     
            Perm_raw[1::2,1::2] = Perm_acq.T
            meas = Perm_raw @ meas
            
            # Square subsampling in the "natural" order
            Ord_sub = np.zeros((N_rec,N_rec))
            Ord_sub[:N_acq,:N_acq]= -np.arange(-N_acq**2,0).reshape(N_acq,N_acq)
            Perm_sub = Permutation_Matrix(Ord_sub) 
            
            # zero filled measurement (N_res resolution)
            zero_filled = np.zeros((2*N_rec**2,len(wavelengths)))
            zero_filled[:2*N_acq**2,:] = meas
            
            meas = zero_filled
            
            Perm_raw = np.zeros((2*N_rec**2,2*N_rec**2))
            Perm_raw[::2,::2] = Perm_sub.T     
            Perm_raw[1::2,1::2] = Perm_sub.T
            
            meas = Perm_raw @ meas
        
        # Reorder measurements  
        if N_rec == N_acq:
            # To reorder measurements
            Perm_sub = Perm_acq[:N_rec**2,:].T
        
        if N_rec <= N_acq:   
            # Get both positive and negative coefficients permutated
            Perm = Perm_rec @ Perm_sub
            Perm_raw = np.zeros((2*N_rec**2,2*N_acq**2))
            Perm_raw[::2,::2] = Perm     
            Perm_raw[1::2,1::2] = Perm
            meas = Perm_raw @ meas
            
        elif N_rec > N_acq:
            Perm = Perm_rec
            Perm_raw = np.zeros((2*N_rec**2,2*N_rec**2))
            Perm_raw[::2,::2] = Perm     
            Perm_raw[1::2,1::2] = Perm
            meas = Perm_raw @ meas
        
        #%% Reconstruct all spectral channels
        # init
        n_batch = 32 # a power of two
        n_wav = wavelengths.shape[0]//n_batch
        rec_net = np.zeros((wavelengths.shape[0],N_rec, N_rec))
        
        # Net
        model.to(device)
        model.PreP.set_expe()
        
        with torch.no_grad():
            for b in range(n_batch):       
    
                ind = range(b*n_wav, (b+1)*n_wav)            
                m = torch.Tensor(meas[:2*M,ind].T).to(device)
                rec_net_gpu = model.reconstruct_expe(m)
                rec_net[ind,:,:] = rec_net_gpu.cpu().detach().numpy().squeeze()
        
        # Save
        if save_root:
            save_root.mkdir(parents=True, exist_ok=True)
            full_path = save_root / (data_folder.name + '_slice_' + net_title + '.npy')
            np.save(full_path, rec_net)
        
        #%% Pick-up a few spectral slice from full reconstruction
        wav_min = 530 
        wav_max = 730
        wav_num = 4
        
        too_long = ['Bitten_Apple_t_5min-im_64x64_Zoom_x1_ti_10ms_tc_0.5ms',
                    'color_checker_full_FOV_64x64_Zoom_x1_ti_15ms_tc_0.2ms']
        
        rec_net = rec_net.reshape((wavelengths.shape[0],-1))
        rec_slice, wavelengths_slice, _ = spectral_slicing(
            rec_net, wavelengths, wav_min, wav_max, wav_num)
        rec_slice = rec_slice.reshape((wavelengths_slice.shape[0],N_rec,N_rec))
        
        # rotate 180°
        if data_folder.name not in ['usaf_x2','usaf_x12']:
            rec_slice = np.rot90(rec_slice,2,(1,2))
        
        # either save
        if save_root is not False: 
            # filenames
            if data_folder.name not in too_long:
                save_prefix = data_folder.name
            elif data_folder.name == too_long[0]:
                save_prefix = 'Bitten_Apple'
            elif data_folder.name == too_long[1]:
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
        if data_folder.name not in ['usaf_x2','usaf_x12']:
            rec_net = np.rot90(rec_net,2,(1,2))
        
        # either save
        if save_root is not False:
            
            # filenames
            if data_folder.name not in too_long:
                save_prefix = data_folder.name
            elif data_folder.name == too_long[0]:
                save_prefix = 'Bitten_Apple'
            elif data_folder.name == too_long[1]:
                save_prefix = 'color_checker'        
            full_path = save_root / (save_prefix + '_bin_' + net_title + '.pdf')
            
            # plot, save, and close
            plot_color(rec_net, wavelengths_bin, fontsize=7, filename=full_path)
            plt.close()
       
        # or plot    
        else:   
            plot_color(rec_net, wavelengths_bin, fontsize=7)
