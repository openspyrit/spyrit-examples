# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 09:35:20 2022

@author: ducros



NB (15-Sep-22): to debug needs to run

import collections
collections.Callable = collections.abc.Callable

"""

#%%
import torch
import numpy as np
import spyrit.misc.walsh_hadamard as wh

from matplotlib import pyplot as plt

from spyrit.learning.model_Had_DCAN import *
from spyrit.misc.disp import torch2numpy, imagesc, plot

from spyrit.learning.nets import *
from spyrit.restructured.Updated_Had_Dcan import *
from spyrit.misc.metrics import psnr_
from spyrit.misc.disp import imagesc, add_colorbar, noaxis

from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Torch device: {device}')

# get debug in spyder
import collections
collections.Callable = collections.abc.Callable

#%% user-defined
img_size = 64
M_list = [2048]#[4095, 2048, 1024, 512, 256]  
N0 = 10 # Check if we used 10 in the paper 
stat_folder = Path('data_online/') 
average_file= stat_folder / ('Average_{}x{}'.format(img_size,img_size)+'.npy')
cov_file    = stat_folder / ('Cov_{}x{}'.format(img_size,img_size)+'.npy')

net_arch_list    = ['dc-net','pinv-net']
net_denoi_list   = ['cnn'] #['cnn', 'unet'] #['unet', 'cnn']    # 'cnn' 'cnnbn' or 'unet'
net_data    = 'stl10'    # 'imagenet' or 'stl10'
save_root = Path('recon_all_expe_pdf') # False or Path('some_folder') 

#%% Loop over compression ratios, network architectures, and image domain denoisers
for M, net_arch, net_denoi in [
                            (M,net_arch,net_denoi) for M in M_list 
                            for net_arch in net_arch_list 
                            for net_denoi in net_denoi_list]:   
#for M in M_list:
    net_suffix  = f'N0_{N0}_N_64_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07'

    net_folder= f'{net_arch}_{net_denoi}_{net_data}/'
    net_title = f'{net_arch}_{net_denoi}_{net_data}_{net_suffix}'
    title = './model_v2/' + net_folder + net_title
    
    #%% Init and load trained network
    Mean = np.load(average_file)
    Cov  = np.load(cov_file)
    
    H =  wh.walsh2_matrix(img_size)
    Ord = Cov2Var(Cov)
    Perm = Permutation_Matrix(Ord)
    Hperm = Perm@H
    Pmat = Hperm[:M,:]
    
    Cov_perm = Perm @ Cov @ Perm.T
    
    Forward = Split_Forward_operator_ft_had(Pmat, Perm, img_size, img_size)
    Noise = Bruit_Poisson_approx_Gauss(N0, Forward)
    Prep = Split_diag_poisson_preprocess(N0, M, img_size**2)
    
    # Image-domain denoising layer
    if net_denoi == 'cnn':      # CNN no batch normalization
        Denoi = ConvNet()
    elif net_denoi == 'cnnbn':  # CNN with batch normalization
        Denoi = ConvNetBN()
    elif net_denoi == 'unet':   # Unet
        Denoi = Unet()
    
    # Global Architecture
    if net_arch == 'dc-net':        # Denoised Completion Network
        Cov_perm = Perm @ Cov @ Perm.T
        DC = Generalized_Orthogonal_Tikhonov(sigma_prior = Cov_perm, 
                                             M = M, 
                                             N = img_size**2)
        model = DC2_Net(Noise, Prep, DC, Denoi)
        
    elif net_arch == 'pinv-net':    # Pseudo Inverse Network
        DC = Pinv_orthogonal()
        model = Pinv_Net(Noise, Prep, DC, Denoi)
    
    # Load trained DC-Net
    load_net(title, model, device)
    model.eval()                    # Mandantory when batchNorm is used
    
    #%% Load expe data and unsplit
    data_root = Path('data_online/')
    
    data_folder_list = [Path('usaf_x12/'), 
                        Path('usaf_x2/'), 
                        Path('star_sector_x2/'), 
                        Path('star_sector_x12/'),
                        Path('star_sector_linear'),
                        Path('tomato_slice_x2/'),
                        Path('tomato_slice_x12'),
                        Path('cat/'),
                        Path('cat_linear/'),
                        Path('horse/')]
    
    data_file_prefix_list = ['zoom_x12_usaf_group5',
                             'zoom_x2_usaf_group2',
                             'zoom_x2_starsector',
                             'zoom_x12_starsector',
                             'SeimensStar_whiteLamp_linear_color_filter',
                             'tomato_slice_2_zoomx2',
                             'tomato_slice_2_zoomx12',
                             'Cat_whiteLamp',
                             'Cat_LinearColoredFilter',
                             'Horse_whiteLamp']
    
    #%% LOOP OVER DATASETS
    for data_folder,data_file_prefix in zip(data_folder_list,data_file_prefix_list):
        print(data_folder / data_file_prefix)
        print('-')
        
        full_path = data_root / data_folder / (data_file_prefix + '_spectraldata.npz')
        raw = np.load(full_path)
        meas= raw['spectral_data']
    
        #%% 
        from spas import read_metadata, plot_color, spectral_binning, spectral_slicing
        meta_path = data_root / data_folder / (data_file_prefix + '_metadata.json')
        
        _, acquisition_parameters, _, _ = read_metadata(meta_path)
        wavelengths = acquisition_parameters.wavelengths
        
        # for info: wavelengths[500] = 579.097; wavelengths[1000] = 639.3931; wavelengths[1500] = 695.1232
            
        #%% Reconstruct all spectral channels
        # init
        n_batch = 8 # a power of two
        n_wav = wavelengths.shape[0]//n_batch
        rec_net = np.zeros((wavelengths.shape[0],img_size, img_size))
        
        # Net
        #model.to(device)
        model.PreP.set_expe(0.77,739,17,1)
        #model.PreP.set_expe()
        
        with torch.no_grad():
            for b in range(n_batch):       
    
                ind = range(b*n_wav, (b+1)*n_wav)            
                m = torch.Tensor(meas[:2*M,ind].T).to(device)
                rec_net_gpu = model.reconstruct_expe(m)
                rec_net[ind,:,:] = rec_net_gpu.cpu().detach().numpy().squeeze()
        
        # Save
        if save_root:
            save_root.mkdir(parents=True, exist_ok=True)
            #full_path = save_root / (data_folder.name + '_slice_' + net_title + '.png')
            #np.save(full_path, rec_net)
            
        
        #%% Pick-up a few spectral slice from full reconstruction
        wav_min = 530 
        wav_max = 730
        wav_num = 8
        
        rec_net = rec_net.reshape((wavelengths.shape[0],-1))
        rec_slice, wavelengths_slice, _ = spectral_slicing(
            rec_net, wavelengths, wav_min, wav_max, wav_num)
        rec_slice = rec_slice.reshape((wavelengths_slice.shape[0],img_size,img_size))
        
        # rotate
        rec_slice = np.rot90(rec_slice,2,(1,2))
        
        # either save
        if save_root is not False: 
            full_path = save_root / (data_folder.name + '_slice_' + net_title + '.pdf')
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
        # rotate
        rec_net = np.rot90(rec_net,2,(1,2))
        # either save
        if save_root is not False:
            full_path = save_root / (data_folder.name + '_bin_' + net_title + '.pdf')
            plot_color(rec_net, wavelengths_bin, fontsize=7, filename=full_path)
            plt.close()
        # or plot    
        else:   
            plot_color(rec_net, wavelengths_bin, fontsize=7)