# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:07:39 2023

@author: ducros
"""
from spyrit.core.train import remove_model_attributes, rename_model_attributes
import os


model_path_list = [
    'dc-net_unet_imagenet_rect_N0_10_N_64_M_256_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07.pth',
    'dc-net_unet_imagenet_rect_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07.pth',
    'dc-net_unet_imagenet_rect_N0_10_N_128_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07.pth',
    'dc-net_unet_stl10_rect_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07.pth',       # model_stl10
    'dc-net_unet_stl10_rect_N0_2_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07_seed_0.pth', # model_stl10/reprod
    'dc-net_unet_stl10_rect_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07_seed_0.pth',# model_stl10/reprod
    'dc-net_unet_stl10_rect_N0_50_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07_seed_0.pth' # model_stl10/reprod
    ]


for model_path in model_path_list: 

    remove_model_attributes(model_path, 'Acq.FO.', 'tmp.pth')
    print('--')
    
    remove_model_attributes('tmp.pth', 'DC_layer.')
    print('--')
    
    new_model_path = model_path[:-4] + '_light.pth'
    rename_model_attributes('tmp.pth', 'Denoi.', 'denoi.', new_model_path)
    
    os.remove('tmp.pth')