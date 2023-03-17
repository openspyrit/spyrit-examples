# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:07:39 2023

@author: ducros
"""
from spyrit.core.train import remove_model_attributes, rename_model_attributes
import os


model_path_list = [
    'dc-net_unet_imagenet_rect_N0_10_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07.pth',
    'dc-net_unet_imagenet_var_N0_10_N_128_M_2048_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07.pth',
    'dc-net_unet_imagenet_var_N0_10_N_128_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07.pth',
    'dc-net_unet_imagenet_var_N0_10_N_128_M_512_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07.pth',
    ]


for model_path in model_path_list: 

    remove_model_attributes(model_path, 'Acq.FO.', 'tmp.pth')
    print('--')
    
    remove_model_attributes('tmp.pth', 'DC_layer.')
    print('--')
    
    new_model_path = model_path[:-4] + '_light.pth'
    rename_model_attributes('tmp.pth', 'Denoi.', 'denoi.', new_model_path)
    
    os.remove('tmp.pth')