# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 11:16:59 2023

@author: ducros
"""

#%% 10 photons, Hadam1Spit
from spyrit.core.train import read_param, Train_par
from pathlib import Path
import matplotlib.pyplot as plt

save_tag = False
model_folder = './model/'
suffix = 'pinv-net_unet_imagenet_ph_10_Hadam1_N_512_M_128_epo_30_lr_0.001_sss_10_sdr_0.5_bs_20_reg_1e-07'
     
train_path = Path(model_folder) / ('TRAIN_' + suffix + '.pkl')
train_param = read_param(train_path)

fig_folder = './figure/'
train_param.plot_log(start=1)
if save_tag:
    save_path = Path(model_folder) / ('TRAIN_' + suffix + '.png')
    plt.savefig(Path(save_path), bbox_inches='tight', dpi=600)
    
#%% 10 photons, Experimental patterns
from spyrit.core.train import read_param, Train_par
from pathlib import Path
import matplotlib.pyplot as plt

save_tag = True
model_folder = './model/'
suffix = 'pinv-net_unet_imagenet_ph_10_exp_N_512_M_128_epo_30_lr_0.001_sss_10_sdr_0.5_bs_20_reg_1e-07'
     
train_path = Path(model_folder) / ('TRAIN_' + suffix + '.pkl')
train_param = read_param(train_path)

fig_folder = './figure/'
train_param.plot_log(start=1)
if save_tag:
    save_path = Path(model_folder) / ('TRAIN_' + suffix + '.png')
    plt.savefig(Path(save_path), bbox_inches='tight', dpi=600)