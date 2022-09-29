# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:14:39 2022

@author: ducros

NB (15-Sep-22): to debug needs to run

import collections
collections.Callable = collections.abc.Callable

"""

#%%
from pathlib import Path

#%% user-defined
img_size = 64
M_list = [4095]#, 2048, 1024, 512, 256]
N0_list = [10]#,10]


net_arch_list    = ['pinv-net', 'dc-net']
net_denoi_list   = ['cnn','unet'] #['unet', 'cnn']    # 'cnn' 'cnnbn' or 'unet'
net_data    = 'stl10'    # 'imagenet' or 'stl10'

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

#%% Loop over compression ratios, network architectures, and image domain denoisers
with open('caption.tex', 'w') as f:
    for data_folder, M, N0, net_arch, net_denoi in [
                                (data_folder, M, N0, net_arch, net_denoi) 
                                for data_folder in data_folder_list
                                for M in M_list 
                                for N0 in N0_list
                                for net_arch in net_arch_list 
                                for net_denoi in net_denoi_list]:   
    
        net_suffix  = f'N0_{N0}_N_64_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07'
        net_title = f'{net_arch}_{net_denoi}_{net_data}_{net_suffix}'
        
        full_path_slice = (data_folder.name + '_slice_' + net_title)
        full_path_bin = (data_folder.name + '_bin_' + net_title)
        print(full_path_bin)
        
        text = f'''
        \\begin{{center}}
            \includegraphics[width=\linewidth]{{{{{full_path_slice}}}.pdf}}
            \includegraphics[width=\linewidth]{{{{{full_path_bin}}}.pdf}}
            \captionof{{figure}}{{Reconstruction of height spectral channels (top) and spectral bins (bottom) using a Pinv-Net trained at $N0 = {N0}$ photons for $M = {M}$ measurements. The image domain denoiser is a {net_denoi}.}}
        \end{{center}}'''#.format(full_path_slice, full_path_bin)
        
        f.write(text)
        f.write('\n\n')