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
img_size = 128
M_list = [4096, 1024, 512]
sub_list = ['rect','var','var']
N0 = 10


net_arch  = 'dc-net'   # ['pinv-net', 'dc-net']
net_denoi = 'unet'     # ['unet', 'cnn']
net_data  = 'imagenet'   # 'imagenet' or 'stl10'

data_folder_list = [Path('usaf_x2'), 
                    Path('usaf_x12'),
                    Path('star_sector_x2'), 
                    Path('star_sector_x12'),
                    Path('star_sector_linear'),
                    Path('half_SiemensStar_with_ThorLabsName/'),
                    Path('tomato_slice_x2'),
                    Path('tomato_slice_x12'),
                    Path('cat'),
                    Path('cat_linear'),
                    Path('horse'),
                    Path('Bitten_Apple_t_5min-im_64x64_Zoom_x1_ti_10ms_tc_0.5ms/'),
                    Path('color_checker_full_FOV_64x64_Zoom_x1_ti_15ms_tc_0.2ms/'),
                    Path('green_Tree_leaf_ti_20ms_zoomx4_objx40/'),
                    Path('Thorlabs_box_ti_10ms_without_telecentric/'),
                    ]


sample_list = ['USAF ×2 zoom',
               'USAF ×12 zoom',
               'Star sector ×2 zoom', 
               'Star sector ×12 zoom',
               'Colored star sector',
               'Off centered star sector',
               'Tomato slice ×2 zoom',
               'Tomato slice ×12 zoom',
               'Cat',
               'Colored cat',
               'Horse',
               'Apple',
               'Color checker',
               'Leaf',
               'Thorlabs box',
               ]

too_long = ['Bitten_Apple_t_5min-im_64x64_Zoom_x1_ti_10ms_tc_0.5ms',
            'color_checker_full_FOV_64x64_Zoom_x1_ti_15ms_tc_0.2ms']

#%% Loop over compression ratios, network architectures, and image domain denoisers
with open('caption.tex', 'w') as f:
    
    # sample list
    f.write('We consider the following samples\n')
    f.write('\\begin{enumerate}\n')
    for data_folder, sample in zip(data_folder_list,sample_list):
        f.write(f'\\item {sample} (see Sec. \\ref{{sec:{data_folder.name}}}).\n')    
    f.write('\end{enumerate}')
    f.write('\n\n')
    
    f.write('\\newpage\n\n')
    
    for data_folder, sample in zip(data_folder_list,sample_list):
        
        if data_folder.name not in too_long:
            save_prefix = data_folder.name
        elif data_folder.name == too_long[0]:
            save_prefix = 'Bitten_Apple'
        elif data_folder.name == too_long[1]:
            save_prefix = 'color_checker'   
        
        text = f'''\subsection{{{sample}}}
        \label{{sec:{data_folder.name}}}\n'''
        f.write(text)
        f.write('\n')
        
        for M, sub in zip(M_list,sub_list):   
    
            net_suffix  = f'{sub}_N0_{N0}_N_128_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07'
            net_title = f'{net_arch}_{net_denoi}_{net_data}_{net_suffix}'
            
            full_path_slice = (save_prefix + '_slice_' + net_title)
            full_path_bin = (save_prefix + '_bin_' + net_title)
            print(full_path_bin)
            
            text = f'''
            \\begin{{center}}
                \includegraphics[width=\linewidth,trim={{0 2.7cm 0 2.5cm}},clip]{{{{{full_path_bin}}}.pdf}}
                \includegraphics[width=\linewidth,trim={{0 2.7cm 0 2.9cm}},clip]{{{{{full_path_slice}}}.pdf}}
                \captionof{{figure}}{{Reconstruction of four spectral bins (top) and channels (bottom) using a {net_arch} trained for $K = {M}$ measurements.}}
            \end{{center}}'''#.format(full_path_slice, full_path_bin)
            #\captionof{{figure}}{{Reconstruction of four spectral bins (top) and channels (bottom) using a {net_arch} trained at $\\alpha = {N0}$ photons for $M = {M}$ measurements. The image domain denoiser is a {net_denoi}.}}
            
            f.write(text)
            f.write('\n\n')
            
        f.write('            \\newpage')
        f.write('\n\n')