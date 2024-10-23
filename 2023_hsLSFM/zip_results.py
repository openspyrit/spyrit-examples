# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:17:56 2023

@author: ducros
"""


from shutil import copytree, ignore_patterns, make_archive, rmtree
from pathlib import Path

#%% eGFP_DsRed
source = Path('./data/2023_03_13_2023_03_14_eGFP_DsRed_3D/')
destination = Path('./data/2023_03_13_2023_03_14_eGFP_DsRed_3D_recon/')

source_sub_list = [
    Path('Reconstruction/hypercube/pinv'),
    Path('Reconstruction/hypercube/pinv_shift'),
    Path('Reconstruction/hypercube/tikhonet50_div1.5'),
    Path('Reconstruction/hypercube/tikhonet50_div1.5_shift')
    ]
    

for source_sub in source_sub_list:
    copytree(source / source_sub, destination / source_sub, ignore=ignore_patterns('*.npy'))
    
make_archive(destination.parent / destination.name, 'zip', destination)

#%% DsRed mRFP
source = Path('./data/2023_02_28_mRFP_DsRed_3D/')
destination = Path('./data/2023_02_28_mRFP_DsRed_3D_recon/')

source_sub_list = [
    Path('Reconstruction/hypercube/pinv'),
    Path('Reconstruction/hypercube/pinv_shift'),
    Path('Reconstruction/hypercube/tikhonet50_div1.5'),
    Path('Reconstruction/hypercube/tikhonet50_div1.5_shift')
    ]
    

for source_sub in source_sub_list:
    copytree(source / source_sub, destination / source_sub, ignore=ignore_patterns('*.npy'))
    
make_archive(destination.parent / destination.name, 'zip', destination)
rmtree(destination)