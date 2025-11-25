# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 15:51:51 2025

@author: ducros
"""

import torch
from spyrit.misc.statistics import stat_imagenet
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
        
#%% Covariance of imageNet images
data_folder = '../../data/ILSVRC2012_v10102019'
save_folder = './stat/ILSVRC2012_v10102019'

for img_size in [32,64,128]:
    print(f'Image size: {img_size}')
    
    for get_size in ['original', 'resize','ccrop', 'rcrop']:
        print(f'Transform: {get_size}')
        
        stat_folder = Path(save_folder + '_' + get_size)
        stat_imagenet(stat_root = stat_folder, 
                      data_root = data_folder,
                      img_size = img_size, 
                      batch_size = 1024, 
                      get_size = get_size,
                      normalize = False,
                      device = device,
                      ext = 'pt')
