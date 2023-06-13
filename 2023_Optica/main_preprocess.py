# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:22:16 2023

@author: ducros
"""
#%%
data_folder = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D/'
meas_folder = '/Preprocess/'
meas_prefix = 'T1_RUN0002'
meas_suffix = '_2023_03_13_Had_512_128_128.npy'
#y_exp = np.load(Path(data_folder + meas_folder) / (meas_prefix + meas_suffix))


import numpy as np
from PIL import Image
import sys
sys.path.append('./fonction')
from load_data import Select_data
from matrix_tools import bining_colonne, bining_line
from pathlib import Path

def load_data_pos_neg(Dir, Run, l_start, l_end, l_bin, lambda_bin):
      
    Path_files, list_files = Select_data(Dir,Run)
    
    # get shapes
    Nh = len(list_files)//2
    Nl, Nc = np.rot90(np.array(Image.open(Path_files+list_files[0]))).shape
    Nl_bin =  l_end - l_start
    
    # Load raw data
    print((Nl,Nh,Nc))
    Data_pos = np.zeros((Nl,Nh,Nc))
    Data_neg = np.zeros((Nl,Nh,Nc))
    
    for i in range(0,2*Nh,2):      
        print(Path_files+list_files[i])
        print(Path_files+list_files[i+1])       
        Data_pos[:,i//2] = np.float_(np.rot90(np.array(Image.open(Path_files+list_files[i]))))
        Data_neg[:,i//2] = np.float_(np.rot90(np.array(Image.open(Path_files+list_files[i+1]))))
    
    # Crop raw data
    # We only have 2048 lines on the imaging camera we remove 56 lines 
    # at the top and the bottom of the spectrale images
    Data_pos = Data_pos[l_start:l_end,:]
    Data_neg = Data_neg[l_start:l_end,:]
    
    # init output
    Nl_bin = Nl_bin // l_bin 
    Nc_bin = Nc // lambda_bin
    stack_pos = np.zeros((Nl_bin,Nh,Nc_bin))
    stack_neg = np.zeros((Nl_bin,Nh,Nc_bin)) 
    
    # Spectral binning AND spatial binning across lines
    for i in range(Nh):
        tmp = bining_colonne(Data_pos[:,i,:], lambda_bin) 
        stack_pos[:,i] = bining_line(tmp, l_bin)
        #
        tmp = bining_colonne(Data_neg[:,i,:], lambda_bin) 
        stack_neg[:,i] = bining_line(tmp, l_bin)
    
    return stack_pos, stack_neg

#%% DATA NOT IN THE WAREHOUSE (hard drive only)!
#Dir = '../../data/2023_02_28_mRFP_DsRed_3D/Raw_data_chSPSIM_and_SPIM/data_2023_02_28/'
Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/data_2023_03_13/'
Run = 'RUN0006' 
save_folder = '/Preprocess/'

# Binning is chosen such that:
# 56 - 2104 = 2048 rows, hence 512 rows after x4 binning
# 20 = 128 spectral channels 
stack_pos, stack_neg = load_data_pos_neg(Dir,Run,56,2104,4,20)

# Save
Nl, Nh, Nc = stack_pos.shape
Ns = int(Run[-1])-1
filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_pos.npy'
np.save(Path(data_folder+save_folder) / filename, stack_pos)
filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_neg.npy'
np.save(Path(data_folder+save_folder) / filename, stack_neg)

#%% Check the prep data (pos neg should math with seb's prep)
Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/data_2023_03_14/'
Run = 'RUN0004' 
Ns = int(Run[-1])-1
save_folder = '/Preprocess/'
Nl, Nh, Nc = 512, 128, 128

filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}.npy'
prep = np.load(Path(data_folder+save_folder) / filename)

filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_pos.npy'
prep_pos = np.load(Path(data_folder+save_folder) / filename)

filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_neg.npy'
prep_neg =  np.load(Path(data_folder+save_folder) / filename)

print(f'error: {np.linalg.norm(prep_pos-prep_neg-prep)/np.linalg.norm(prep)}')

#%% DATA NOT IN THE WAREHOUSE (hard drive only)!
Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/data_2023_03_14/'
Run = 'RUN0005' 
save_folder = '/Preprocess/'

# Binning is chosen such that:
# 56 - 2104 = 2048 rows, hence 512 rows after x4 binning
# 20 = 128 spectral channels 
stack_pos, stack_neg = load_data_pos_neg(Dir,Run,56,2104,4,20)

# Save
Nl, Nh, Nc = stack_pos.shape
Ns = int(Run[-1])+5
filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_pos.npy'
np.save(Path(data_folder+save_folder) / filename, stack_pos)
filename = f'T{Ns}_{Run}_2023_03_13_Had_{Nl}_{Nh}_{Nc}_neg.npy'
np.save(Path(data_folder+save_folder) / filename, stack_neg)