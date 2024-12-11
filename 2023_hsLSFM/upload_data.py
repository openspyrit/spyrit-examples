# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:36:56 2024

@author: ducros
"""

from pathlib import Path 
import girder_client as gc

# api Rest url of the warehouse
url='https://pilot-warehouse.creatis.insa-lyon.fr/api/v1'


# Generate the warehouse client
gc = gc.GirderClient(apiUrl=url)

# Authentification
txt_file = open(Path('C:/Users/ducros/.apikey/pilot-warehouse.txt'), 'r', encoding='utf8')
apiKey = txt_file.read()
gc.authenticate(apiKey=apiKey)  # Authentication to the warehouse

#%% green-red fish

# Uncomment to create folder
#folderId = '63caaa937bef31845d991353' # data folder
#name = '2023_03_13_2023_03_14_eGFP_DsRed_3D'
#gc.createFolder(folderId, name)


folder = r'.\data\2023_03_13_2023_03_14_eGFP_DsRed_3D'
subfolderList = [#r'Preprocess',
                 #r'Reconstruction',
                 #r'Unmixing_shift',
                 r'Visualisation_shift'
                 ]

folderId = '6708d7990e9f151150f3c100'
for path in subfolderList:
    gc.upload(str(Path(folder)/Path(path)), folderId, reuseExisting=True)
    
#%% red-red zebrafish
folder = r'.\data\2023_02_28_mRFP_DsRed_3D'
subfolderList = [r'Preprocess',
                 #r'Reconstruction',
                 #r'Reconstruction/hypercube/tikhonet50_div1.5',
                 #r'Reconstruction/hypercube/tikhonet50_div1.5_shift',
                 #r'Unmixing_shift',
                 #r'Visualisation_shift'
                 ]

folderId = '66ff9c49ae27f5ad8259f38a'
for path in subfolderList:
    gc.upload(str(Path(folder)/Path(path)), folderId, reuseExisting=True)

#%% raw data
folder = r'F:/HSPIM_seb_acquisition/Post_doc/'

# chSPSIM/Raw_data/2023_03_13_2023_03_14_eGFP_DsRed_3D/Raw_data_chSPSIM_and_SPIM
subfolderList = [r'data_2023_03_13',
                 r'data_2023_03_14'
                 ]
folderId = '672e1f190e9f151150f4234b'
for path in subfolderList:
    gc.upload(str(Path(folder)/Path(path)), folderId, reuseExisting=True)    
    

# chSPSIM/Raw_data/2023_02_28_mRFP_DsRed_3D/Raw_data_chSPSIM_and_SPIM
subfolderList = [r'data_2023_02_28']
folderId = '672e1eb00e9f151150f4230d'
for path in subfolderList:
    gc.upload(str(Path(folder)/Path(path)), folderId, reuseExisting=True) 
