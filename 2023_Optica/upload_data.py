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
subfolderList = [r'Preprocess',
                 r'Reconstruction',
                 r'Unmixing_calib_blind_shift' ,
                 r'Visualisation_calib_blind_shift'
                 ]

folderId = '6708d7990e9f151150f3c100'
for path in subfolderList:
    gc.upload(str(Path(folder)/Path(path)), folderId)