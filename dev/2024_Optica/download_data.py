# %%
# IMPORTS

import girder_client
from pathlib import Path

from spyrit.misc.load_data import download_girder


# %%
# DOWNLOADS

# api Rest url of the warehouse
url='https://pilot-warehouse.creatis.insa-lyon.fr/api/v1'

# Download the covariance matrix and mean image
# [Todo: add Ids from tomoradio-warehouse and remove unused files]

stat_folder = './stat/'
statId_list = ['63935b584d15dd536f04849f', # for reconstruction (imageNet, 128)
            '63935a224d15dd536f048490', # for reconstruction (imageNet, 128)
            '63935b624d15dd536f0484a5', # for reconstruction (imageNet, 64)
            '63935a224d15dd536f048496', # for reconstruction (imageNet, 64)
            ]
download_girder(url, statId_list, stat_folder)

# Download the models

model_folder = './model/'
modelId_list = ['6410889f0386da274769778f',
            '641088930386da2747697789',
            '6410889a0386da274769778c',
            '6410888d0386da2747697786',
            '644a38c985f48d3da07140ba', # N_rec = 64, M = 4095
            '644a38c785f48d3da07140b7', # N_rec = 64, M = 1024
            '644a38c585f48d3da07140b4', # N_rec = 64, M = 512
            ]
download_girder(url, modelId_list, model_folder)

# Download the raw data
# [Todo: remove unused data]

data_folder = './data/'
folderId_list = [
                # '637512e84d15dd536f048247', # bitten apple
                # '622b65d543258e76eab217b3', # cat
                # '622b5ea843258e76eab21740', # cat colored
                # '639332784d15dd536f04838e', # color checker
                # '622b634743258e76eab21767', # horse
                '616fe8ac478214d8c8a30d8e', # star sector x2
                # '616fff63478214d8c8a30e2f', # star sector x12
                # '61e19b3fcdb6910b899d0147', # star sector colored
                # '622b57f743258e76eab21719', # star sector off-centered
                # '6328942febe9129ae9936f8d', # Thorlab box
                '61b07f563f7ce79f5e565c46', # tomato x2
                '61b0924b3f7ce79f5e565c7a', # tomato x12
                # '632b2adaebe9129ae9937250', # tree leaf
                # '617289d2478214d8c8a30f5c', # usaf x2
                '6172a77c478214d8c8a30fde', # usaf x12
                ]
# in the folder list, download only the files with the following suffixes
suffix = ('_spectraldata.npz', '_metadata.json')

# Generate the warehouse client
gc = girder_client.GirderClient(apiUrl=url)

for folderId in folderId_list:
    for item in gc.listItem(folderId):
        for file in gc.listFile(item['_id']):
            
            if file['name'].endswith(suffix):
                folder_name = gc.getFolder(folderId)['name']
                download_girder(url, file['_id'],
                                Path(data_folder) / Path(folder_name))

# %%
