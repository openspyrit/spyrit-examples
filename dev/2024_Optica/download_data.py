# -*- coding: utf-8 -*-
#%%
import girder_client
from pathlib import Path

# api Rest url of the warehouse
url='https://pilot-warehouse.creatis.insa-lyon.fr/api/v1'

# Generate the warehouse client
gc = girder_client.GirderClient(apiUrl=url)

#%% Download the covariance matrix and mean image [Todo: add Ids from tomoradio-warehouse and remove unused files]
data_folder = './stat/'
dataId_list = ['63935b584d15dd536f04849f', # for reconstruction (imageNet, 128)
            '63935a224d15dd536f048490', # for reconstruction (imageNet, 128)
            '63935b624d15dd536f0484a5', # for reconstruction (imageNet, 64)
            '63935a224d15dd536f048496', # for reconstruction (imageNet, 64)
            ]

for dataId in dataId_list:
    myfile = gc.getFile(dataId)
    gc.downloadFile(dataId, data_folder + myfile['name'])

#%% Download the models [Todo: add Ids from tomoradio-warehouse]
data_folder = './model/'
dataId_list = ['6410889f0386da274769778f',
            '641088930386da2747697789',
            '6410889a0386da274769778c',
            '6410888d0386da2747697786',
            '644a38c985f48d3da07140ba', # N_rec = 64, M = 4095
            '644a38c785f48d3da07140b7', # N_rec = 64, M = 1024
            '644a38c585f48d3da07140b4', # N_rec = 64, M = 512
            ]

for dataId in dataId_list:
    myfile = gc.getFile(dataId)
    gc.downloadFile(dataId, data_folder + myfile['name'])

#%% Download the raw data [Todo: remove unused data]
data_folder = './data/'

folderId_list = [
                '637512e84d15dd536f048247', # bitten apple
                '622b65d543258e76eab217b3', # cat
                '622b5ea843258e76eab21740', # cat colored
                '639332784d15dd536f04838e', # color checker
                '622b634743258e76eab21767', # horse
                '616fe8ac478214d8c8a30d8e', # star sector x2
                '616fff63478214d8c8a30e2f', # star sector x12
                '61e19b3fcdb6910b899d0147', # star sector colored
                '622b57f743258e76eab21719', # star sector off-centered
                '6328942febe9129ae9936f8d', # Thorlab box
                '61b07f563f7ce79f5e565c46', # tomato x2
                '61b0924b3f7ce79f5e565c7a', # tomato x12
                '632b2adaebe9129ae9937250', # tree leaf
                '617289d2478214d8c8a30f5c', # usaf x2
                '6172a77c478214d8c8a30fde', # usaf x12
                ]

suffix_list = ['_spectraldata.npz', '_metadata.json']

for folderId in folderId_list:
    folder = gc.getFolder(folderId)
    folder_name = folder['name']

    list_item = gc.listItem(folderId)
    for item in list_item:
        if (suffix_list[0] in item['name']):
            item_id = item['_id']
            print('item found : ' + item['name'])
            gc.downloadItem(item_id, Path(data_folder) / Path(folder_name), item['name'])

        if (suffix_list[1] in item['name']):
            item_id = item['_id']
            print('item found : ' + item['name'])
            gc.downloadItem(item_id, Path(data_folder) / Path(folder_name), item['name'])
