# %%
# IMPORTS

from pathlib import Path
from spyrit.misc.load_data import download_girder

# %%
# %% SETTINGS
# change this if you want to download the files in a different folder
destination = Path.cwd() # / Path("your_subfolder")
print("Copying folder in:", destination)

# %%
# download data from the Pilot warehouse
url_pilot = 'https://pilot-warehouse.creatis.insa-lyon.fr/api/v1'
data_subfolder = Path("data")
data_files = [
    "61b07f593f7ce79f5e565c5f", # tomato x2 spectraldata.npz
    "61b07f583f7ce79f5e565c5c", # tomato x2 metadata.json
    "61b0924e3f7ce79f5e565c93", # tomato x12 spectraldata.npz
    "61b0924d3f7ce79f5e565c90", # tomato x12 metadata.json
    "616fff65478214d8c8a30e45", # starsector x12 spectraldata.npz
    "616fff64478214d8c8a30e42", # starsector x12 metadata.json
    "6172a77e478214d8c8a30ff7", # usaf x12 spectraldata.npz
    "6172a77d478214d8c8a30ff4", # usaf x12 metadata.json   
]
download_girder(url_pilot, data_files, data_subfolder)

# %%
# download images for simulation from the Tomoradio warehouse
url_tomoradio = 'https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1'
images_subfolder = data_subfolder / Path("images/cropped")
image_files = [
    "66cdd369b891f94a08ad81eb", # ILSVRC2012_val_00000003_crop.JPEG
    "66cdd36ab891f94a08ad81ee", # ILSVRC2012_val_00000012_crop.JPEG
    "66cdd36ab891f94a08ad81f1", # ILSVRC2012_val_00000019_crop.JPEG
    "66cdd36ab891f94a08ad81f4", # ILSVRC2012_val_00000051_crop.JPEG
    "66cdd36ab891f94a08ad81f7", # ILSVRC2012_val_00000056_crop.JPEG
]
download_girder(url_tomoradio, image_files, images_subfolder)


# %%
# download models from the Tomoradio warehouse
model_subfolder = Path("model")
model_files = [
    "66cf34f5b891f94a08ad8212", # Pinv-net
    "66cf34f5b891f94a08ad8218", # LPGD
    "66c72804b891f94a08ad8193", # DC-Net
    "66cc9a7bb891f94a08ad81e4", # Pinv - PnP
    "66d6cd3eb891f94a08ad8221", # dfb-net nch_1 / varnoise0.05
    "66d6cd3eb891f94a08ad8224", # dfb-net nch_1 / varnoise0.1
    "66cf34f5b891f94a08ad8215", # Pinv-net retrained
]
download_girder(url_tomoradio, model_files, model_subfolder)
    

# %%
# download statistics from the Tomoradio warehouse
stat_subfolder = Path("stat")
stat_files = [
    "66c8602db891f94a08ad81ae", # 64x64
    "66c8602bb891f94a08ad81ab", # 128x128
]
download_girder(url_tomoradio, stat_files, stat_subfolder)


# %%
# create recon folder
recon_subfolder = Path("recon")
recon_subfolder.mkdir(exist_ok=True)



# %%

# download files from this storage
# url = 'https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1'

# folder_ids = [
#     "66c860cab891f94a08ad81af", # data -> pilot
#     "66c72270b891f94a08ad818d", # models -> tomoradio en précisant doublons
#     "66c85fddb891f94a08ad81a8", # stat -> tomoradio en précisant doublons
# ]

# download_girder_folder(url, folder_ids, destination, False)





# =============================================================================





# # %%
# # DOWNLOADS

# # api Rest url of the warehouse
# url='https://pilot-warehouse.creatis.insa-lyon.fr/api/v1'

# # Download the covariance matrix and mean image
# # [Todo: add Ids from tomoradio-warehouse and remove unused files]

# stat_folder = './stat/'
# statId_list = [
#     '63935b584d15dd536f04849f', # Cov for reconstruction (imageNet, 128)
#     '63935b624d15dd536f0484a5', # Cov for reconstruction (imageNet, 64)
#     # '63935a224d15dd536f048490', # Average for reconstruction (imageNet, 128)
#     # '63935a224d15dd536f048496', # Average for reconstruction (imageNet, 64)
# ]
# download_girder(url, statId_list, stat_folder)

# # Download the models

# model_folder = './model/'
# modelId_list = [
#     '6410889f0386da274769778f', # DCNet - Unet / N_rec = 128, M = 4096
#     # '6410889a0386da274769778c', # N_rec = 128, M = 2048
#     # '641088930386da2747697789', # N_rec = 128, M = 1024
#     # '6410888d0386da2747697786', # N_rec = 128, M = 512
#     # '644a38c985f48d3da07140ba', # N_rec = 64, M = 4095
#     # '644a38c785f48d3da07140b7', # N_rec = 64, M = 1024
#     # '644a38c585f48d3da07140b4', # N_rec = 64, M = 512
# ]
# download_girder(url, modelId_list, model_folder)

# # missing models
# url2 = 'https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1'
# modelId_list2 = [
#     '66c72539b891f94a08ad8190', # lpgd - Unet / N_rec = 128, M = 4096
# ]
# download_girder(url2, modelId_list2, model_folder)

# # Download the raw data
# # [Todo: remove unused data]

# data_folder = './data/'
# folderId_list = [
#     # '637512e84d15dd536f048247', # bitten apple
#     # '622b65d543258e76eab217b3', # cat
#     # '622b5ea843258e76eab21740', # cat colored
#     # '639332784d15dd536f04838e', # color checker
#     # '622b634743258e76eab21767', # horse
#     # '616fe8ac478214d8c8a30d8e', # star sector x2
#     '616fff63478214d8c8a30e2f', # star sector x12
#     # '61e19b3fcdb6910b899d0147', # star sector colored
#     # '622b57f743258e76eab21719', # star sector off-centered
#     # '6328942febe9129ae9936f8d', # Thorlab box
#     '61b07f563f7ce79f5e565c46', # tomato x2
#     '61b0924b3f7ce79f5e565c7a', # tomato x12
#     # '632b2adaebe9129ae9937250', # tree leaf
#     # '617289d2478214d8c8a30f5c', # usaf x2
#     '6172a77c478214d8c8a30fde', # usaf x12
# ]
# # in the folder list, download only the files with the following suffixes
# suffix = ('_spectraldata.npz', '_metadata.json')

# # Generate the warehouse client
# gc = girder_client.GirderClient(apiUrl=url)

# for folderId in folderId_list:
#     for item in gc.listItem(folderId):
#         for file in gc.listFile(item['_id']):
            
#             if file['name'].endswith(suffix):
#                 folder_name = gc.getFolder(folderId)['name']
#                 download_girder(url, file['_id'],
#                                 Path(data_folder) / Path(folder_name))

# # %%
