#%%
import girder_client


my_folder = '6ac65d6ff5d51d66558ef673'
gc = girder_client.GirderClient(apiUrl="https://pilot-warehouse.creatis.insa-lyon.fr/api/v1")
gc.authenticate(apiKey = '4Qkli33iujtL8s9GV2LggnyZEMoac130Pl2fWTUU')

gc.upload(
    rf'C:\Users\ceidigh\Documents\2026-05-05_calib_bruit\dark',
    my_folder,
)



# gc.upload(
#     rf'C:\Users\ceidigh\Documents\2026-05-05_calib_bruit\bright',
#     my_folder,
# )


# Then make the newly created subfolder public
# (find the subfolder's ID — it's the child folder named "bright" under my_folder)
children = list(gc.listFolder(my_folder, parentFolderType='folder'))
dark_folder = next(f for f in children if f['name'] == 'dark')

gc.put(f"folder/{dark_folder['_id']}", parameters={'public': True})

# %%
