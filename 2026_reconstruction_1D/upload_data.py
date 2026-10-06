#%%
import girder_client

gc = girder_client.GirderClient(apiUrl='https://https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e)')
                        
gc.authenticate(username='username', interactive=True)
# %%


gc.upload(
    'C:/Users/ceidigh/Documents/PFE/CalibrationData',  # local folder
    '69f362d4b1e3942938623fd3',                          # target folder ID in Girder
    reuse_existing=True   # skip re-uploading files that already exist there
)

gc.upload(
    'C:/Users/ceidigh/Documents/2026-05-05_calib_bruit',  # local folder
    '69f362d4b1e3942938623fd3',                          # target folder ID in Girder
    reuse_existing=True   # skip re-uploading files that already exist there
)
