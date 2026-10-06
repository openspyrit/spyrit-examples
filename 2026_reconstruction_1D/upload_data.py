#%%

import girder_client
gc = girder_client.GirderClient(apiUrl='https://data.kitware.com/api/v1')
gc.authenticate('username', 'password')
gc.inheritAccessControlRecursive('54b43e9b8926486c0c06cb4f')
# %%
