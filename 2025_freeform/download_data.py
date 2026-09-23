# This file is used to download all data used in the paper. 
# It uses the girder_client library to connect to the Girder API and get the 
# data from the specified folders stored in the Pilot warehouse.

# %%
from pathlib import Path
import girder_client

# download data from the Pilot warehouse
url_pilot = "https://pilot-warehouse.creatis.insa-lyon.fr/api/v1"
data_subfolder = Path("data")

gc = girder_client.GirderClient(apiUrl=url_pilot)

# Girder folder ID
folder_list = [
    "6aa3a5dbf5d51d66558ed13b",  # 2026-09-11_freeform_publication
    "68d5069cc68404167c562973",  # 2025-09-25_freeform_publication
    "68c2c688c68404167c562799",  # 2025-09-11_freeform_SNR
]

for folder_id in folder_list:
    # 1. Fetch the folder metadata to get its name
    folder_info = gc.get(f"folder/{folder_id}")
    folder_name = folder_info["name"]
    destination_path = data_subfolder / folder_name

    # 2. Download the folder recursively
    gc.downloadFolderRecursive(folder_id, destination_path)