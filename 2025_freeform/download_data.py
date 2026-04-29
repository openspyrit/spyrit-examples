# %%
# IMPORTS

from pathlib import Path
from spyrit.misc.load_data import download_girder

# %%
# download data from the Pilot warehouse
url_pilot = "https://pilot-warehouse.creatis.insa-lyon.fr/api/v1"
data_subfolder = Path("data")
data_files = [
    "61b07f593f7ce79f5e565c5f",  # tomato x2 spectraldata.npz
    "61b07f583f7ce79f5e565c5c",  # tomato x2 metadata.json
    "61b0924e3f7ce79f5e565c93",  # tomato x12 spectraldata.npz
    "61b0924d3f7ce79f5e565c90",  # tomato x12 metadata.json
    "616fff65478214d8c8a30e45",  # starsector x12 spectraldata.npz
    "616fff64478214d8c8a30e42",  # starsector x12 metadata.json
    "6172a77e478214d8c8a30ff7",  # usaf x12 spectraldata.npz
    "6172a77d478214d8c8a30ff4",  # usaf x12 metadata.json
]
try:
    download_girder(url_pilot, data_files, data_subfolder)
except Exception as e:
    print("Unable to download data from the Pilot warehouse")
    print(e)