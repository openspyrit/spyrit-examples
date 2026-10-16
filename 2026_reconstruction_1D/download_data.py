#%% Download bright and dark acquisitions to calculate calibration data

from pathlib import Path
from spyrit.misc.load_data import download_girder

# note: this downloads the entire folder (all integration time, camera gain configs) so takes quite a while (hours)
# to download only one set, change hexid to relevant subfolder 

# alternatively, the precalculated dark images for each camera configuration and integration time are available to 
# to download from the tomoradio warehouse (see next cell)

os.makedirs("acquisitionData", exist_ok=True)
# where to save data
destination = Path("acquisitionData")
print("Copying folder in:", destination)

url_pilot = "https://pilot-warehouse.creatis.insa-lyon.fr/api/v1"
download_girder(url_pilot, hex_ids ='6ac7963ff5d51d6655939c22' , local_folder = destination, gc_type="folder",) # bright
download_girder(url_pilot, hex_ids ='6ac796f1f5d51d66559499bf' , local_folder = destination, gc_type="folder",) # dark

download_girder(url_pilot, hex_ids ='6ad1eba4f5d51d665595dae0' , local_folder = destination, gc_type="folder",) #bright images


#%% Download pre-calculated noise parameters from tomoradio warehouse


os.makedirs("calibrationData", exist_ok=True)
# where to save data
destination = Path("calibrationData")
print("Copying folder in:", destination)

# download pre calculated system parameters data from the PILOT warehouse
url_tomoradio = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"

download_girder(url_pilot, hex_ids ='6ac8c8eedba19bd8c7063274' , local_folder = destination, gc_type="folder",) #dark images
# download_girder(url_pilot, hex_ids ='6ad1eba4f5d51d665595dae0' , local_folder = destination, gc_type="folder",) #bright images
download_girder(url_pilot, hex_ids ='6ac8c897dba19bd8c7063264' , local_folder = destination, gc_type="folder",) #gain images

# download pretrained models
download_girder(url_pilot, hex_ids ='6ad223bff5d51d665595dba2' , local_folder = Path(""), gc_type="folder",)


# download cov matrix
os.makedirs("data", exist_ok=True)
destination = Path("data")
print("Copying folder in:", destination)
download_girder(url_pilot, hex_ids ='6ad2242cf5d51d665595dbac' , local_folder = destination)

#%% Download raw spectral data from PILOT

from pathlib import Path
from spyrit.misc.load_data import download_girder

# where to save data

destination = Path("data")

print("Copying folder in:", destination)

# download data from the Pilot warehouse
url_pilot = "https://pilot-warehouse.creatis.insa-lyon.fr/api/v1"

datasets = [
    {
        "subfolder": "opticalTuningCat2",
        "comment": "SPIHIM/data/setup_v2.02026-06-19_optical_tuning/",
        "files": [
            "6a353465c392eb77b9ec1316",  # obj_cat2_..._zoom_x2/spectral_data.npz
            "6a35344cc392eb77b9ec0fff",  # metadata
            "6a35344cc392eb77b9ec0ffc",  # had
        ],
    },
    {
        "subfolder": "opticalTuningCat3",
        "comment": "SPIHIM/data/setup_v2.02026-06-19_optical_tuning/",
        "files": [
            "6a354d7cc392eb77b9ec162b",  # obj_cat3_..._zoom_x2/spectral_data.npz
            "6a354d65c392eb77b9ec131d",  # metadata
            "6a354d65c392eb77b9ec131a",  # had
        ],
    },
    {
        "subfolder": "USAF",
        "comment": "SPIHIM/data/setup_v2.02025-06-20_lens_tuning",
        "files": [
            "685522df3978e07db274b8a9",  # obj_USAF_..._zoom_x1/spectral_data.npz
            "685522b53978e07db274b592",  # metadata
            "685522b53978e07db274b58f",  # had
        ],
    },
    {
        "subfolder": "USAF2",
        "comment": "SPIHIM/data/setup_v2.02025-06-20_lens_tuning",
        "files": [
            "6855414a3978e07db274cb0e",  # obj_USAF2_..._zoom_x1/spectral_dat.npz
            "6855412b3978e07db274c4f4",  # metadata
            "6855412b3978e07db274c4f1",  # had
        ],
    },
]

for ds in datasets:
    data_subfolder = Path(ds["subfolder"])
    try:
        download_girder(url_pilot, ds["files"], data_subfolder)
    except Exception as e:
        print(f"Unable to download data from the Pilot warehouse ({ds['subfolder']})")
        print(e)


#%% alternatively download pre-formatted data directly from PILOT:

os.makedirs("data", exist_ok=True)
# where to save data
destination = Path("data")
print("Copying folder in:", destination)

url_pilot = "https://pilot-warehouse.creatis.insa-lyon.fr/api/v1"
download_girder(url_pilot, hex_ids = '6ad22481f5d51d665595dbad' , local_folder = destination, gc_type="folder",) # cat


# %% Some spectral data on the PILOT is saved in a different form
from tools import load_experiment
import numpy as np

# download spectral data folder from PILOT, saved to the following path
destination = Path("data")
download_girder(url_pilot, '6ac66658f5d51d66558fbd43',  local_folder = destination, gc_type="folder" )

# choose data to read in 
path = Path("data/cat")

# extract info - make this more general someday
data = load_experiment(path, gr=2, lc="600")

# dimensions
M = data["M"]
N = data["N"]
L = data["L"]

patterns = data["patterns"]
wavelengths = data["wavelengths"]
Lc = data["Lc"]

# measurements
raw_data = data["raw_data"]  # shape (N,L,P) unbinned
spectral_data_all = data["spectral_data_all"] # shape (N,L,P) binned
spatial_data = data["spatial_data"]

import os
os.makedirs("data/cat", exist_ok=True)
np.save("data/cat/m_binned.npy", spectral_data_all)

# %%
