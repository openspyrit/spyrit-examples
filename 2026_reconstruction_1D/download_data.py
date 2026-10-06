
#%% Download data from PILOT

from pathlib import Path
from spyrit.misc.load_data import download_girder

# where to save data
destination = Path("C:/Users/ceidigh/Documents/PFE/")
print("Copying folder in:", destination)

# download data from the Pilot warehouse
url_pilot = "https://pilot-warehouse.creatis.insa-lyon.fr/api/v1"

datasets = [
    {
        "subfolder": "data/opticalTuningCat2",
        "comment": "SPIHIM/data/setup_v2.02026-06-19_optical_tuning/",
        "files": [
            "6a353465c392eb77b9ec1316",  # obj_cat2_..._zoom_x2/spectral_data.npz
            "6a35344cc392eb77b9ec0fff",  # metadata
            "6a35344cc392eb77b9ec0ffc",  # had
        ],
    },
    {
        "subfolder": "data/opticalTuningCat3",
        "comment": "SPIHIM/data/setup_v2.02026-06-19_optical_tuning/",
        "files": [
            "6a354d7cc392eb77b9ec162b",  # obj_cat3_..._zoom_x2/spectral_data.npz
            "6a354d65c392eb77b9ec131d",  # metadata
            "6a354d65c392eb77b9ec131a",  # had
        ],
    },
    {
        "subfolder": "data/USAF",
        "comment": "SPIHIM/data/setup_v2.02025-06-20_lens_tuning",
        "files": [
            "685522df3978e07db274b8a9",  # obj_USAF_..._zoom_x1/spectral_data.npz
            "685522b53978e07db274b592",  # metadata
            "685522b53978e07db274b58f",  # had
        ],
    },
    {
        "subfolder": "data/USAF2",
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

# %% Some data on the PILOT is saved in a different form
from tools import load_experiment
import numpy as np

# download spectral data folder from PILOT, saved to the following path
# choose data to read in 
selected_paths = {
    
    # "cat": (Path("obj_cat_Lc_18.06dB_source_white_LED_Lc_540nm_Gr_1_Walsh_im_32x32_ti_1.0ms_zoom_x2"), 1),
    #"cat": (Path("obj_cat_Lc_18.06dB_source_white_LED_Lc_540nm_Gr_2_Walsh_im_128x128_ti_1.0ms_zoom_x2"), 2)
   # "cat": (Path("obj_cat_Lc_18.06dB_source_white_LED_Lc_540nm_Gr_2_Walsh_im_64x64_ti_1.0ms_zoom_x2"), 2),

    "cat" : (Path("C:/Users/ceidigh/Documents/2026-05-05_calib_bruit/obj_cat_12dB_source_white_LED_Lc_600nm_Gr_2_Walsh_im_128x128_ti_2.0ms_zoom_x1"), 2, "600")
}

datasets = {
    name: load_experiment(path, gr=gr, lc=lc)
    for name, (path, gr,lc) in selected_paths.items()
}


# extract info - make this more general someday
data = datasets["cat"]

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
