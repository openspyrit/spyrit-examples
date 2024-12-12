"""Run this file to setup files and directories for the project.
"""

import os
import configparser

from spyrit.misc.load_data import download_girder


config = configparser.ConfigParser()
config.read("robustness_config.ini")
general = config["GENERAL"]

# get directory names
stats_folder = general.get("stats_folder")
nn_models_folder = general.get("nn_models_folder")
deformation_folder = general.get("deformation_folder")
image_sample_folder = general.get("image_sample_folder")

# create directories
os.makedirs(stats_folder, exist_ok=True)
os.makedirs(nn_models_folder, exist_ok=True)
os.makedirs(deformation_folder, exist_ok=True)
os.makedirs(image_sample_folder, exist_ok=True)

# download the 4 covariance matrices (> 2GB)
url = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
folder = stats_folder
file_ids = [
    "672b8077f03a54733161e970",
    "672b80acf03a54733161e973",
    "67486be9a438ad25e7a001fa",
    "67486be8a438ad25e7a001f7",
]
# download_girder(url, file_ids, folder)
