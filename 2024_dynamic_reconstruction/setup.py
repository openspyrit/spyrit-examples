"""Run this file to setup files and directories for the project.
"""

# %%
import os
import configparser

from spyrit.misc.load_data import download_girder
import setup_aux


# %%
configfile_path = "config.ini"
config = configparser.ConfigParser()
config.read(configfile_path)
general = config["GENERAL"]


# %%
# get directory names & create directories
stats_folder = general.get("stats_folder")
nn_models_folder = general.get("nn_models_folder")
deformation_folder = general.get("deformation_folder")
image_sample_folder = general.get("image_sample_folder")

os.makedirs(stats_folder, exist_ok=True)
os.makedirs(nn_models_folder, exist_ok=True)
os.makedirs(deformation_folder, exist_ok=True)
os.makedirs(image_sample_folder, exist_ok=True)


# %%
# download the 4 covariance matrices (> 2GB)
url = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
folder = stats_folder
file_ids = [
    "672b8077f03a54733161e970",  # measurement covariance 64x64
    "67486be9a438ad25e7a001fa",  # image covariance 64x64
    "672b80acf03a54733161e973",  # measurement covariance 128x128
    "67486be8a438ad25e7a001f7",  # image covariance 128x128
]
download_girder(url, file_ids, folder)


# %%
# save the deformation field
setup_aux.save_elastic_deformation(configfile_path)
# create a Siemens star and save it
setup_aux.save_siemens_star(configfile_path)
print("Setup complete!")


# %%
