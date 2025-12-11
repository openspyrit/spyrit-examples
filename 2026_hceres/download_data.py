# %%
# IMPORTS

from pathlib import Path
from spyrit.misc.load_data import download_girder

# %% SETTINGS
# change this if you want to download the files in a different folder
destination = Path.cwd()  # / Path("your_subfolder")
print("Copying folder in:", destination)


# %%
# download data from the Pilot warehouse
url_pilot = "https://pilot-warehouse.creatis.insa-lyon.fr/api/v1"
data_subfolder = Path("data")
data_files = [
    "691204cfc68404167c562af2",  # setup_v1.3.1/2025-11-10_test_HCERES/ folder
]
try:
    download_girder(url_pilot, data_files, data_subfolder)
except Exception as e:
    print("Unable to download data from the Pilot warehouse")
    print(e)


# %%
# download images for simulation from the Tomoradio warehouse
url_tomoradio = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
images_subfolder = data_subfolder / Path("images/cropped")
image_files = [
    "670911acf03a54733161e956",  # HSI_Brain_012-01_crop.jpeg
    "66cdd369b891f94a08ad81eb",  # ILSVRC2012_val_00000003_crop.JPEG
    "66cdd36ab891f94a08ad81ee",  # ILSVRC2012_val_00000012_crop.JPEG
    "66cdd36ab891f94a08ad81f1",  # ILSVRC2012_val_00000019_crop.JPEG
    "66cdd36ab891f94a08ad81f4",  # ILSVRC2012_val_00000051_crop.JPEG
    "66cdd36ab891f94a08ad81f7",  # ILSVRC2012_val_00000056_crop.JPEG
]
try:
    download_girder(url_tomoradio, image_files, images_subfolder)
except Exception as e:
    print("Unable to download images from the Tomoradio warehouse")
    print(e)

image_fig2_subfolder = data_subfolder / Path("images/figure_2")
image_files_fig2 = "668e986a7d138728d4806d7a"  # ILSVRC2012_test_00000002.jpeg
try:
    download_girder(url_tomoradio, image_files_fig2, image_fig2_subfolder)
except Exception as e:
    print("Unable to download image for figure 2 from the Tomoradio warehouse")
    print(e)


# %%
# download models from the Tomoradio warehouse
model_subfolder = Path("model")
model_files = [
    "66cf34f5b891f94a08ad8212",  # Pinv-net
    "66cf34f5b891f94a08ad8218",  # LPGD
    "66c72804b891f94a08ad8193",  # DC-Net
    "66cc9a7bb891f94a08ad81e4",  # Pinv - PnP
    "66d6cd3eb891f94a08ad8221",  # dfb-net nch_1 / varnoise0.05
    "66d6cd3eb891f94a08ad8224",  # dfb-net nch_1 / varnoise0.1
    "66cf34f5b891f94a08ad8215",  # Pinv-net retrained
]
try:
    download_girder(url_tomoradio, model_files, model_subfolder)
except Exception as e:
    print("Unable to download images from the Tomoradio warehouse")
    print(e)


# %%
# download statistics from the Tomoradio warehouse
stat_subfolder = Path("stat")
# these are the same matrices, in numpy or pytorch format
stat_files = [
    "692965fb4c0b0d3d4bc700ae",  # 128x128 pt, resize
]
try:
    download_girder(url_tomoradio, stat_files, stat_subfolder)
except Exception as e:
    print("Unable to download images from the Tomoradio warehouse")
    print(e)


# %%
# create recon folder
recon_subfolder = Path("recon")
recon_subfolder.mkdir(exist_ok=True)
