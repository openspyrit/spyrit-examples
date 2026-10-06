#%% 
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
from tools import binArray

#%% Choose acquisition params
# bin dark images to match convention of whatever data they'll be used for
bin_fact = 3  # eg here bin spatiall dim (y) by 3 : 384 -> 128
G = 18.06 # 12.04 # 6.02 # 0.0
ti = 1.0 # 10.0 # 100.0 #1000.0
#%% RUN TO UPDATE THE SAVED DARK IMAGES ARRAY -  BINNING CHOICE ETC

# where raw dark acquisitions are stored
data_folder = Path(r"C:/Users/ceidigh/Documents/2026-05-05_calib_bruit/dark")
raw_data_folder = (data_folder/ f"ti_{ti}ms"/ ( f"obj_{G}dB_source_No source_" f"Lc_600nm_Gr_2_Walsh_im_1x1_" f"ti_{ti}ms_zoom_x1")/ "raw_data" )

# dimensions - could of course extract to avoid hard coding blah blah blah 
N, L = 384, 608
n_images  = 1000

dark_images = np.zeros((n_images, N, L))

mu_dark = np.empty((N, L), )
var_dark = np.empty((N, L), )
sigma_dark = np.empty((N, L), )

# Running sum 
acc = np.zeros((N, L))
acc_sq = np.zeros((N, L))
count = 0

# store all K dark acqs
for k in range(n_images):
    file_path = (raw_data_folder/ f"spectral_NR_0_Gr_2_Lc_600nm_NA_{k}_NS_0.npz")

    if not file_path.exists():
        print(f"    WARNING: Missing file {file_path}")
        continue

    dark_images[k, : ,: ] = np.load(file_path)["arr_0"].astype(np.float64)


dark_images = binArray(dark_images, 1, bin_fact, bin_fact, func=np.sum)
dark_images.shape

# SAVE DATA

os.makedirs("CalibrationData/dark", exist_ok=True)
np.save(f"CalibrationData/dark/dark_images_binned_x{bin_fact}_{ti}_{G}.npy", dark_images)

#%% LOAD SAVED DARK IMAGES AND CALCULATE DARK MEAN IMAGE AND DARK VARIANCE IMAGE
dark_images = np.load(f"CalibrationData/dark/dark_images_binned_x{bin_fact}_{ti}_{G}.npy")

mu_dark_image = dark_images.mean(axis=0)
var_dark_image = dark_images.var(axis=0, ddof=1)
sigma_dark_image = dark_images.std(axis=0, ddof=1)

mu_dark = mu_dark_image.mean()
var_dark = var_dark_image.mean()
sigma_dark = sigma_dark_image.mean()

fig, axs = plt.subplots(1, 3, figsize=(16, 8))
im = axs[0].imshow(mu_dark_image)

axs[0].set_xlabel(r'$\Lambda$')
axs[0].set_ylabel(r'$N_y$')
axs[0].set_title(
    f"Dark Mean Image \n"
    f"$\\hat{{\\bar{{\\mu}}}}_{{DARK}}$ = {mu_dark:.2f} "
    f"$\\pm${mu_dark_image.std():.2f}"
)

plt.colorbar(im, ax=axs[0], orientation='horizontal')

im = axs[1].imshow(var_dark_image)

axs[1].set_xlabel(r'$\Lambda$')
axs[1].set_ylabel(r'$N_y$')
axs[1].set_title(
    f"Dark Variance Image \n"
    f"$\\hat{{\\bar{{\\sigma}}}}^2_{{DARK}}$ = {var_dark:.2f} "
    f"$\\pm${var_dark_image.std():.2f}"
)

plt.colorbar(im, ax=axs[1], orientation='horizontal')

im = axs[2].imshow(sigma_dark_image)

axs[2].set_xlabel(r'$\Lambda$')
axs[2].set_ylabel(r'$N_y$')
axs[2].set_title(
    f"Sigma Dark Image \n"
    f"$\\hat{{\\bar{{\\sigma}}}}_{{DARK}}$ = {sigma_dark:.2f} "
    f"$\\pm${sigma_dark_image.std():.2f}"
)

plt.colorbar(im, ax=axs[2], orientation='horizontal')

# %% SAVE DATA
import os
os.makedirs("CalibrationData/dark", exist_ok=True)
np.save(f"CalibrationData/dark/mu_dark_image_binned_x{bin_fact}_{ti}_{G}.npy", mu_dark_image)
np.save(f"CalibrationData/dark/var_dark_image_binned_x{bin_fact}_{ti}_{G}.npy", var_dark_image)
np.save(f"CalibrationData/dark/sigma_dark_image_binned_x{bin_fact}_{ti}_{G}.npy", sigma_dark_image)

