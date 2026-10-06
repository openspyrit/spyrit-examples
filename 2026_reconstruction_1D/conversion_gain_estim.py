#%%
# 
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tools import binArray
import os

os.makedirs("CalibrationData/bright", exist_ok=True)
#%% params
bin_fact = 3
G = 0 # 12.04 # 6.02 # 0

# used for dark - all bright images were acquired at 1.5ms
ti = 1.0 # 10.0 # 100.0 #1000.0


# Acquired bright images - local 
data_folder = Path(r"C:/Users/ceidigh/Documents/2026-05-05_calib_bruit/white")
intensities = ["ND0", "ND1", "ND2"] #, "ND3", "ND4"]

# ROI - ADD SCRIPTS WHERE THIS WAS CHOOSEN!!!
x1, x2 = 75,95
y1, y2 = 221, 271

N, L = 384, 608
n_images  = 1000

mu_dark = np.load(f"CalibrationData/dark/mu_dark_image_binned_x{bin_fact}_{ti}_{G}.npy")  # shape (128, 608)
var_dark = np.load(f"CalibrationData/dark/var_dark_image_binned_x{bin_fact}_{ti}_{G}.npy") # shape (128, 608)
mu_dark_ROI = mu_dark[x1:x2, y1:y2]   # crop

# %% FFP METHOD

all_bright_images = np.empty((n_images, len(intensities), 128, L))
signals_g1 = np.zeros((500, len(intensities)))
variances_g1 = np.zeros((500, len(intensities)))

# For each pair
for i in range(0, 1000, 2):
    a = i
    b = i + 1

    # Loop through each intensity measured
    for t_idx, ti in enumerate(intensities):
        raw_data_folder = (data_folder / f"{ti}_ti_1.5ms" /
                            (f"obj_{G}dB_source_white_LED _"
                             f"Lc_600nm_Gr_2_Walsh_im_1x1_"
                             f"ti_1.5ms_zoom_x1") / "raw_data")
        file_path_A = raw_data_folder / f"spectral_NR_0_Gr_2_Lc_600nm_NA_{a}_NS_0.npz"
        file_path_B = raw_data_folder / f"spectral_NR_0_Gr_2_Lc_600nm_NA_{b}_NS_0.npz"

        # load each image, bin as necessary
        A = np.load(file_path_A)["arr_0"].astype(np.float64)
        B = np.load(file_path_B)["arr_0"].astype(np.float64)
        A = binArray(A, 0, bin_fact, bin_fact, func=np.sum)
        B = binArray(B, 0, bin_fact, bin_fact, func=np.sum)

        # store binned images
        all_bright_images[i, t_idx, :, :] = A
        all_bright_images[i + 1, t_idx, :, :] = B

        # extract ROI, dark-subtracted
        A_roi = A[x1:x2,y1:y2]
        B_roi = B[x1:x2,y1:y2]

        # calculate and store (S, \sigma^2) point for this pair
        S = ((A_roi-mu_dark_ROI).mean()
            +(B_roi-mu_dark_ROI).mean())/2

        D = A_roi - B_roi
        sigma2 = (D.var(ddof=1)/2) - var_dark[x1:x2, y1:y2].mean()

        signals_g1[i // 2, t_idx] = S
        variances_g1[i // 2, t_idx] = sigma2

# calculate average over all pairs
mu_g1 = signals_g1[0:1, :].mean(0)
var_g1 = variances_g1[0:1,:].mean(0)


# save all the bright images
np.save(f'CalibrationData/bright/all_bright_images_bin_x{bin_fact}_{G}.npy', all_bright_images)

# save the average mu & sigma value, per ND, accross all 500 pairs
np.save(f'CalibrationData/bright/mu_g1_bin_x{bin_fact}_{G}.npy', signals_g1.mean(0))
np.save(f'CalibrationData/bright/var_g1_bin_x{bin_fact}_{G}.npy', variances_g1.mean(0))

# %% TEMPORAL ESTIMATOR

# load all images - prepocessed in previous 
all_ims = np.load(f"CalibrationData/bright/all_bright_images_bin_x{bin_fact}_{G}.npy") - mu_dark[None, None, :, :]

# calculate and store average mu and variance images
sum_frames = all_ims.sum(0)
sum_ROI = sum_frames[:, x1:x2, y1:y2]
signals_g2 = (sum_ROI / 1000).mean((1,2))
variances_g2 = (all_ims[:, :, x1:x2, y1:y2].var(axis=0, ddof=1)- var_dark[x1:x2,y1:y2]).mean((1,2)) 

np.save(f'CalibrationData/bright/mu_g2_bin_x{bin_fact}_{G}.npy', signals_g2)
np.save(f'CalibrationData/bright/var_g2_bin_x{bin_fact}_{G}.npy', variances_g2)

#%% GAIN PER PIXEL WITH TEMPORAL ESTIMATOR

# average bright mu and var images, one per intensity 
mu_bright_image = np.load(f"CalibrationData/bright/all_bright_images_bin_x{bin_fact}_{G}.npy").mean(0) - mu_dark
var_bright_image = np.load(f"CalibrationData/bright/all_bright_images_bin_x{bin_fact}_{G}.npy").var(axis=0,ddof=1) - var_dark

x = mu_bright_image              # (5,128,608)
y = var_bright_image

# mean across ND levels
x_bar = x.mean(axis=0)
y_bar = y.mean(axis=0)

# Least-squares slope
num = np.sum((x - x_bar) * (y - y_bar), axis=0)
den = np.sum((x - x_bar) ** 2, axis=0)

gain_image = num / den

# Corresponding intercept map
intercept_image = y_bar - gain_image * x_bar

print(f"ROI gain mean : {gain_image[x1:x2,y1:y2].mean():.5f}")
print(f"ROI gain std  : {gain_image[x1:x2,y1:y2].std():.5f}")

#%% PLOT GAIN MAP
plt.figure(figsize=(10,4))

plt.subplot(121)
plt.imshow(gain_image)
plt.xlabel(r'$\Lambda$')
plt.ylabel(r'$N_y$')
plt.title(
    f"Gain map for {G} dB\n"
    + rf"$\bar{{\gamma}}_{{ROI}}$ = {gain_image[x1:x2, y1:y2].mean():.5f} $\pm$ {gain_image[x1:x2,y1:y2].std():.5f} counts/photon" 
)
plt.colorbar(orientation='horizontal')

#%% PLOT AVERAGE GAIN VALUE
fig, axs = plt.subplots(1, 2, figsize=(8, 6))
fig.suptitle(f"{G}\nPhoton Transfer Curve ( FFP vs Temporal Estimator)", fontsize=16)

labels = ["ND0", "ND1", "ND2", "ND3", "ND4"]


# =========================
# COLUMN 1 — AVERAGE
# =========================
x = mu_g1
y = var_g1

slope, intercept = np.polyfit(x, y, 1)

x_fit = np.linspace(x.min(), x.max(), 200)
y_fit = slope * x_fit + intercept

axs[0].plot(x_fit, y_fit, '--r', label=f"fit")
axs[0].text(0.05, 0.95,
              f"slope = {slope:.3e}",
            #   f"slope = {slope:.3e}\nG = {1/slope:.3e}",
              transform=axs[0].transAxes,
              va='top',
              bbox=dict(facecolor='white', alpha=0.7))

for xk, yk, lab in zip(x, y, labels):
    axs[0].plot(xk, yk, 'o', label=lab)

axs[0].plot(x, y, '-k', alpha=0.5)
axs[0].set_title("Flatfield Pair PTC")
axs[0].set_xlabel(rf"Mean signal $\bar{{S}}$")
axs[0].set_ylabel("Variance σ²")
axs[0].grid(True)
axs[0].legend()

slope_g1 = slope
# =========================
# COLUMN 2 — PRESUMED AVG
# =========================
x = signals_g2
y = variances_g2

slope, intercept = np.polyfit(x, y, 1)

x_fit = np.linspace(x.min(), x.max(), 200)
y_fit = slope * x_fit + intercept

axs[1].plot(x_fit, y_fit, '--r', label=f"fit")
axs[1].text(0.05, 0.95,
              f"slope = {slope:.3e}",
            #   f"slope = {slope:.3e}\nG = {1/slope:.3e}",
              transform=axs[1].transAxes,
              va='top',
              bbox=dict(facecolor='white', alpha=0.7))

for xk, yk, lab in zip(x, y, labels):
    axs[1].plot(xk, yk, 'o', label=lab)

axs[1].plot(x, y, '-k', alpha=0.5)
axs[1].set_title("Temporal Estimation PTC")
axs[1].set_xlabel(rf"Mean signal $\bar{{S}}$")
axs[1].set_ylabel("Variance σ²")
axs[1].grid(True)
axs[1].legend()

slope_g2 = slope

plt.tight_layout()
plt.show()

# %% SAVE GAIN ESTIMATIONS
np.save(f"CalibrationData/bright/slope_g1_bin_x{bin_fact}_{G}.npy", slope_g1)
np.save(f"CalibrationData/bright/slope_g2_bin_x{bin_fact}_{G}.npy", slope_g2)
np.save(f"CalibrationData/bright/slope_g3_bin_x{bin_fact}_{G}.npy", gain_image[x1:x2, y1:y2].mean())
np.save(f"CalibrationData/gain_image_bin_x{bin_fact}_{G}.npy", gain_image)

# %%
