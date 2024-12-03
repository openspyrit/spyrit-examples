"""Script that computes Tikhonov reconstructions under different subsampling rates.
No noise is used.

The reconstruction is done using the Tikhonov regularization which corrects the
reconstructed image using a prior image covariance matrix. A denoising nn
can then be applied to perfect the result.

In this case, a 128x128 Siemens Star is used as a base image, deformed using a
dynamic elastic deformation field. The measurements patterns are 64x64 and are
centered so that the center of the image is aligned with the measurement patterns.

The results are saved in png and/or pt files, so they can be analyzed in the
corresponding `_analysis` script.
"""

# %%

import pathlib
import configparser

# import math
# import torch.nn as nn
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# import spyrit.core.prep as prep
# import spyrit.core.warp as warp
# import spyrit.core.nnet as nnet
# import spyrit.core.noise as noise
# import spyrit.core.train as train
# import spyrit.misc.statistics as stats
# import spyrit.misc.disp as disp
# import spyrit.core.meas as meas
# import spyrit.core.recon as recon
# import spyrit.core.torch as spytorch
# from spyrit.misc.load_data import download_girder

from aux_functions import PSNR_from_file, SSIM_from_file, fractions


# %%
# READ PARAMETERS
config = configparser.ConfigParser()
config.read("config.ini")

# General
force_cpu = config.getboolean("GENERAL", "force_cpu")
# Image
image_size = config.getint("IMAGE", "image_size")
siemens_star_branches = config.getint("IMAGE", "siemens_star_branches")
# Measurements
pattern_size = config.getint("MEASUREMENTS", "pattern_size")
subsampling_steps = config.getint("MEASUREMENTS", "subsampling_steps")
# Load name
siemens_star_template = config.get("LOAD_NAME", "siemens_star_template")
siemens_star_fillers = config.get("LOAD_NAME", "siemens_star_fillers")
# Save name
reconstruction_template = config.get("SAVE_NAME", "reconstruction_template")
reconstruction_fillers = config.get("SAVE_NAME", "reconstruction_fillers")
metric_template = config.get("SAVE_NAME", "metric_template")
metric_fillers = config.get("SAVE_NAME", "metric_fillers")
# Paths
load_folder_siemens_star = pathlib.Path(
    config.get("FOLDERS", "load_folder_siemens_star")
)
save_folder_subsampling_reconstruction = pathlib.Path(
    config.get("FOLDERS", "save_folder_subsampling_reconstruction")
)


# %%
# Derived parameters
# =============================================================================
subsamplings = torch.tensor(
    [(subsampling_steps - sub) / subsampling_steps for sub in range(subsampling_steps)]
)
n_measurements_list = [int(pattern_size**2 * subs) for subs in subsamplings]
n_frames_list = [2 * n for n in n_measurements_list]
frames = max(n_frames_list)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() and not force_cpu else "cpu"
)
print(f"Using device: {device}")
# =============================================================================


# %%
# Load the reference image
image_name = siemens_star_template.format(*eval(siemens_star_fillers))
image_path = load_folder_siemens_star / image_name

if not image_path.exists():
    from aux_functions import save_siemens_star

    print(f"Image {image_name} not found in {load_folder_siemens_star}, creating it")
    save_siemens_star(image_path, figsize=image_size, n=siemens_star_branches)

x = torchvision.io.read_image(
    load_folder_siemens_star / image_name, torchvision.io.ImageReadMode.GRAY
)
x = x.detach().float().to(device)
# rescale x from [0, 255] to [-1, 1]
x = 2 * (x / 255) - 1
c, h, w = x.shape


# %%
# Load each reconstructed image, compute the PSNR and SSIM

PSNRs = torch.zeros(subsampling_steps)
SSIMs = torch.zeros(subsampling_steps)

for i in range(subsampling_steps):
    print("Computing PSNR and SSIM for subsampling", i + 1)

    n_frames = n_frames_list[i]
    n_measurements = n_measurements_list[i]
    reconstruction_name = (
        f"{reconstruction_template.format(*eval(reconstruction_fillers))}.pt"
    )
    reconstruction_path = save_folder_subsampling_reconstruction / reconstruction_name

    PSNRs[i] = PSNR_from_file(reconstruction_path, x, center_crop=64)
    SSIMs[i] = SSIM_from_file(reconstruction_path, x, center_crop=64)

# save the metrics
metric_name = "PSNR"
psnr_save_name = metric_template.format(*eval(metric_fillers))
metric_name = "SSIM"
ssim_save_name = metric_template.format(*eval(metric_fillers))

torch.save(PSNRs, save_folder_subsampling_reconstruction / f"{psnr_save_name}.pt")
print("PSNRs saved in", save_folder_subsampling_reconstruction / f"{psnr_save_name}.pt")
torch.save(SSIMs, save_folder_subsampling_reconstruction / f"{ssim_save_name}.pt")
print("SSIMs saved in", save_folder_subsampling_reconstruction / f"{ssim_save_name}.pt")


# %%
# Plot the results
# subsampling_ratio = 1.0 / subsamplings
subsampling_ratio = subsamplings  # .flip(0)  # flip so that we have 1 at the left
PSNRs_ = PSNRs  # .flip(0)  # flip the rest
SSIMs_ = SSIMs  # .flip(0)

# put the PSNRs and SSIMs in the same plot using left and right y-axis
fig, ax = plt.subplots(dpi=300)
ax2 = ax.twinx()

ax.plot(
    subsampling_ratio,
    PSNRs_,
    label="PSNR",
    color="tab:blue",
    linestyle="--",
    lw=0.75,
    marker="+",
)
ax.set_xlabel("Proportion of total measurements")
ax.set_ylabel("PSNR", color="tab:blue")
ax.tick_params(axis="y", labelcolor="tab:blue")

ax2.plot(
    subsampling_ratio,
    SSIMs_,
    label="SSIM",
    color="tab:red",
    linestyle="--",
    lw=0.75,
    marker="+",
)
ax2.set_ylabel("SSIM", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.0 / subsampling_steps))
ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0 / subsampling_steps))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fractions(subsampling_steps / 2)))
ax.set_xlim(1, 0)
ax.grid(axis="both")

plt.title("Reconstruction quality when subsampling")
plt.show()
