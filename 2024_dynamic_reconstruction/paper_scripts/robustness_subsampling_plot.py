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

import re
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
config.read("robustness_config.ini")

general = config["GENERAL"]
networks = config["NETWORKS"]
subsampling_analysis = config["SUBSAMPLING_ANALYSIS"]
deformation = config["DEFORMATION"]

# General
image_sample_folder = pathlib.Path(general.get("image_sample_folder"))
stats_folder = pathlib.Path(general.get("stats_folder"))
deformation_folder = pathlib.Path(general.get("deformation_folder"))
nn_models_folder = pathlib.Path(general.get("nn_models_folder"))
force_cpu = general.getboolean("force_cpu")
image_size = general.getint("image_size")
pattern_size = general.getint("pattern_size")
siemens_star_branches = general.getint("siemens_star_branches")
siemens_star_template = general.get("siemens_star_template")
siemens_star_fillers = general.get("siemens_star_fillers")

# Subsampling analysis
subsampling_steps = subsampling_analysis.getint("subsampling_steps")
save_images = subsampling_analysis.getboolean("save_images")
save_tensors = subsampling_analysis.getboolean("save_tensors")
reconstruction_folder = pathlib.Path(subsampling_analysis.get("reconstruction_folder"))
save_reconstruction_template = subsampling_analysis.get("save_reconstruction_template")
save_reconstruction_fillers = subsampling_analysis.get("save_reconstruction_fillers")
nn_to_use = subsampling_analysis.get("nn_to_use")

# Deformation
torch_seed = deformation.getint("torch_seed")
displacement_magnitude = deformation.getint("displacement_magnitude")
displacement_smoothness = deformation.getint("displacement_smoothness")
frame_generation_period = deformation.getint("frame_generation_period")
deformation_mode = deformation.get("deformation_mode")
compensation_mode = deformation.get("compensation_mode")
save_deformation_template = deformation.get("save_deformation_template")
save_deformation_fillers = deformation.get("save_deformation_fillers")
save_metric_template = deformation.get("save_metric_template")
save_metric_fillers = deformation.get("save_metric_fillers")


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

valid_nn_names = [
    valid_name for valid_name in re.split(r"\W", nn_to_use) if len(valid_name) > 0
]
nn_filenames = {name: networks.get(name) for name in valid_nn_names}
# =============================================================================


# %%
# Load the reference image
image_name = siemens_star_template.format(*eval(siemens_star_fillers))
image_path = image_sample_folder / image_name

if not image_path.exists():
    from aux_functions import save_siemens_star

    print(f"Image {image_name} not found in {image_sample_folder}, creating it")
    save_siemens_star(image_path, figsize=image_size, n=siemens_star_branches)

x = torchvision.io.read_image(
    image_sample_folder / image_name, torchvision.io.ImageReadMode.GRAY
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
    reconstruction_name = save_reconstruction_template.format(
        *eval(save_reconstruction_fillers)
    )
    reconstruction_path = reconstruction_folder / reconstruction_name

    PSNRs[i] = PSNR_from_file(reconstruction_path, x, center_crop=64)
    SSIMs[i] = SSIM_from_file(reconstruction_path, x, center_crop=64)

# save the metrics
metric_name = "PSNR"
psnr_save_name = save_metric_template.format(*eval(save_metric_fillers))
metric_name = "SSIM"
ssim_save_name = save_metric_template.format(*eval(save_metric_fillers))

torch.save(PSNRs, reconstruction_folder / psnr_save_name)
print("PSNRs saved in", reconstruction_folder / psnr_save_name)
torch.save(SSIMs, reconstruction_folder / ssim_save_name)
print("SSIMs saved in", reconstruction_folder / ssim_save_name)


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

# %%
