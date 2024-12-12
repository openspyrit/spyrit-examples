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
noise_analysis = config["NOISE_ANALYSIS"]
deformation = config["DEFORMATION"]
# folders = config["FOLDERS"]

# General
image_sample_folder = pathlib.Path(general.get("image_sample_folder"))
force_cpu = general.getboolean("force_cpu")
image_size = general.getint("image_size")
pattern_size = general.getint("pattern_size")
siemens_star_branches = general.getint("siemens_star_branches")
siemens_star_template = general.get("siemens_star_template")
siemens_star_fillers = general.get("siemens_star_fillers")

# Noise analysis
noise_values = eval(noise_analysis.get("noise_values"))
keep_split_measurements = noise_analysis.getboolean("keep_split_measurements")
save_images = noise_analysis.getboolean("save_images")
save_tensors = noise_analysis.getboolean("save_tensors")
reconstruction_folder = pathlib.Path(noise_analysis.get("reconstruction_folder"))
save_reconstruction_template = noise_analysis.get("save_reconstruction_template")
save_reconstruction_fillers = noise_analysis.get("save_reconstruction_fillers")
save_metric_template = noise_analysis.get("save_metric_template")
save_metric_fillers = noise_analysis.get("save_metric_fillers")
dcnet_unet = noise_analysis.get("dcnet_unet")
pinvnet_unet = noise_analysis.get("pinvnet_unet")
nn_to_use = noise_analysis.get("nn_to_use")

# Deformation
torch_seed = deformation.getint("torch_seed")
displacement_magnitude = deformation.getint("displacement_magnitude")
displacement_smoothness = deformation.getint("displacement_smoothness")
frame_generation_period = deformation.getint("frame_generation_period")
deformation_mode = deformation.get("deformation_mode")
compensation_mode = deformation.get("compensation_mode")
deformation_model_template = deformation.get("deformation_model_template")
deformation_model_fillers = deformation.get("deformation_model_fillers")

# Folders
# siemens_star_folder = pathlib.Path(folders.get("siemens_star"))
# deformation_model_folder = pathlib.Path(folders.get("deformation_model"))
# nn_models_folder = pathlib.Path(folders.get("nn_models"))
# subsampling_reconstruction_folder = pathlib.Path(folders.get("noise_reconstruction"))


# %%
# Derived parameters
# =============================================================================
n_measurements = pattern_size**2
n_frames = 2 * n_measurements
frames = n_frames  # this is used for the deformation field import
n_noise_values = len(noise_values)

valid_nn_names = [
    valid_name for valid_name in re.split(r"\W", nn_to_use) if len(valid_name) > 0
]
valid_nn_names.append(None)
n_neural_networks = len(valid_nn_names)
neural_network_linestyles = ["--", ":", "-"]
assert len(neural_network_linestyles) == len(valid_nn_names)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() and not force_cpu else "cpu"
)
print(f"Using device: {device}")
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

PSNRs = torch.zeros(n_neural_networks, n_noise_values)
SSIMs = torch.zeros(n_neural_networks, n_noise_values)

for i in range(n_noise_values):
    noise_alpha = noise_values[i]
    print("Computing PSNR and SSIM for noise value", noise_alpha)

    for j, neural_network_name in enumerate(valid_nn_names):

        reconstruction_name = save_reconstruction_template.format(
            *eval(save_reconstruction_fillers)
        )
        reconstruction_path = reconstruction_folder / reconstruction_name

        PSNRs[j, i] = PSNR_from_file(reconstruction_path, x, center_crop=64)
        SSIMs[j, i] = SSIM_from_file(reconstruction_path, x, center_crop=64)

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
# subsampling_ratio = subsamplings  # .flip(0)  # flip so that we have 1 at the left
PSNRs_ = PSNRs  # .flip(0)  # flip the rest
SSIMs_ = SSIMs  # .flip(0)

# put the PSNRs and SSIMs in 2 subsplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

# PSNR
axs[0].set_xlabel("Poisson noise coefficient")

axs[0].set_ylabel("PSNR", color="tab:blue")
axs[0].tick_params(axis="y", labelcolor="tab:blue")

for j, neural_network_name in enumerate(valid_nn_names):
    axs[0].plot(
        noise_values,
        PSNRs[j, :],
        label=str(neural_network_name),
        color="tab:blue",
        linestyle=neural_network_linestyles[j],
        lw=0.75,
        marker="+",
        # alpha=0.65,
    )

    axs[0].legend(title="Neural network")

    axs[0].grid(axis="both")


# SSIM
axs[1].set_xlabel("Poisson noise coefficient")

axs[1].set_ylabel("SSIM", color="tab:red")
axs[1].tick_params(axis="y", labelcolor="tab:red")

for j, neural_network_name in enumerate(valid_nn_names):
    axs[1].plot(
        noise_values,
        SSIMs[j, :],
        label=str(neural_network_name),
        color="tab:red",
        linestyle=neural_network_linestyles[j],
        lw=0.75,
        marker="+",
        # alpha=0.65,
    )

    axs[1].legend(title="Neural network")

    axs[1].grid(axis="both")

plt.suptitle("Reconstruction quality with noise")
plt.show()

# %%

# ax2 = ax.twinx()
# ax.set_xlabel("Poisson noise coefficient")
# ax.set_ylabel("PSNR", color="tab:blue")
# ax.tick_params(axis="y", labelcolor="tab:blue")
# ax2.set_ylabel("SSIM", color="tab:red")
# ax2.tick_params(axis="y", labelcolor="tab:red")

# for j, neural_network_name in enumerate(valid_nn_names):
#     ax.plot(
#         noise_values,
#         PSNRs[j, :],
#         # label=neural_network_name,
#         color="tab:blue",
#         linestyle=neural_network_linestyles[j],
#         lw=0.75,
#         marker="+",
#         alpha=0.65,
#     )
#     ax2.plot(
#         noise_values,
#         SSIMs[j, :],
#         # label=neural_network_name,
#         color="tab:red",
#         linestyle=neural_network_linestyles[j],
#         lw=0.75,
#         marker="+",
#         alpha=0.65,
#     )

#     # make a fake plot to add the legend in black
#     ax.plot(
#         [],
#         [],
#         color="black",
#         linestyle=neural_network_linestyles[j],
#         label=str(neural_network_name),
#     )

#     ax.legend()
#     # ax2.legend()

# # add a title to the legend
# ax.legend(title="Neural network")

# # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.0 / subsampling_steps))
# # ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0 / subsampling_steps))
# # ax.xaxis.set_major_formatter(ticker.FuncFormatter(fractions(subsampling_steps / 2)))
# # ax.set_xlim(min(noise_values), max(noise_values))
# ax.grid(axis="both")

# plt.title("Reconstruction quality with noise")
# plt.show()

# %%
