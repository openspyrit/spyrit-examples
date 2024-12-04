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

import torch
import torchvision
import matplotlib.pyplot as plt

import spyrit.misc.disp as disp
import spyrit.core.meas as meas
import spyrit.core.recon as recon
import spyrit.core.torch as spytorch
from spyrit.misc.load_data import download_girder


# %%
# READ PARAMETERS
config = configparser.ConfigParser()
config.read("config.ini")

general = config["GENERAL"]
subsampling_analysis = config["SUBSAMPLING_ANALYSIS"]
deformation = config["DEFORMATION"]
save = config["SAVE"]
folders = config["FOLDERS"]

# General
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
save_reconstruction_template = subsampling_analysis.get("save_reconstruction_template")
save_reconstruction_fillers = subsampling_analysis.get("save_reconstruction_fillers")

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
siemens_star_folder = pathlib.Path(folders.get("siemens_star"))
deformation_model_folder = pathlib.Path(folders.get("deformation_model"))
nn_models_folder = pathlib.Path(folders.get("nn_models"))
subsampling_reconstruction_folder = pathlib.Path(
    folders.get("subsampling_reconstruction")
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
# Load the image
image_name = siemens_star_template.format(*eval(siemens_star_fillers))
image_path = siemens_star_folder / image_name

if not image_path.exists():
    from aux_functions import save_siemens_star

    print(f"Image {image_name} not found in {siemens_star_folder}, creating it")
    save_siemens_star(image_path, figsize=image_size, n=siemens_star_branches)

x = torchvision.io.read_image(
    siemens_star_folder / image_name, torchvision.io.ImageReadMode.GRAY
)
x = x.detach().float().to(device)
# rescale x from [0, 255] to [-1, 1]
x = 2 * (x / 255) - 1
c, h, w = x.shape

print(f"Shape of input image: {x.shape}")
disp.imagesc(x[0, :, :].cpu(), title="Original image")


# %%
# Get measurements covariance matrix / image covariance matrix
url = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
data_folder = "./stat/"
meas_cov_Id = "672b80acf03a54733161e973"  # different ID than tuto6 but same matrix
image_cov_Id = "67486be8a438ad25e7a001f7"
meas_cov_name = "Cov_64x64.pt"
image_cov_name = "Image_Cov_8_128x128.pt"

# download
file_abs_path = download_girder(url, meas_cov_Id, data_folder, meas_cov_name)
measurements_covariance = torch.load(file_abs_path, weights_only=True)
print(f"Cov matrix {meas_cov_name} loaded")
file_abs_path = download_girder(url, image_cov_Id, data_folder, image_cov_name)
image_covariance = torch.load(file_abs_path, weights_only=True).to(device)
print("Image covariance matrix loaded")
# get variance from covariance
measurements_variance = spytorch.Cov2Var(measurements_covariance)


# %%
# Load the deformation field, warp the image
deform_load_name = deformation_model_template.format(*eval(deformation_model_fillers))
deformation_model_path = deformation_model_folder / deform_load_name

# if the field does not exist, generate it
if not deformation_model_path.exists():
    from aux_functions import save_elastic_deformation

    save_elastic_deformation()
    # now it should exist

ElasticDeform = torch.load(
    deformation_model_folder / deform_load_name, weights_only=False
).to(device)
warped_image = ElasticDeform(x, mode=deformation_mode)


# %%
# 0. Update variable values
# 1. Build the measurement operator
# 2. Measure the frames
# 3. Build the dynamic measurement operator
# 4. Reconstruct the images using Tikhonov class
# 5. Save the images as png and pt

for i in range(subsampling_steps):
    print(f"Subsampling {i+1}/{subsampling_steps}")
    # 0.
    n_measurements = n_measurements_list[i]
    n_frames = n_frames_list[i]
    # 1.
    meas_op = meas.DynamicHadamSplit(
        n_measurements,
        pattern_size,
        measurements_variance,
        (image_size, image_size),
    ).to(device)
    # 2.
    warped_image = warped_image[:n_frames, :, :, :]
    measurements = meas_op(warped_image)
    # 3.
    ElasticDeform.field = ElasticDeform.field[:n_frames, :, :, :]
    meas_op.build_H_dyn(ElasticDeform, mode=compensation_mode)
    # 3.5 we use the difference of the measurements, optional
    measurements = measurements[..., ::2] - measurements[..., 1::2]  # diff
    meas_op.H_dyn = meas_op.H[::2, :] - meas_op.H[1::2, :]  # diff
    # 4.
    tikho = recon.Tikhonov(meas_op, image_covariance, False).to(device)
    # no noise so variance is 0
    x_hat = tikho(
        measurements,
        torch.zeros(measurements.shape[-1], measurements.shape[-1], device=device),
    ).reshape(c, h, w)
    # 5. Save the images / tensors
    reconstruction_name = save_reconstruction_template.format(
        *eval(save_reconstruction_fillers)
    )

    if save_images:
        plt.imsave(
            subsampling_reconstruction_folder / f"{reconstruction_name}.png",
            x_hat[0, :, :].cpu(),
            cmap="gray",
        )
        print(
            "Reconstructed image saved in",
            subsampling_reconstruction_folder / f"{reconstruction_name}.png",
        )
    if save_tensors:
        torch.save(
            x_hat, subsampling_reconstruction_folder / f"{reconstruction_name}.pt"
        )
        print(
            "Reconstructed tensor saved in",
            subsampling_reconstruction_folder / f"{reconstruction_name}.pt",
        )

# %%
