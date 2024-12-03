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

import torch
import torchvision
import matplotlib.pyplot as plt

import spyrit.misc.disp as disp
import spyrit.core.meas as meas
import spyrit.core.nnet as nnet
import spyrit.core.prep as prep
import spyrit.core.noise as noise
import spyrit.core.recon as recon
import spyrit.core.torch as spytorch
from spyrit.misc.load_data import download_girder


# %%
# READ PARAMETERS
config = configparser.ConfigParser()
config.read("config.ini")

general = config["GENERAL"]
noise_analysis = config["NOISE_ANALYSIS"]
deformation = config["DEFORMATION"]
folders = config["FOLDERS"]

# General
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
generated_frame_period = deformation.getint("generated_frame_period")
deformation_mode = deformation.get("deformation_mode")
compensation_mode = deformation.get("compensation_mode")
deformation_model_template = deformation.get("deformation_model_template")
deformation_model_fillers = deformation.get("deformation_model_fillers")

# Folders
siemens_star_folder = pathlib.Path(folders.get("siemens_star"))
deformation_model_folder = pathlib.Path(folders.get("deformation_model"))
nn_models_folder = pathlib.Path(folders.get("nn_models"))
subsampling_reconstruction_folder = pathlib.Path(folders.get("noise_reconstruction"))


# %%
# Derived parameters
# =============================================================================

n_measurements = pattern_size**2
n_frames = 2 * n_measurements
frames = n_frames  # this is used for the deformation field import
n_noise_values = len(noise_values)

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
# Load the deformation field
deform_load_name = deformation_model_template.format(*eval(deformation_model_fillers))
ElasticDeform = torch.load(
    deformation_model_folder / f"{deform_load_name}.pt", weights_only=False
).to(device)
# warp the image
warped_image = ElasticDeform(x, mode=deformation_mode)

# define measurement operator
meas_op = meas.DynamicHadamSplit(
    n_measurements,
    pattern_size,
    measurements_variance,
    (image_size, image_size),
).to(device)
# build the dynamic measurement operator
meas_op.build_H_dyn(ElasticDeform, mode=compensation_mode)

# build the noise operator and preprocessing operator, we will update the noise
# value in the loop
noise_op = noise.Poisson(meas_op, 1).to(device)
if keep_split_measurements:
    prep_op = prep.DirectPoisson(1, meas_op).to(device)
else:
    prep_op = prep.SplitPoisson(1, meas_op).to(device)
    # update the value of the H_dyn matrix in the measurement operator
    meas_op.H_dyn = meas_op.H[::2, :] - meas_op.H[1::2, :]

# build the tikho reconstructor
tikho = recon.Tikhonov(meas_op, image_covariance, False).to(device)


# %%
# Load the neural networks
valid_nn_names = [
    valid_name for valid_name in re.split(r"\W", nn_to_use) if len(valid_name) > 0
]
nn_dict = {name: nnet.Unet().to(device) for name in valid_nn_names}

for i, nn_name in enumerate(valid_nn_names):

    model_path = nn_models_folder / eval(nn_to_use)[i]
    print(f"Loading model `{nn_name}` at : {model_path}")
    nn_weights = torch.load(model_path, weights_only=True)

    # remove "denoi" prefix
    for key in list(nn_weights.keys()):
        nn_weights[key.removeprefix("denoi.")] = nn_weights.pop(key)

    nn_dict[nn_name].load_state_dict(nn_weights)
    nn_dict[nn_name].eval()

# add a None neural network for the case where no denoising is applied
valid_nn_names.append(None)


# %%
# 0. Update variable values
# 1. Update noise value in noise/prep operators
# 2. Measure the frames
# 3. Preprocess measurements
# 4. Reconstruct the images using Tikhonov class
# 5. Denoise if needed
# 6. Save the images as png and pt


for i in range(n_noise_values):
    print(f"Noise value {i+1}/{n_noise_values}")
    # 0.
    noise_alpha = noise_values[i]
    # 1.
    noise_op.alpha = noise_alpha
    prep_op.alpha = noise_alpha
    # 2.
    measurements = noise_op(warped_image)
    # 3.
    prep_measurements = prep_op(measurements)
    var_measurements = prep_op.sigma(measurements)
    # 4
    x_hat_tikho = tikho(prep_measurements, var_measurements)
    # 5.
    for neural_network_name in valid_nn_names:
        if neural_network_name is None:
            x_hat = x_hat_tikho
        else:
            x_hat = (
                nn_dict[neural_network_name](x_hat_tikho.unsqueeze(0))
                .squeeze(0)
                .detach()
            )
        # 6.
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
