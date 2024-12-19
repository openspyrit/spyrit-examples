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
config.read("../../config.ini")

general = config["GENERAL"]
noise_analysis = config["NOISE_ANALYSIS"]
deformation = config["DEFORMATION"]
hyperparameter = config["HYPERPARAMETER"]

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

# Noise analysis
noise_values = [10, 100]  # eval(noise_analysis.get("noise_values"))
keep_split_measurements = noise_analysis.getboolean("keep_split_measurements")
save_images = True  # noise_analysis.getboolean("save_images")
save_tensors = True  # noise_analysis.getboolean("save_tensors")
reconstruction_folder = pathlib.Path(__file__).parent.resolve() / "output"
save_reconstruction_template = noise_analysis.get("save_reconstruction_template")
save_reconstruction_fillers = noise_analysis.get("save_reconstruction_fillers")

# use any neural networks ?
nn_to_use = hyperparameter.get("nn_to_use")

# Deformation
torch_seed = deformation.getint("torch_seed")
displacement_magnitude = deformation.getint("displacement_magnitude")
displacement_smoothness = deformation.getint("displacement_smoothness")
frame_generation_period = deformation.getint("frame_generation_period")
deformation_mode = deformation.get("deformation_mode")
compensation_mode = deformation.get("compensation_mode")
save_deformation_template = deformation.get("save_deformation_template")
save_deformation_fillers = deformation.get("save_deformation_fillers")


# %%
# Derived parameters
# =============================================================================
pathlib.Path(reconstruction_folder).mkdir(parents=True, exist_ok=True)

n_measurements = pattern_size**2
n_frames = 2 * n_measurements
frames = n_frames  # this is used for the deformation field import
n_noise_values = len(noise_values)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() and not force_cpu else "cpu"
)
print(f"Using device: {device}")

valid_nn_names = [
    valid_name for valid_name in re.split(r"\W", nn_to_use) if len(valid_name) > 0
]
nn_dict = {name: nnet.Unet().to(device) for name in valid_nn_names}
# add identity neural network
valid_nn_names.append("identity")
nn_dict["identity"] = nnet.Identity().to(device)
# =============================================================================


# %%
# Load the image
image_name = siemens_star_template.format(*eval(siemens_star_fillers))
image_path = image_sample_folder / image_name

x = torchvision.io.read_image(
    image_sample_folder / image_name, torchvision.io.ImageReadMode.GRAY
)
x = x.detach().float().to(device)
# rescale x from [0, 255] to [-1, 1]
x = 2 * (x / 255) - 1
c, h, w = x.shape

print(f"Shape of input image: {x.shape}")


# %%
# Get measurements covariance matrix / image covariance matrix
file_abs_path = stats_folder / "Cov_64x64.pt"
measurements_covariance = torch.load(file_abs_path, weights_only=True)
print("Measurement covariance matrix loaded")
# get variance from covariance
measurements_variance = spytorch.Cov2Var(measurements_covariance)

file_abs_path = stats_folder / "Image_Cov_8_128x128.pt"
image_covariance = torch.load(file_abs_path, weights_only=True).to(device)
print("Image covariance matrix loaded")


# %%
# Load the deformation field
deform_load_name = save_deformation_template.format(*eval(save_deformation_fillers))
deform_load_path = deformation_folder / deform_load_name
ElasticDeform = torch.load(deform_load_path, weights_only=False).to(device)
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
tikho_identity = recon.Tikhonov(
    meas_op, torch.eye(image_size**2, device=device), False
).to(device)


# %%
# 0. Update variable values
# 1. Update noise value in noise/prep operators
# 2. Measure the frames
# 3. Preprocess measurements
# 4. Reconstruct the images using Tikhonov class
# 5. Denoise if needed
# 6. Save the images as png and pt

noise_scale_values = [0] + [float(10**pow) for pow in [-10, -3, -2, -1, 0, 1, 2, 3, 4]]

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

    for nsv in noise_scale_values:
        # 4
        # artificially change the noise scale
        var_measurements *= nsv
        # use the tikho_identity if the noise scale is 0
        if nsv == 0:
            x_hat_tikho = tikho_identity(prep_measurements, var_measurements)
        else:
            x_hat_tikho = tikho(prep_measurements, var_measurements)
        x_hat_tikho = x_hat_tikho.reshape(1, c, h, w)

        # 5.
        for neural_network_name, nn_model in nn_dict.items():
            x_hat = nn_model(x_hat_tikho)

        # 6.
        reconstruction_name = (
            save_reconstruction_template.format(*eval(save_reconstruction_fillers))
            + f"_noisescale_{nsv}"
        )
        if save_images:
            plt.imsave(
                reconstruction_folder / f"{reconstruction_name}.png",
                x_hat[0, 0, :, :].cpu(),
                cmap="gray",
            )
            print(
                "Reconstructed image saved in",
                reconstruction_folder / f"{reconstruction_name}.png",
            )
        if save_tensors:
            torch.save(x_hat, reconstruction_folder / f"{reconstruction_name}.pt")
            print(
                "Reconstructed tensor saved in",
                reconstruction_folder / f"{reconstruction_name}.pt",
            )

# %%
