# %%
# Imports
# --------------------------------------------------------------------
import ast
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

import spyrit.core.meas as meas
import spyrit.core.prep as prep
import spyrit.core.noise as noise
import spyrit.core.recon as recon
import spyrit.misc.sampling as samp
import spyrit.external.drunet as drunet


# %%
# General
# --------------------------------------------------------------------
# Experimental data
data_folder = "data/"  # measurements
model_folder = "model/"  # reconstruction models
stat_folder = "stat/"  # statistics
recon_folder = "recon/supplemental_figure_S7/"  # reconstructed images

# Full paths
data_folder_full = Path.cwd() / Path(data_folder)
model_folder_full = Path.cwd() / Path(model_folder)
stat_folder_full = Path.cwd() / Path(stat_folder)
recon_folder_full = Path.cwd() / Path(recon_folder)
recon_folder_full.mkdir(parents=True, exist_ok=True)

# choose by name which experimental data to use
data_title = [
    "zoom_x12_usaf_group5",
    "zoom_x12_starsector",
    "tomato_slice_2_zoomx12",
    "tomato_slice_2_zoomx2",
]
suffix = {"data": "_spectraldata.npz", "metadata": "_metadata.json"}
n_meas = len(data_title)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
img_size = 128
subsampling_factor = 2


# %%
# Measurement (and reconstruction) operators
# ====================================================================
# number of measurements
M = (img_size // subsampling_factor) ** 2

# Measurement and noise operators
Ord_rec = np.ones((img_size, img_size))
Ord_rec[:, img_size // 2 :] = 0
Ord_rec[img_size // 2 :, :] = 0

meas_op = meas.HadamSplit(M, img_size, torch.from_numpy(Ord_rec))
noise_op = noise.Poisson(meas_op, 2)  # parameter alpha is unimportant here
prep_op = prep.SplitPoisson(2, meas_op)  # same here


# %%
# Load experimental data and prepare it for reconstruction
# ====================================================================
print("Looking for data in", data_folder_full)

# Collect data in numpy
exp_data = [
    np.load(data_folder_full / (title + suffix["data"]))["spectral_data"]
    for title in data_title
]
# Collect metadata
patterns = [[] for _ in range(n_meas)]
wavelengths = [[] for _ in range(n_meas)]
for ii, title in enumerate(data_title):
    file = open(data_folder_full / (title + suffix["metadata"]), "r")
    json_metadata = json.load(file)[3]
    file.close()
    patterns[ii] = ast.literal_eval(json_metadata["patterns"])
    wavelengths[ii] = ast.literal_eval(json_metadata["wavelengths"])


# %%
# Reorder measurements to match with the reconstruction order
# ====================================================================
acq_size = img_size // subsampling_factor
Ord_acq = [
    (-np.array(patterns[i])[::2] // 2).reshape((acq_size, acq_size))
    for i in range(n_meas)
]

# %%
# Define the two permutation matrices used to reorder the measurements
# measurement order -> natural order -> reconstruction order
Perm_rec = samp.Permutation_Matrix(Ord_rec)
Perm_acq = [samp.Permutation_Matrix(Ord_acq[i]).T for i in range(n_meas)]
# each element of 'measurements' has shape (measurements, wavelengths)
measurements = [samp.reorder(exp_data[i], Perm_acq[i], Perm_rec) for i in range(n_meas)]


# %%
# Reconstruct using a single spectral slice
# ====================================================================
lambda_select = 579.0970
measurements_slice = [np.zeros((2 * M, 1)) for _ in range(n_meas)]

# select the measurements in the right spectral slice
for i in range(n_meas):

    lambda_index = wavelengths[i].index(lambda_select)
    # take only the first 2*M measurements of the right wavelength
    measurements_slice[i] = measurements[i][: 2 * M, lambda_index].reshape((1, 2 * M))
    measurements_slice[i] = torch.from_numpy(measurements_slice[i]).to(
        device, dtype=torch.float32
    )


# %%
# Pinv - PnP
# ====================================================================
model_name = "drunet_gray.pth"

# in order: USAF target, starsector, tomato x12, tomato x2
nu_list = [
    [20, 30, 40],  # USAF target
    [35, 45, 55],  # starsector
    [30, 40, 50],  # tomato x12
    [30, 40, 50],  # tomato x2
]

# Initialize network
denoi = drunet.DRUNet()
pinvnet = recon.PinvNet(noise_op, prep_op, denoi)
pinvnet.eval()

# load_net(model_folder_full / model_name, pinvnet, device, False)
pinvnet.denoi.load_state_dict(
    torch.load(model_folder_full / model_name, weights_only=True), strict=False
)
pinvnet.denoi.eval()
pinvnet = pinvnet.to(device)

with torch.no_grad():
    for ii, y in enumerate(measurements_slice):
        # set noise level for measurement operator and PnP denoiser
        pinvnet.prep.set_expe()

        # iterate over noise levels
        for nu in nu_list[ii]:
            pinvnet.denoi.set_noise_level(nu)
            x_pinvnet = pinvnet.reconstruct_expe(y)

            filename = (
                f"{data_title[ii]}_{M}_{img_size}_pinv-net_drunet_nlevel_{nu}.png"
            )
            full_path = recon_folder_full / filename
            plt.imsave(
                full_path, x_pinvnet[0, 0, :, :].cpu().detach().numpy(), cmap="gray"
            )


# %%
