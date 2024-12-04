"""Script that analyses the superresolution effect and its limitation. 

Superresolution happens when a moving object is mesured at different times,
leading to a higher resolution than the one of the camera. A Siemens star is
used.

An original image of 80x80 pixels is used, and then measured at different
scales using subsampling at 64x64 resolution (and not directly at a 32x32
resolution for instance). The reconstructed image is thus at 80x80 resolution.

This simulation is done using a fixed size image (80x80 pixels) and simulating
its movement across multiple frames. The number of frames depends on the number
of measurements, so that the first and last frame are always identical (i.e.
represent the same overall movement, but with a different sampling rate). See
_fig2 for a simulation with a finer image, deformed and then subsampled to
the measurement size.
"""

# %%

import pathlib

import math
import torch
import torchvision
import matplotlib.pyplot as plt

import spyrit.core.meas as meas
import spyrit.core.warp as warp
import spyrit.core.torch as spytorch
import spyrit.misc.disp as disp
from spyrit.misc.load_data import download_girder
from aux_functions import save_params


# %%
## PARAMETERS
# =============================================================================
# image and measurements
image_size = 80  # reconstruction size
n_branches = 16  # number of branches in the Siemens star
base_pattern_size = 64
pattern_sizes = [64, 32, 16, 8, 4]  # pattern sizes

# reconstruction parameters
reg = "H1"  # which regularization to use
eta = 1e2  # regularization parameter

# time parameters
t0 = 0  # initial time
tf = 2  # final time

# deformation field
a = 0.2  # amplitude
omega = 2 * math.pi  # [rad/s] frequency
deform_mode = "bilinear"  # choose between 'bilinear' and 'bicubic'
compensation_mode = "bilinear"  # choose between 'bilinear' and 'bicubic'

# where to load the reference image from
load_path = pathlib.Path(r"C:\Users\phan\Documents\SPYRIT\deep_dyn_recon\reference")
# relative path to save the images
save_path = pathlib.Path(
    r"C:\Users\phan\Documents\SPYRIT\deep_dyn_recon\2_superresolution\fig_1"
)
# models' absolute root paths
model_paths = pathlib.Path(
    r"C:\Users\phan\Documents\Spyrit archive\optica2024\spyrit-examples\2024_spyrit\model"
)

# use gpu ?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # cpu is faster
print(f"Using device: {device}")
# =============================================================================
save_name = "params.txt"
save_params(
    save_path / save_name,
    image_size=image_size,
    base_pattern_size=base_pattern_size,
    pattern_sizes=pattern_sizes,
    reg=reg,
    eta=eta,
    t0=t0,
    tf=tf,
    a=a,
    omega=omega,
    deform_mode=deform_mode,
    compensation_mode=compensation_mode,
    save_path=save_path,
    model_paths=model_paths,
    device=device,
)


# %%
# Load the image
image_name = f"siemens_star_{image_size}x{image_size}_{n_branches}_branches.png"
image_path = load_path / image_name

if not image_path.exists():
    from aux_functions import save_siemens_star

    print(f"Image {image_name} not found in {load_path}, creating it")
    save_siemens_star(image_path, figsize=image_size, n=n_branches)

x = torchvision.io.read_image(load_path / image_name, torchvision.io.ImageReadMode.GRAY)

# rescale x from [0, 255] to [-1, 1]
x = x.detach().float()
x = 2 * (x / 255) - 1
c, h, w = x.shape

print(f"Shape of input image: {x.shape}")
disp.imagesc(x[0, :, :], title="Original image")


# %%
# Get measurements covariance matrix

url = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
dataId = "672b80acf03a54733161e973"  # different ID than tuto6 but same matrix
data_folder = "./stat/"
cov_name = "Cov_64x64.pt"

# download
file_abs_path = download_girder(url, dataId, data_folder, cov_name)
measurements_covariance = torch.load(file_abs_path, weights_only=True)
print(f"Cov matrix {cov_name} loaded")

measurements_variance = spytorch.Cov2Var(measurements_covariance)


# %%
# Define subsampling strategy and number of measurements in each case

# subsampling strategy: define order of measurements
orders = {ps: torch.zeros_like(measurements_variance) for ps in pattern_sizes}
for ps in pattern_sizes:
    orders[ps][:ps, :ps] = measurements_variance[:ps, :ps]
# number of measurements
n_measurements = {ps: ps**2 for ps in pattern_sizes}

# define measurement operator
meas_ops = {
    ps: meas.DynamicHadamSplit(
        n_measurements[ps], base_pattern_size, orders[ps], (image_size, image_size)
    )
    for ps in pattern_sizes
}


# %%
## Define the deformation field
def s(t):
    return 1 + a * math.sin(omega * t)


def f(t):
    return torch.tensor(
        [[s(t), 0, 0], [0, 1 / s(t), 0], [0, 0, 1]], dtype=torch.float64
    )


time_vectors = {
    ps: torch.linspace(t0, tf, 2 * n_measurements[ps]) for ps in pattern_sizes
}
deformations = {
    ps: warp.AffineDeformationField(f, time_vectors[ps], (image_size, image_size))
    for ps in pattern_sizes
}


# %%
# Deform the image and measure
warped_images = {ps: deformations[ps](x, mode=deform_mode) for ps in pattern_sizes}
measurements = {ps: meas_ops[ps](warped_images[ps]) for ps in pattern_sizes}
print("Shape of warped images:", warped_images[pattern_sizes[0]].shape)
print("Shape of dynamic measurements:", measurements[pattern_sizes[0]].shape)


# %%
# Build dynamic measurement operators
for ps in pattern_sizes:
    meas_ops[ps].build_H_dyn(deformations[ps], mode=compensation_mode)
# reconstruct
reconstructed_images = {
    ps: meas_ops[ps].pinv(measurements[ps], reg=reg, eta=eta) for ps in pattern_sizes
}


# %%
# plot the images

fig, axs = plt.subplots(
    1, len(pattern_sizes) + 1, figsize=(5 * len(pattern_sizes) + 1, 5)
)
axs[0].imshow(x[0, :, :], cmap="gray")
axs[0].add_patch(
    plt.Rectangle(
        ((image_size - base_pattern_size) // 2, (image_size - base_pattern_size) // 2),
        base_pattern_size,
        base_pattern_size,
        edgecolor="blue",
        facecolor="none",
    )
)
axs[0].set_title("Original image\n")
for i, ps in enumerate(pattern_sizes):
    axs[i + 1].imshow(reconstructed_images[ps][0, :, :], cmap="gray")
    # show the pattern size
    axs[i + 1].add_patch(
        plt.Rectangle(
            (
                (image_size - base_pattern_size) // 2,
                (image_size - base_pattern_size) // 2,
            ),
            base_pattern_size,
            base_pattern_size,
            edgecolor="blue",
            facecolor="none",
        )
    )
    axs[i + 1].set_title(f"Reconstructed image\npattern size {ps}")

# save them in the folder
plt.imsave(save_path / "original_image.png", x[0, :, :], cmap="gray")
for i, ps in enumerate(pattern_sizes):
    plt.imsave(
        save_path / f"reconstructed_image_{ps}.png",
        reconstructed_images[ps][0, :, :],
        cmap="gray",
    )


# %%
# Search for the resolution depending on the angle
# first for pattern size = 64

radiuses = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35])
r = radiuses[3]
n_pts = 500  # 4 * (2*r + 1) # upper bound of number of points
angles = torch.linspace(0.0, 2 * math.pi, (n_pts + 1))[:-1]
center = torch.tensor([(image_size - 1) // 2, (image_size - 1) // 2])


coordinates = torch.zeros((n_pts, 2))
coordinates[:, 0] = r * torch.cos(angles)
coordinates[:, 1] = r * torch.sin(angles)
# add the center
coordinates += center

coordinates = torch.round(coordinates).to(torch.int32)
coordinates = torch.unique_consecutive(coordinates, dim=0)  # remove duplicates
# remove possible first and last duplicates
if coordinates[0, :].equal(coordinates[-1, :]):
    coordinates = coordinates[:-1, :]

re_angles = (
    torch.atan2(
        coordinates[:, 1] - (image_size - 1) // 2,
        coordinates[:, 0] - (image_size - 1) // 2,
    )
    * 180
    / math.pi
) % 360
# values at the coordinates
values = reconstructed_images[32][0, coordinates[:, 0], coordinates[:, 1]]
# plot the values
fig, ax = plt.subplots()
ax.plot(re_angles, values)
ax.set_title(f"Values at radius {r}")
ax.set_xlabel("Angle [rad]")
ax.set_ylabel("Value")
# set min and max y values to 1.1 and -1.1
ax.set_ylim(-1.1, 1.1)
ax.set_xlim(0, 360)
plt.show()


print(coordinates)
print(re_angles)


# print(coordinates)
# %%
