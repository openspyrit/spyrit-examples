"""
This script is used to test the dynamic module and dynamic reconstruction in
the case of a deformation equal to the identity.
"""

# %%

import pathlib

import math
import torch
import torchvision

import spyrit.core.meas as meas
import spyrit.core.prep as prep
import spyrit.core.warp as warp
import spyrit.misc.disp as disp
import spyrit.misc.statistics as stats


# %%
## PARAMETERS
# =============================================================================
# image and measurements
image_size = 64  # 80 or 128 or 160
pattern_size = 64  # 64 or 128

# time parameters
t0 = 0  # initial time
tf = 2  # final time

# deformation field
deform_mode = "bilinear"  # choose between 'bilinear' and 'bicubic'
compensation_mode = "bilinear"  # choose between 'bilinear' and 'bicubic'

# reconstruction
reg = "H1"
eta = 1e-6

# use gpu ?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # force cpu
print(f"Using device: {device}")
# =============================================================================


# %%
## Get the image
imgs_path = pathlib.Path(r"reference_images/")
transform = stats.transform_gray_norm(img_size=image_size)
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=7)
x, _ = next(iter(dataloader))

# Select the `i`-th image in the batch
i = 1  # Image index (modify to change the image)
x = x[i : i + 1, :, :, :]
x = x.detach().clone()
b, c, h, w = x.shape
disp.imagesc(x[0, 0, :, :], title="Original image")
print(f"Shape of input image: {x.shape}")

# %%
## Define operators

# square subsampling strategy: define order of measurements
order = torch.tensor(
    [
        [min(i, j) for i in range(pattern_size, 0, -1)]
        for j in range(pattern_size, 0, -1)
    ],
    dtype=torch.int32,
)

n_measurements = pattern_size**2
meas_op = meas.DynamicHadamSplit(
    n_measurements, pattern_size, order, (image_size, image_size)
)
prep_op = prep.DirectPoisson(1, meas_op)
meas_op.to(device)
prep_op.to(device)


## Define the deformation field
def f(t):
    return torch.eye(3, dtype=torch.float64)


time_vector = torch.linspace(t0, tf, 2 * n_measurements)
deformation = warp.AffineDeformationField(f, time_vector, (image_size, image_size))
deformation.to(device)
meas_op.build_H_dyn(deformation, mode=compensation_mode)
x = x.to(device)

warped_images = deformation(x, mode=deform_mode)
print("Shape of warped images:", warped_images.shape)
print("Shape of H_dyn:", meas_op.H_dyn.shape)


# %%
## Measure & reconstruct
# measure
measurements = meas_op(warped_images)
print("Shape of measurements:", measurements.shape)

# reconstruct
x_hat = meas_op.pinv(measurements, reg=reg, eta=eta, diff=False)
print("Shape of reconstructed image:", x_hat.shape)

disp.imagesc(x_hat[0, 0, :, :], title=f"Reconstructed image, {eta=}")
disp.imagesc(x_hat[0, 0, :, :] - x.cpu()[0, 0, :, :], title="Difference image")

# %%
