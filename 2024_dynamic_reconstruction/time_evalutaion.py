"""
Script to evaluate the time taken for the different steps of the dynamic
reconstruction process.
"""

"""
This script is used to test the dynamic module and dynamic reconstruction in
the case of a deformation equal to the identity.
"""

# %%

import time
import pathlib

import math
import torch

import spyrit.core.meas as meas
import spyrit.core.prep as prep
import spyrit.core.warp as warp


# %%
## PARAMETERS
# =============================================================================
# image and measurements
image_size = 128  # 80 or 128 or 160
pattern_size = 128  # 64 or 128

# noise parameters
noise_level = 10  # used for Poisson law
torch_seed = 0

# time parameters
ti = 0  # initial time
tf = 2  # final time

# deformation field
a = 0.2  # amplitude
omega = 2 * math.pi  # [rad/s] frequency
deform_mode = "bilinear"  # choose between 'bilinear' and 'bicubic'
compensation_mode = "bilinear"  # choose between 'bilinear' and 'bicubic'

# cooldown at the end of each device
cooldown = 60  # seconds
# =============================================================================


# %%
## Get the image
x = torch.randn(1, 1, image_size, image_size)
b, c, h, w = x.shape
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

subsampling_factor = 1
n_measurements = pattern_size**2 // subsampling_factor
meas_op = meas.DynamicHadamSplit(
    n_measurements, pattern_size, order, (image_size, image_size)
)
prep_op = prep.DirectPoisson(noise_level, meas_op)


## Define the deformation field
def s(t):
    return 1 + a * math.sin(omega * t)


def f(t):
    return torch.tensor(
        [[s(t), 0, 0], [0, 1 / s(t), 0], [0, 0, 1]], dtype=torch.float64
    )


time_vector = torch.linspace(ti, tf, 2 * n_measurements)
deformation = warp.AffineDeformationField(f, time_vector, (image_size, image_size))


# %%
# Send to device

device_list = ["cuda:0", "cpu"]
dict = {k: [] for k in device_list}

for d in device_list:
    device = torch.device(d)
    print(f"Using device: {device}")

    x = x.to(device)
    meas_op.to(device)
    prep_op.to(device)
    deformation.to(device)

    # Warping
    t0 = time.time()
    warped_images = deformation(x, mode=deform_mode)
    t1 = time.time()
    print("Shape of warped images:", warped_images.shape)

    # Measurements
    t2 = time.time()
    measurements = meas_op(warped_images)
    t3 = time.time()
    print("Shape of measurements:", measurements.shape)

    # H_dyn matrix
    t4 = time.time()
    meas_op.build_H_dyn(deformation, mode=compensation_mode)
    t5 = time.time()
    print("Shape of H_dyn:", meas_op.H_dyn.shape)

    # pinv using H1 regularization
    eta = 1e-6
    t6 = time.time()
    x_hat = meas_op.pinv(measurements, reg="H1", eta=eta, diff=False)
    t7 = time.time()
    print("Shape of reconstructed image:", x_hat.shape)

    # computing H_dyn_pinv
    t8 = time.time()
    meas_op.build_H_dyn_pinv(reg="H1", eta=eta)
    t9 = time.time()
    print("Shape of H_dyn_pinv:", meas_op.H_dyn_pinv.shape)

    # pinv using H_dyn_pinv
    t10 = time.time()
    x_hat_dyn = meas_op.pinv(measurements, diff=False)
    t11 = time.time()
    print("Shape of reconstructed image using H_dyn_pinv:", x_hat_dyn.shape)

    dict[d] = [t1 - t0, t3 - t2, t5 - t4, t7 - t6, t9 - t8, t11 - t10]

    if d != device_list[-1]:
        print(f"Cooldown for {cooldown} seconds...")
        time.sleep(cooldown)
        print("Done!")


# %%
# Save results in txt file

cwd = pathlib.Path.cwd()
datetime = time.strftime("%Y%m%d-%H%M%S")
save_name = f"time_report_{datetime}.txt"

print(f"Saving results in {cwd / save_name}...")

with open(cwd / save_name, "w") as f:
    l1 = "ACQUISITION PARAMETERS:\n"
    l2 = "-----------------------\n"
    l3 = f"Image_size: {image_size}\n"
    l4 = f"Pattern_size: {pattern_size}\n"
    l5 = f"Subsampling: {subsampling_factor}\n"
    l6 = f"Number of measurements: {n_measurements}\n"
    l7 = f"Noise level: {noise_level}\n"
    l8 = "\n"
    l9 = f"DEFORMATION PARAMETERS:\n"
    l10 = "-----------------------\n"
    l11 = f"Number of frames: {2*n_measurements}\n"
    l12 = f"Deformation mode: {deform_mode}\n"
    l13 = f"Compensation mode: {compensation_mode}\n"
    l14 = f"Initial time: {ti}\n"
    l15 = f"Final time: {tf}\n"
    l16 = f"Deformation amplitude: {a}\n"
    l17 = f"Deformation frequency: {omega}\n"
    l18 = "\n"
    l19 = f"TIME REPORT:\n"
    l20 = "------------\n"
    l21 = f"{'Device':<12} {'Warping':<12} {'Measurements':<12} {'H_dyn':<12} {'Pinv (H_dyn)':<12} {'H_dyn_pinv':<12} {'Pinv (H_dyn_pinv)':<12}\n"

    f.writelines(
        [
            l1,
            l2,
            l3,
            l4,
            l5,
            l6,
            l7,
            l8,
            l9,
            l10,
            l11,
            l12,
            l13,
            l14,
            l15,
            l16,
            l17,
            l18,
            l19,
            l20,
            l21,
        ]
    )
    for device, times in dict.items():
        f.write(
            f"{device:<12} {times[0]:<12.4f} {times[1]:<12.4f} {times[2]:<12.4f} {times[3]:<12.4f} {times[4]:<12.4f} {times[5]:<12.4f}\n"
        )

print("Done!")
