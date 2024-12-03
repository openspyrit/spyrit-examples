"""Auxiliary functions for the Tikhonov dynamic reconstruction project.

Author: Romain Phan
Date: 2024-11-21
"""

import time

import torch
import torch.nn as nn
import torchmetrics.image as metrics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection


def save_params(path, **kwargs):
    with open(path, "w") as f:
        f.write(f"File created: {time.ctime()}\n\n")
        for key, value in kwargs.items():
            f.write(f"{key}: {value}\n")
    print(f"Parameters saved in {path}")


def save_siemens_star(path, figsize=64, n=8):
    """Saves a Siemens star in a file.

    The star has n black branches and n white branches and is saved as a square
    image with a white background.

    Args:
        path (str): path to save the file. Must contain the file extension.
        figsize (int, optional): Size of the figure side. Defaults to 64.
        n (int, optional): Number of black branches in the star. Defaults to 8.
    """
    # create a figure and axis with no frame
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    # remove all margins / make the plot fill the whole figure
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    # add the star to the axis
    ax.add_collection(gen_siemens_star((0, 0), 1, n))
    # set the limits of the axis and remove the axis
    plt.axis([-1.0, 1.0, -1.0, 1.0])
    ax.set_axis_off()
    # save the figure, close it and print a message
    fig.savefig(
        path,
        bbox_inches="tight",
        dpi=1,
        pad_inches=0,
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)
    print(f"Siemens star saved in {path}")


def gen_siemens_star(origin, radius, n):
    """Generates a siemens star with n black branches and n white branches.

    This function returns a PatchCollection object with n black branches and n white branches.

    Args:
        origin (tuple): coordinate of the center of the star.
        radius (float): radius of the star.
        n (int): number of branches of the star.

    Returns:
        matplotlib.collection.PatchCollection: PatchCollection object with n
        black branches and n white branches.
    """
    centres = np.linspace(0, 360, n + 1)[:-1]
    step = ((360.0) / n) / 4.0
    centres -= step
    patches = []
    for c in centres:
        patches.append(Wedge(origin, radius, c - step, c + step))
    return PatchCollection(patches, facecolors="k", edgecolors="none")


def downscale(img, out_resolution):
    """Downscale an image using the mean of the pixels in each block.

    This function takes an image and downscales it to the desired resolution
    by taking the mean of the pixels in each block.

    Args:
        img (torch.tensor): image to downscale.
        out_resolution (int): desired resolution of the output image.

    Returns:
        torch.tensor: downsampled image.
    """
    if img.dim() > 4:
        raise ValueError("The image must have at most 4 dimensions.")
    elif img.dim() == 2:
        img = img.unsqueeze(0)
        no_channel = True
    else:
        no_channel = False

    ratio = img.shape[-1] // out_resolution
    down = nn.AvgPool2d(kernel_size=ratio, stride=ratio)

    return down(img).squeeze(0) if no_channel else down(img)


def fractions(denominator):
    step = 1.0 / denominator

    # define the function that is returned
    def frac(x, pos):
        if np.isclose((x / step) % (1.0 / step), 0.0):
            return "{:.0f}".format(x)
        else:
            return "$\\frac{{{:2.0f}}}{{{:2.0f}}}$".format(x / step, 1.0 / step)

    return frac


def metric_from_file(estimate_path, reference_obj, metric, center_crop=None):
    estimate_obj = torch.load(estimate_path, weights_only=True)
    # check that everything is on same device
    if estimate_obj.device != reference_obj.device:
        estimate_obj = estimate_obj.to(reference_obj.device)
    metric = metric.to(reference_obj.device)

    if center_crop is not None:
        height, width = estimate_obj.shape[-2:]
        # indices to select the center of the image
        left = (width - center_crop) // 2
        right = left + center_crop
        bottom = (height - center_crop) // 2
        top = bottom + center_crop
        # crop both objects
        estimate_obj = estimate_obj[..., bottom:top, left:right]
        reference_obj = reference_obj[..., bottom:top, left:right]

    if estimate_obj.dim() == 2:
        estimate_obj = estimate_obj.unsqueeze(0)
    if estimate_obj.dim() == 3:
        estimate_obj = estimate_obj.unsqueeze(0)
    if reference_obj.dim() == 2:
        reference_obj = reference_obj.unsqueeze(0)
    if reference_obj.dim() == 3:
        reference_obj = reference_obj.unsqueeze(0)

    return metric(estimate_obj, reference_obj)


def PSNR_from_file(estimate_path, reference_obj, center_crop=None, *args, **kwargs):
    psnr = metrics.PeakSignalNoiseRatio(*args, **kwargs)
    return metric_from_file(estimate_path, reference_obj, psnr, center_crop)


def SSIM_from_file(estimate_path, reference_obj, center_crop=None, *args, **kwargs):
    ssim = metrics.StructuralSimilarityIndexMeasure(*args, **kwargs)
    return metric_from_file(estimate_path, reference_obj, ssim, center_crop)
