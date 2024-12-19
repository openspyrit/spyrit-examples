import pathlib
import configparser

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection

import spyrit.core.warp as warp


def save_elastic_deformation(configfile_path):

    # READ PARAMETERS
    config = configparser.ConfigParser()
    config.read(configfile_path)

    general = config["GENERAL"]
    deformation = config["DEFORMATION"]

    # General
    force_cpu = general.getboolean("force_cpu")
    image_size = general.getint("image_size")
    pattern_size = general.getint("pattern_size")
    deformation_folder = pathlib.Path(general.get("deformation_folder"))

    # Deformation
    torch_seed = deformation.getint("torch_seed")
    displacement_magnitude = deformation.getint("displacement_magnitude")
    displacement_smoothness = deformation.getint("displacement_smoothness")
    frame_generation_period = deformation.getint("frame_generation_period")
    save_deformation_template = deformation.get("save_deformation_template")
    save_deformation_fillers = deformation.get("save_deformation_fillers")

    # Derived parameters
    image_shape = (image_size, image_size)
    n_measurements = pattern_size**2
    n_frames = 2 * n_measurements
    frames = n_frames  # this is used for the deformation field export name

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not force_cpu else "cpu"
    )
    save_name = save_deformation_template.format(*eval(save_deformation_fillers))
    save_path = deformation_folder / save_name

    # if file exists, do nothing
    if save_path.exists():
        print(f"File already exists at {save_path}")
        return

    print(f"Saving on device: {device}")
    print("Saving Elastic Deformation in :", save_path)

    # Create the deformation field and save it
    torch.manual_seed(torch_seed)
    Elastic = warp.ElasticDeformation(
        displacement_magnitude,
        displacement_smoothness,
        image_shape,
        n_frames,
        frame_generation_period,
    ).to(device)

    torch.save(Elastic, save_path)
    print("Done")


def save_siemens_star(configfile_path):
    """Saves a Siemens star in an image file.

    The star has n black branches and n white branches and is saved as a square
    image with a white background.

    Args:
        path (str): path to save the file. Must contain the file extension.

        figsize (int, optional): Size of the figure side. Defaults to 64.

        n (int, optional): Number of black branches in the star. Defaults to 8.
    """
    config = configparser.ConfigParser()
    config.read(configfile_path)
    general = config["GENERAL"]
    # get parameters
    image_size = general.getint("image_size")
    siemens_star_branches = general.getint("siemens_star_branches")

    # get the save path
    image_sample_folder = pathlib.Path(general.get("image_sample_folder"))
    siemens_star_template = general.get("siemens_star_template")
    siemens_star_fillers = general.get("siemens_star_fillers")
    image_name = siemens_star_template.format(*eval(siemens_star_fillers))
    image_path = image_sample_folder / image_name

    if image_path.exists():
        print(f"File already exists at {image_path}")
        return

    # create a figure and axis with no frame
    fig, ax = plt.subplots(figsize=(image_size, image_size))
    # remove all margins / make the plot fill the whole figure
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    # add the star to the axis
    ax.add_collection(gen_siemens_star((0, 0), 1, siemens_star_branches))
    # set the limits of the axis and remove the axis
    plt.axis([-1.0, 1.0, -1.0, 1.0])
    ax.set_axis_off()
    # save the figure, close it and print a message
    fig.savefig(
        image_path,
        bbox_inches="tight",
        dpi=1,
        pad_inches=0,
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)
    print(f"Siemens star saved in {image_path}")


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
