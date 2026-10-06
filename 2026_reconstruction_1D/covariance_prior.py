#%%
from pathlib import Path
from typing import Any
from pathlib import Path

import time
import torch
import torchvision
import numpy as np
from scipy.stats import rankdata

from spyrit.misc.disp import imagepanel, imagesc
import matplotlib.pyplot as plt

import spyrit.misc.walsh_hadamard as wh
import spyrit.core.torch as spytorch
import spyrit.misc.metrics as sm

from spyrit.misc.statistics import data_loaders_imagenet, cov_1, stat_1, stat_2

#%% METHODS FOR CALCULATING 1D Covariance Matrix over ImageNet dataset - adapted from spyrit

def stat_n1(dataloader, device, root, n_loop=1):
    """
    1D mean and covariance matrix of an image database.

    The statistics are computed across batches, channels, and image rows.

    nloop > 1 is relevant for dataloaders with random crops such as that
    provided by data_loaders_ImageNet

    """
    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    _, _, nx, ny = inputs.shape

    # --------------------------------------------------------------------------
    # 1. Mean
    # --------------------------------------------------------------------------
    mean = mean_1(dataloader, device, n_loop=n_loop)

    # Save
    if n_loop == 1:
        path = root / Path("Average_1_{}x{}".format(nx, ny) + ".npy")
    else:
        path = root / Path("Average_1_{}_{}x{}".format(n_loop, nx, ny) + ".npy")

    if not root.exists():
        root.mkdir()
    np.save(path, mean.cpu().detach().numpy())
    # --------------------------------------------------------------------------
    # 2. Covariance
    # -------------------------------------------------------------------------
    cov = cov_n1(dataloader, mean, device, n_loop=n_loop)

    # Save
    if n_loop == 1:
        path = root / Path("Cov_1_{}x{}".format(nx, ny) + ".npy")
    else:
        path = root / Path("Cov_1_{}_{}x{}".format(n_loop, nx, ny) + ".npy")

    if not root.exists():
        root.mkdir()
    np.save(path, cov.cpu().detach().numpy())

    return mean, cov



def mean_1(dataloader, device, n_loop=1):
    """
    The mean is computed across batches, channels, and image rows.

    nloop > 1 is relevant for dataloaders with random crops such as that
    provided by data_loaders_ImageNet

    """

    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    b, _, nx, ny = inputs.shape
    tot_num = len(dataloader) * b

    # Init
    n = 0
    mean = torch.zeros(ny, dtype=torch.float32)

    # Send to device (e.g., cuda)
    mean = mean.to(device)

    # Compute Mean
    # Accumulate sum over all the image columns in the database
    for i in range(n_loop):
        torch.manual_seed(i)
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            inputs = inputs.view(-1, nx, ny)
            #
            mean = mean.add(inputs.sum((0, 1)))  # Accumulate over images and rows

            # print
            n = n + inputs.shape[0]
            print(f"Mean:  {n} / (less than) {tot_num*n_loop} images", end="\n")
        print("", end="\n")

    # Normalize
    mean = mean / n / nx
    mean = torch.squeeze(mean)

    return mean



def cov_n1(dataloader, mean, device, n_loop=1):
    """
    The covariance is computed across batches, channels, and image rows.

    nloop > 1 is relevant for dataloaders with random crops such as that
    provided by data_loaders_ImageNet

    """

    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    b, _, nx, ny = inputs.shape
    tot_num = len(dataloader) * b

    # H = wh.walsh_matrix(ny).astype(np.float32, copy=False)
    # H = torch.from_numpy(H).to(device)

    # Covariance --------------------------------------------------------------
    # Init
    n = 0
    cov = torch.zeros((ny, ny), dtype=torch.float32)
    cov = cov.to(device)

    # Accumulate (im - mu)^T*(im - mu) over all images in dataset.
    # Each row is assumed to be an observation, so we have to transpose
    for i in range(n_loop):
        torch.manual_seed(i)
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            b, c, _, _ = inputs.shape
            inputs = inputs.view(-1, nx, ny)  # shape (b*c, nx, ny)
            #
            dev = (inputs - mean).mT
            cov = torch.addbmm(cov, dev, dev.mT)
            # print
            n += inputs.shape[0]
            print(f"Cov:  {n} / (less than) {tot_num*n_loop} images", end="\n")
        print("", end="\n")

    # Normalize
    cov = cov / (n - 1) / (nx - 1)

    return cov


def stat_imagenet_1(
    stat_root=Path("./stats/"),
    data_root=Path("./data/ILSVRC2012_img_test_v10102019/"),
    img_size: int = 64,
    batch_size: int = 1024,
    get_size: str = "resize",
    n_loop: int = 1,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    normalize=True,
    ext="npy",
    **rcrop_kwargs,
):
    """
    Args:
        :attr:`stat_root`: path to the folder where the mean and covariance
        matrices are saved

        :attr:`data_root`: path to image database.  :attr:`data_root` needs to
        have all images in a subfolder

        :attr:`img_size`: image size

        :attr:`batch_size`: batch size

        :attr:`get_size`: specifies how images of size :attr:`img_size` are
        obtained (see :mod:`~spyrit.misc.statistics.data_loaders_imagenet`)

            - 'original': random crop with padding

            - 'resize': resize

            - 'ccrop': center crop

            - 'rcrop': random crop

        :attr:`n_loop` (int, optional): Number of loops across image database. Defaults to 1. n_loop > 1 is only relevant for dataloaders with random transforms (e.g., 'rcrop' resizing)

        :attr:`normalize`: Torchvision datasets are images in the range [0, 1]. Setting :attr:`normalize` to True sends them to the range [-1, 1]. When :attr:`normalize` is False, the images are left in the range [0, 1].

        :attr:`ext` (string): Extension of saved files:

            - 'npy' for numpy (default),

            - 'pt' for pytorch,

            - do not save files otherwise.

        :attr:`rcrop_kwargs`: Additional arguments for random crop


    Example:
        >>> data_root =  Path('../data/ILSVRC2012_img_test_v10102019/')
        >>> stat_root =  Path('../stat/ILSVRC2012_img_test_v10102019')
        >>> from spyrit.misc.statistics import stat_imagenet
        >>> stat_imagenet(stat_root = stat_root, data_root = data_root) # doctest: +SKIP

    """
    dataloaders = data_loaders_imagenet(
        data_root,
        img_size=img_size,
        batch_size=batch_size,
        seed=7,
        get_size=get_size,
        normalize=normalize,
        **rcrop_kwargs,
    )

    dataloader = dataloaders["train"]

    # Compute mean and covariance
    time_start = time.perf_counter()

    mean, cov = stat_n1(dataloader, device, stat_root)

    if not stat_root.exists():
        stat_root.mkdir(parents=True, exist_ok=True)

    time_elapsed = time.perf_counter() - time_start

    print(f"Computed in {time_elapsed} seconds")

#%%

data_root =  Path(r"C:\Users\ceidigh\Documents\PFErough\Obair\ImageNet") # where image net data is stored locally
stat_root =  Path(r"C:\Users\ceidigh\Documents\PFE\data") # where to save cov matrices to 

stat_imagenet_1(stat_root = stat_root, data_root = data_root, img_size=128, normalize=False) 


#%%

