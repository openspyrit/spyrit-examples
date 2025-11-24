# %% [markdown]
"""
This script reproduces the results given in Fig. 3 to demonstrate the use of dynamic single-pixel imaging. 
"""

# %% Import bib
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import math

from pathlib import Path
from IPython.display import clear_output

import spyrit.core.torch as spytorch
from spyrit.core.warp import AffineDeformationField
from spyrit.core.prep import Unsplit
from spyrit.core.meas import HadamSplit2d, DynamicHadamSplit2d
from spyrit.misc.disp import torch2numpy, imagesc, blue_box
from spyrit.misc.statistics import transform_gray_norm, Cov2Var
from spyrit.misc.load_data import download_girder



#%% LOAD IMAGE DATA
img_size = 88  # full image side's size in pixels
meas_size = 64  # measurement pattern side's size in pixels (Hadamard matrix)
und = 1 # undersampling factor
M = meas_size ** 2 // und  # number of (pos,neg) measurements
img_shape = (img_size, img_size)
meas_shape = (meas_size, meas_size)

data_root = '../data/data_online/' #os.getcwd()
imgs_path = os.path.join(data_root, "spyrit/")

amp_max = (img_shape[0] - meas_shape[0]) // 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dtype = torch.float64
simu_interp = 'bicubic'
mode = 'bilinear'

# Create a transform for natural images to normalized grayscale image tensors
transform = transform_gray_norm(img_size=img_size)

batch_size = 16

# Create dataset and loader
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)


# %% Select image
i = 0  # Image index (modify to change the image)

img, _ = dataloader.dataset[i]
x = img.unsqueeze(0).to(dtype=dtype, device=device)

print(f"Shape of input images: {x.shape}")

x = (x - x.min()) / (x.max() - x.min())

x_plot = x.view(img_shape).cpu()
imagesc(x_plot, r"Original image $x$")

## EXP ORDER
url_tomoradio = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
local_folder = Path('stats') 
id_files = [
    "6924762104d23f6e964b1441"  # 64x64 Cov_acq.npy
]
try:
    download_girder(url_tomoradio, id_files, local_folder)
except Exception as e:
    print("Unable to download from the Tomoradio warehouse")
    print(e)

Cov_acq = np.load(local_folder / ('Cov_{}x{}'.format(meas_size, meas_size) + '.npy'))
Ord_acq = Cov2Var(Cov_acq)
Ord = torch.from_numpy(Ord_acq)

Cov_acq2 = torch.load(local_folder / f'Cov_{meas_size}x{meas_size}.pt', weights_only=True).to(device)

test_4 = Cov_acq2 / torch.from_numpy(Cov_acq).to(dtype=dtype, device=device)

# %% DEFINE DEFORMATION
with torch.no_grad():
    a = 0.2  # amplitude
    omega = math.pi  # angular speed


    T = 1000  # time of a period

    def s(t):
        return 1 + a * math.sin(t * 2 * math.pi / T)  # base function for f


    def f(t):
        return torch.tensor(
            [
                [1 / s(t), 0, 0],
                [0, s(t), 0],
                [0, 0, 1],
            ],
            dtype=dtype,
        )


    time_vector = torch.linspace(0, 2 * T, 2 * M) # *2 because of the splitting
    def_field = AffineDeformationField(f, time_vector, img_shape, dtype=dtype, device=device)

    time_dim = 1
    x_motion = def_field(x, 0, 2 * M, mode=simu_interp)
    x_motion = x_motion.moveaxis(time_dim, 1)
    print("x_motion.shape:", x_motion.shape)


   
    # %% PLOT ENTIRE DEFORMATION 
    for frame in range(int(meas_size / und ** 0.5)):
        plt.close()
        plt.imshow(x_motion[0, meas_size * frame, 0, amp_max:img_size-amp_max, amp_max:img_size-amp_max].view(meas_shape).cpu().numpy(), cmap="gray")  # in X
        # plt.imshow(x_motion[0, meas_size * frame, :].view(img_shape).cpu().numpy(), cmap="gray")  # in X_ext
        plt.suptitle("frame %d" % (meas_size * frame), fontsize=16)
        plt.colorbar()
        plt.pause(0.01)
        clear_output(wait=True)



    # %% SIMULATE MEASUREMENT
    torch.manual_seed(100)  # for reproductible results

    noise_op = torch.nn.Identity()

    meas_op = DynamicHadamSplit2d(time_dim=time_dim, h=meas_size, M=M, order=Ord,
                                fast=True, reshape_output=False, img_shape=img_shape,
                                noise_model=noise_op, white_acq=None,
                                dtype=dtype, device=device)
    
    y1 = meas_op(x_motion)

    prep_op = Unsplit()
    y2 = prep_op(y1)

    print("y2.shape:", y2.shape)


    # %% Static reconstruction
    meas_op_stat = HadamSplit2d(M=M, h=meas_size, order=Ord, dtype=dtype, device=device) 

    x_stat = meas_op_stat.fast_pinv(y2)


    # %% DYNAMIC MATRIX CONSTRUCTION
    meas_op.build_H_dyn(def_field)
    
    H_dyn_diff = meas_op.H_dyn_diff


    # %% send to cpu for efficient linalg
    H_dyn_diff = H_dyn_diff.to(device='cpu')
    y2 = y2.to(device='cpu')


    # %% COND
    sing_vals = torch.linalg.svdvals(H_dyn_diff)
    print('Spectre de H_dyn_diff = [%.2f, %.2f]' % (sing_vals[-1], sing_vals[0]))
    sigma_max = sing_vals[0]


    # %% CALC FINITE DIFFERENCE MATRIX
    Dx, Dy = spytorch.neumann_boundary(img_shape)
    D2 = Dx.T @ Dx + Dy.T @ Dy
    D2 = D2.type(dtype=torch.float64)


    # %% RECO
    eta = 1e-3

    x_rec = torch.linalg.solve(H_dyn_diff.T @ H_dyn_diff + eta * sigma_max ** 2 * D2, H_dyn_diff.T @ y2.reshape(-1))

    # %% Plot reference, static and dynamic reconstructions side-by-side

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    x_ref_blue = blue_box(torch2numpy(x_plot), amp_max)
    ax[0].imshow(x_ref_blue)
    ax[0].set_title('Reference image', fontsize=16)
    ax[0].axis('off')

    x_stat_wide = torch.zeros(img_shape, dtype=dtype, device=device)
    x_stat_wide[amp_max:-amp_max, amp_max:-amp_max] = x_stat.view(meas_shape)
    x_stat_wide_blue =  blue_box(torch2numpy(x_stat_wide), amp_max)
    ax[1].imshow(x_stat_wide_blue)
    ax[1].set_title('Static reconstruction', fontsize=16)
    ax[1].axis('off')

    x_rec_plot = x_rec.view(img_shape)
    x_rec_blue = blue_box(torch2numpy(x_rec_plot), amp_max)
    ax[2].imshow(x_rec_blue)
    ax[2].set_title('Dynamic reconstruction', fontsize=16)
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()


# %%
