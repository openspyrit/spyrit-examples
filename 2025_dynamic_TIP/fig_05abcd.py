#%% [markdown]
"""
This script is used to reproduce the results given in Fig. 5a, 5b, 5c and 5d.
It compares reconstruction methods using pattern warping (wh) and image warping (wf), 
and the influence of the extended FOV, for a given random elastic deformation field,
on a single image from the STL-10 dataset.
"""

# %% Import bib
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path
from IPython.display import clear_output

import spyrit.core.torch as spytorch
from spyrit.core.prep import Unsplit
from spyrit.core.meas import DynamicHadamSplit2d
from spyrit.core.noise import Poisson
from spyrit.core.warp import DeformationField

from spyrit.misc.disp import torch2numpy, imagesc, blue_box
from spyrit.misc.statistics import transform_gray_norm, Cov2Var, data_loaders_stl10
import spyrit.misc.metrics as score
from spyrit.misc.load_data import download_girder



#%% LOAD IMAGE DATA
img_size = 88  # full image side's size in pixels
meas_size = 64  # measurement pattern side's size in pixels (Hadamard matrix)
und = 1 # undersampling factor
M = meas_size ** 2 // und  # number of (pos,neg) measurements
img_shape = (img_size, img_size)
meas_shape = (meas_size, meas_size)

data_root = '../data/data_online/' 
imgs_path = os.path.join(data_root, "stl-10_binary/")

amp_max = (img_shape[0] - meas_shape[0]) // 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dtype = torch.float32
simu_interp = 'bicubic'
mode = 'bilinear'

# Create a transform for natural images to normalized grayscale image tensors
transform = transform_gray_norm(img_size=img_size)

batch_size = 16

# Create dataset and loader
dataloaders = data_loaders_stl10(
    data_root,
    img_size=img_size,
    batch_size=batch_size,
    seed=7,
    shuffle=True,
    download=False,  # switch to True to download the dataset
)
dataloader = dataloaders['val']


# %% Select image
i = 205  # Image index (modify to change the image)

img, _ = dataloader.dataset[i]
x = img.unsqueeze(0).to(dtype=dtype, device=device)

x = (x - x.min()) / (x.max() - x.min())

x_plot = x.view(img_shape).cpu()
imagesc(x_plot, r"Original image $x$")

# %% Get exp order from Tomoradio warehouse
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



# %% DEFINE DEFORMATION
## We provide the random elastic deformations fields used in the paper's experiments. 
## The inverse deformation field was computed using scipy's griddata function (forward mapping).

with torch.no_grad():
    # load direct deform and inverse deform that were used in the paper's experiments
    path_deform = Path.cwd() / Path('spyrit-examples/2025_dynamic_TIP/') / Path('deformations_elastic')  # replace with your path

    scale_factor = (torch.tensor(img_shape) - 1).to(device=device)
    comp_factor = 10
    deform_batch_size = 50

    # i_deform = i  # deform index, magnitude of 212
    i_deform = 219  # deform index,  magnitude of 487

    index_deform = i_deform % deform_batch_size
    beg_deform_batch, end_deform_batch = (i_deform // deform_batch_size) * deform_batch_size, (i_deform // deform_batch_size + 1) * deform_batch_size
    amp_array_dist = np.load(path_deform / Path('amplitudes_index_%d_%d.npy' % (beg_deform_batch, end_deform_batch)))

    magnitude_amp = amp_array_dist[index_deform]

    def_field_compressed = np.load(path_deform / Path('def_field_index_%d_comp_%d.npz' % (i_deform, comp_factor)))

    def_field_torch = torch.from_numpy(def_field_compressed['direct']).to(dtype=dtype, device=device) / comp_factor
    def_field_torch = def_field_torch * 2 / scale_factor - 1
    def_field = DeformationField(def_field_torch)

    def_field_torch_inv = torch.from_numpy(def_field_compressed['inverse']).to(dtype=dtype, device=device) / comp_factor
    def_field_torch_inv = def_field_torch_inv * 2 / scale_factor - 1
    def_field_inv = DeformationField(def_field_torch_inv)

    del def_field_torch, def_field_torch_inv, def_field_compressed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary(device=device, abbreviated=True))

    time_dim = 1
    x_motion = def_field(x, 0, 2 * M, mode=simu_interp)
    x_motion = x_motion.moveaxis(time_dim, 1)
    print("x_motion.shape:", x_motion.shape)


   
    # %% PLOT ENTIRE DEFORMATION 
    for frame in range(int(meas_size / und ** 0.5)):
        plt.close()
        plt.imshow(x_motion[0, meas_size * frame, 0, amp_max:img_size-amp_max, amp_max:img_size-amp_max].view(meas_shape).cpu().numpy(), cmap="gray")  # in X
        plt.suptitle("frame %d" % (meas_size * frame), fontsize=16)
        plt.colorbar()
        plt.pause(0.01)
        clear_output(wait=True)



    # %% SIMULATE MEASUREMENT
    torch.manual_seed(100)  # for reproductible results

    alpha = 100 # number of photons
    noise_op = Poisson(alpha) #, g = 1 / alpha)

    meas_op = DynamicHadamSplit2d(time_dim=time_dim, h=meas_size, M=M, order=Ord,
                                fast=True, reshape_output=False, img_shape=img_shape,
                                noise_model=noise_op, white_acq=None,
                                dtype=dtype, device=device)
    
    y1 = meas_op(x_motion)

    prep_op = Unsplit()
    y2 = prep_op(y1)

    print("y2.shape:", y2.shape)

    m2 = y2 / alpha
    m2 = m2.to(device='cpu')


    # %% Compute finite difference matrices in X and in X_{ext}
    Dx, Dy = spytorch.neumann_boundary(img_shape)
    D2_in_Xext = Dx.T @ Dx + Dy.T @ Dy
    D2_in_Xext = D2_in_Xext.to(dtype=dtype)

    Dx_in_X, Dy_in_X = spytorch.neumann_boundary(meas_shape)
    D2_in_X = Dx_in_X.T @ Dx_in_X + Dy_in_X.T @ Dy_in_X
    D2_in_X = D2_in_X.type(dtype=dtype)

    eta_in_X = 2e-1
    eta_in_Xext = 1e-1

    
    # %% Reconstruction with pattern warping (in X or in X_{ext})
    meas_op.build_H_dyn(def_field_inv, warping='pattern', mode=mode)
    H_dyn_diff = meas_op.H_dyn_diff.to(device='cpu')
        
    ## in X_ext
    sing_vals = torch.linalg.svdvals(H_dyn_diff)
    print('Spectre de H_dyn_diff = [%.2f, %.2f]' % (sing_vals[-1], sing_vals[0]))
    sigma_max = sing_vals[0]

    x_wh_in_Xext = torch.linalg.solve(H_dyn_diff.T @ H_dyn_diff + eta_in_Xext * sigma_max ** 2 * D2_in_Xext, H_dyn_diff.T @ m2.reshape(-1))

    
    ## in X
    H_dyn_diff = H_dyn_diff.reshape((H_dyn_diff.shape[0], img_size, img_size))[:, amp_max:-amp_max, amp_max:-amp_max].reshape((H_dyn_diff.shape[0], meas_size**2))
    
    sing_vals = torch.linalg.svdvals(H_dyn_diff)
    print('Spectre de H_dyn_diff = [%.2f, %.2f]' % (sing_vals[-1], sing_vals[0]))
    sigma_max = sing_vals[0]

    x_wh_in_X = torch.linalg.solve(H_dyn_diff.T @ H_dyn_diff + eta_in_X * sigma_max ** 2 * D2_in_X, H_dyn_diff.T @ m2.reshape(-1))



    # %% Reconstruction with image warping (in X or in X_{ext})
    meas_op.build_H_dyn(def_field, warping='image', mode=mode)
    H_dyn_diff = meas_op.H_dyn_diff.to(device='cpu')
        
    ## in X_ext
    sing_vals = torch.linalg.svdvals(H_dyn_diff)
    print('Spectre de H_dyn_diff = [%.2f, %.2f]' % (sing_vals[-1], sing_vals[0]))
    sigma_max = sing_vals[0]

    x_wf_in_Xext = torch.linalg.solve(H_dyn_diff.T @ H_dyn_diff + eta_in_Xext * sigma_max ** 2 * D2_in_Xext, H_dyn_diff.T @ m2.reshape(-1))

    
    ## in X
    H_dyn_diff = H_dyn_diff.reshape((H_dyn_diff.shape[0], img_size, img_size))[:, amp_max:-amp_max, amp_max:-amp_max].reshape((H_dyn_diff.shape[0], meas_size**2))
    
    sing_vals = torch.linalg.svdvals(H_dyn_diff)
    print('Spectre de H_dyn_diff = [%.2f, %.2f]' % (sing_vals[-1], sing_vals[0]))
    sigma_max = sing_vals[0]

    x_wf_in_X = torch.linalg.solve(H_dyn_diff.T @ H_dyn_diff + eta_in_X * sigma_max ** 2 * D2_in_X, H_dyn_diff.T @ m2.reshape(-1))


    #%% METRICS in X
    x_in_fov = x[0, 0, amp_max:-amp_max, amp_max:-amp_max]

    x_wh_in_Xext_in_fov = x_wh_in_Xext.view(img_shape)[amp_max:-amp_max, amp_max:-amp_max]
    psnr_wh_in_Xext = score.psnr_(torch2numpy(x_wh_in_Xext_in_fov), torch2numpy(x_in_fov), r=1)
    ssim_wh_in_Xext = score.ssim(torch2numpy(x_wh_in_Xext_in_fov), torch2numpy(x_in_fov))

    x_wh_in_X_in_fov = x_wh_in_X.view(meas_shape)
    psnr_wh_in_X = score.psnr_(torch2numpy(x_wh_in_X_in_fov), torch2numpy(x_in_fov), r=1)
    ssim_wh_in_X = score.ssim(torch2numpy(x_wh_in_X_in_fov), torch2numpy(x_in_fov))

    x_wf_in_Xext_in_fov = x_wf_in_Xext.view(img_shape)[amp_max:-amp_max, amp_max:-amp_max]
    psnr_wf_in_Xext = score.psnr_(torch2numpy(x_wf_in_Xext_in_fov), torch2numpy(x_in_fov), r=1)
    ssim_wf_in_Xext = score.ssim(torch2numpy(x_wf_in_Xext_in_fov), torch2numpy(x_in_fov))
    
    x_wf_in_X_in_fov = x_wf_in_X.view(meas_shape)
    psnr_wf_in_X = score.psnr_(torch2numpy(x_wf_in_X_in_fov), torch2numpy(x_in_fov), r=1)
    ssim_wf_in_X = score.ssim(torch2numpy(x_wf_in_X_in_fov), torch2numpy(x_in_fov))
    

    # %% Plot reconstructions (a), (b), (c), (d) from Fig. 5
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    x_wh_in_X_wide = torch.zeros((img_size, img_size), dtype=dtype, device=device)
    x_wh_in_X_wide[amp_max:-amp_max, amp_max:-amp_max] = x_wh_in_X.view(meas_shape)
    ax[0, 0].imshow(x_wh_in_X_wide.cpu().numpy(), cmap='gray')
    ax[0, 0].set_title('wh in $X$\nPSNR: %.2f dB | SSIM: %.4f' % (psnr_wh_in_X, ssim_wh_in_X), fontsize=16)
    ax[0, 0].axis('off')

    x_wf_in_X_wide = torch.zeros((img_size, img_size), dtype=dtype, device=device)
    x_wf_in_X_wide[amp_max:-amp_max, amp_max:-amp_max] = x_wf_in_X.view(meas_shape)
    ax[0, 1].imshow(x_wf_in_X_wide.cpu().numpy(), cmap='gray')
    ax[0, 1].set_title('wf in $X$\nPSNR: %.2f dB | SSIM: %.4f' % (psnr_wf_in_X, ssim_wf_in_X), fontsize=16)
    ax[0, 1].axis('off')

    ax[1, 0].imshow(blue_box(x_wh_in_Xext.view(img_shape).cpu().numpy(), amp_max))
    ax[1, 0].set_title('wh in $X_{ext}$\nPSNR: %.2f dB | SSIM: %.4f' % (psnr_wh_in_Xext, ssim_wh_in_Xext), fontsize=16)
    ax[1, 0].axis('off')

    ax[1, 1].imshow(blue_box(x_wf_in_Xext.view(img_shape).cpu().numpy(), amp_max))
    ax[1, 1].set_title('wf in $X_{ext}$\nPSNR: %.2f dB | SSIM: %.4f' % (psnr_wf_in_Xext, ssim_wf_in_Xext), fontsize=16)
    ax[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

# %%
