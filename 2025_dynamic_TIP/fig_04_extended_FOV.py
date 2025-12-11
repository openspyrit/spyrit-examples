# %% [markdown]
"""
This script reproduces the results given in Fig. 4 and demonstrate the importance of
modeling and extended field of view (FOV) in dynamic single-pixel imaging.
"""

# %% Import bib
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm

from pathlib import Path

import spyrit.core.torch as spytorch
from spyrit.core.meas import DynamicHadamSplit2d
from spyrit.core.prep import Unsplit
from spyrit.core.warp import AffineDeformationField

from spyrit.misc.disp import torch2numpy, blue_box
from spyrit.misc.statistics import transform_norm, Cov2Var
import spyrit.misc.metrics as score
from spyrit.misc.load_data import download_girder, generate_synthetic_tumors



#%% LOAD IMAGE DATA
save_fig = True

img_size = 88  # full image side's size in pixels
meas_size = 64  # measurement pattern side's size in pixels (Hadamard matrix)
und = 1 # undersampling factor
M = meas_size ** 2 // und  # number of (pos,neg) measurements
img_shape = (img_size, img_size)
meas_shape = (meas_size, meas_size)

amp_max = (img_shape[0] - meas_shape[0]) // 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dtype = torch.float64
simu_interp = 'bicubic'
mode = 'bilinear'

# Download images from Tomoradio's warehouse if needed
url_tomoradio = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
data_root = Path('../data/data_online/2025_dynamic')   # local path to data
imgs_path = data_root / Path("images/")
id_files = [
    "69248e3204d23f6e964b16b7"  # brain_surface_colorized.png
]
try:
    download_girder(url_tomoradio, id_files, imgs_path)
except Exception as e:
    print("Unable to download from the Tomoradio warehouse")
    print(e)

# Create a transform for natural images to normalized image tensors
transform = transform_norm(img_size=img_size)

batch_size = 1

# Create dataset and loader
dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# %% Select image
i = 0  # Image index (modify to change the image)

img, _ = dataloader.dataset[i]
x_healthy = img.unsqueeze(0).to(dtype=dtype, device=device)

print(f"Shape of input images: {x_healthy.shape}")

x_healthy = (x_healthy - x_healthy.min()) / (x_healthy.max() - x_healthy.min())

x_plot = x_healthy.moveaxis(1, -1).squeeze().cpu().numpy()

plt.imshow(x_plot)
if x_healthy.shape[1] == 1:
    plt.colorbar(fraction=0.046, pad=0.04)
plt.show()


# %% Add some tumors
n_wav = 3  # RGB
x = x_healthy.clone()

tumor_params = [
    {'center': (50, 50), 'sigma_x': 8, 'sigma_y': 8, 'amplitude': 1.3, 'channels': [0], 'angle': 0},  # Red
    
    {'center': (70, 60), 'sigma_x': 5, 'sigma_y': 10, 'amplitude': 0.5, 'channels': [1], 'angle': 30},  # Green
    
    {'center': (17, 70), 'sigma_x': 6, 'sigma_y': 6, 'amplitude': 0.7, 'channels': [2], 'angle': 0},  # Blue

    {'center': (65, 25), 'sigma_x': 10, 'sigma_y': 7, 'amplitude': 1, 'channels': [0], 'angle': 45},
    {'center': (65, 25), 'sigma_x': 10, 'sigma_y': 7, 'amplitude': 0.5, 'channels': [2], 'angle': 45}, 

    {'center': (25, 20), 'sigma_x': 12, 'sigma_y': 8, 'amplitude': 0.9, 'channels': [1], 'angle': 0},  
    {'center': (25, 20), 'sigma_x': 12, 'sigma_y': 8, 'amplitude': 0.6, 'channels': [0], 'angle': 0}, 
]

tumors, x = generate_synthetic_tumors(x, tumor_params)

tumors_plot = tumors.moveaxis(1, -1).squeeze().cpu().numpy()
plt.imshow(tumors_plot)
plt.title(r"Synthetic tumors (RGB)", fontsize=16)
plt.show()


#%%
x_plot = x.moveaxis(1, -1).squeeze().cpu().numpy()
plt.imshow(x_plot)
plt.title(r"Original RGB image with synthetic tumors $x$", fontsize=16)
plt.show()


# %% DEFINE DEFORMATION (affine deform)
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
        device=device
    )

def f_inverse(t):
    return torch.tensor(
        [
            [s(t), 0, 0],
            [0, 1 / s(t), 0],
            [0, 0, 1],
        ],
        dtype=dtype,
        device=device
    )


# %% CALC FINITE DIFFERENCE MATRIX
Dx, Dy = spytorch.neumann_boundary(img_shape)
D2_in_Xext = Dx.T @ Dx + Dy.T @ Dy
D2_in_Xext = D2_in_Xext.to(dtype=dtype)

Dx_in_X, Dy_in_X = spytorch.neumann_boundary(meas_shape)
D2_in_X = Dx_in_X.T @ Dx_in_X + Dy_in_X.T @ Dy_in_X
D2_in_X = D2_in_X.type(dtype=dtype)

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



# %% SIMULATE DATA
time_vector = torch.linspace(0, 2 * T, (meas_size**2) * 2, dtype=dtype, device=device) # *2 because of the splitting

aff_field = AffineDeformationField(f, time_vector, img_shape, dtype=dtype, device=device)
aff_field_inverse = AffineDeformationField(f_inverse, time_vector, img_shape, dtype=dtype, device=device)


x_motion = aff_field(x, 0, (meas_size**2) * 2, mode=simu_interp)


# %% SIMULATE MEASUREMENT
meas_op = DynamicHadamSplit2d(time_dim=1, h=meas_size, M=meas_size**2, order=Ord, img_shape=img_shape, dtype=dtype, device=device)

y1 = meas_op(x_motion)

prep_op = Unsplit()
y2 = prep_op(y1).cpu()  # send to cpu for linalg operations




#%%
# eta_list = [1e-10, 1e-9, 1e-8, 1e-6, 1e-4, 1e-1, 1, 1e1, 1e2, 1e3, 5e3, 1e4]
eta_list = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]

in_X_list = [False, True]
warping_list = ['image', 'pattern']

psnr_array = np.zeros((len(in_X_list), len(warping_list), len(eta_list)))
ssim_array = np.zeros((len(in_X_list), len(warping_list), len(eta_list)))

x_rec_array = np.zeros((len(in_X_list), len(warping_list), len(eta_list), img_size, img_size, n_wav))


for warping_ind, warping in enumerate(warping_list):
    if warping == 'pattern':
        warping_ind = 1
        meas_op.build_dynamic_forward(aff_field_inverse, warping=warping, mode='bilinear')
    elif warping == 'image':
        warping_ind = 0
        meas_op.build_dynamic_forward(aff_field, warping=warping, mode='bilinear')

    H_dyn_diff = meas_op.H_dyn.cpu()
    
    for in_X_ind, in_X in enumerate(in_X_list):
        if in_X:
            A_isornot_in_X = H_dyn_diff.reshape((H_dyn_diff.shape[0], img_size, img_size))[:, amp_max:-amp_max, amp_max:-amp_max].reshape((H_dyn_diff.shape[0], meas_size**2))
            D2 = D2_in_X
        else:
            A_isornot_in_X = H_dyn_diff
            D2 = D2_in_Xext

        sing_vals = torch.linalg.svdvals(A_isornot_in_X)
        sigma_max = sing_vals[0]

        for eta_ind, eta in tqdm(enumerate(eta_list)):
            x_rec = torch.linalg.solve(A_isornot_in_X.T @ A_isornot_in_X + eta * sigma_max ** 2 * D2, (A_isornot_in_X.T @ y2.moveaxis(1, -1)))
            
            if in_X:
                x_rec_in_X = x_rec.reshape((meas_size, meas_size, n_wav))
                x_rec_array[int(in_X), warping_ind, eta_ind, amp_max:-amp_max, amp_max:-amp_max] = x_rec_in_X
            else:
                x_rec = x_rec.reshape((img_size, img_size, n_wav))
                x_rec_in_X = x_rec[amp_max:-amp_max, amp_max:-amp_max]
                x_rec_array[int(in_X), warping_ind, eta_ind] = x_rec

            x_in_X = x[0, :, amp_max:-amp_max, amp_max:-amp_max].moveaxis(0, -1)
            psnr_array[int(in_X), warping_ind, eta_ind] = score.psnr_(torch2numpy(x_rec_in_X), torch2numpy(x_in_X), r=1).reshape((1, 1))
            ssim_array[int(in_X), warping_ind, eta_ind] = score.ssim(torch2numpy(x_rec_in_X), torch2numpy(x_in_X)).reshape((1, 1))


    del H_dyn_diff, A_isornot_in_X
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()

# %% Plot reconstructions (c), (d), (f), (g) from Fig. 4
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(blue_box(x_rec_array[0, 1, 6, :, :], amp_max))
ax[0, 0].set_title('wh in X_{ext}', fontsize=16)
ax[0, 0].axis('off')

ax[0, 1].imshow(x_rec_array[1, 1, 6, :, :], cmap='gray')
ax[0, 1].set_title('wh in X', fontsize=16)
ax[0, 1].axis('off')

ax[1, 0].imshow(blue_box(x_rec_array[0, 0, 5, :, :], amp_max))
ax[1, 0].set_title('wf in X_{ext}', fontsize=16)
ax[1, 0].axis('off')

ax[1, 1].imshow(x_rec_array[1, 0, 6, :, :], cmap='gray')
ax[1, 1].set_title('wf in X', fontsize=16)
ax[1, 1].axis('off')

plt.tight_layout()
plt.show()

if save_fig:
    results_root = Path('/home/maitre/Images/images_thèse/2024_article/ablation_study/visual/rgb_scene/exp_2')
    results_root.mkdir(parents=True, exist_ok=True)

    plt.imsave(results_root / Path('wh_Xext.pdf'), blue_box(x_rec_array[0, 1, 6, :, :], amp_max).clip(0, 255))
    plt.imsave(results_root / Path('wh_X.pdf'), x_rec_array[1, 1, 6, :, :].clip(0, 1))
    plt.imsave(results_root / Path('wf_Xext.pdf'), blue_box(x_rec_array[0, 0, 5, :, :], amp_max).clip(0, 255))
    plt.imsave(results_root / Path('wf_X.pdf'), x_rec_array[1, 0, 6, :, :].clip(0, 1))


# %% PLOT PSNR % eta
methods_list = ['wf', 'wh']

colors = ['red', 'blue']
markers = ['o', 's']

linestyles = ['dashed', 'dashed']

fontsize = 35
plt.figure(figsize=(15, 15))
plt.rc('legend', fontsize=18)
plt.rc(('xtick', 'ytick'), labelsize=30)

for is_in_X, linestyle, marker in zip(in_X_list, linestyles, markers):
    in_X_str = '$X$' if is_in_X else '$X_{ext}$'
    for warping, color in zip(warping_list, colors):
        warping_ind = 0 if warping == 'image' else 1
        psnrs = psnr_array[int(is_in_X), warping_ind, :]
        plt.plot(np.array(eta_list), psnrs, marker=marker, markersize=16, 
                        linestyle=linestyle, linewidth=3,
                        color=color, label=methods_list[warping_ind] + ' / ' + in_X_str)
    

plt.xlabel(r'Regularization parameter $\eta$', fontsize=fontsize)
plt.ylabel('PSNR', fontsize=fontsize)
plt.xscale('log')
plt.legend(ncol=2, loc='lower right', fontsize=fontsize)
plt.tight_layout()
if save_fig:
    plt.savefig(results_root / Path('psnr_vs_eta.pdf'))
plt.show()


# %% SSIM
plt.figure(figsize=(15, 15))
plt.rc('legend', fontsize=18)
plt.rc(('xtick', 'ytick'), labelsize=30)

for is_in_X, linestyle, marker in zip(in_X_list, linestyles, markers):
    in_X_str = '$X$' if is_in_X else '$X_{ext}$'
    for warping, color in zip(warping_list, colors):
        warping_ind = 0 if warping == 'image' else 1
        ssims = ssim_array[int(is_in_X), warping_ind, :]
        plt.plot(np.array(eta_list), ssims, marker=marker, markersize=16, 
                        linestyle=linestyle, linewidth=3,
                        color=color, label=methods_list[warping_ind] + ' / ' + in_X_str)

plt.xlabel(r'Regularization parameter $\eta$', fontsize=fontsize)
plt.ylabel('SSIM', fontsize=fontsize)
plt.xscale('log')
plt.legend(ncol=2, fontsize=fontsize)
plt.tight_layout()
if save_fig:
    plt.savefig(results_root / Path('ssim_vs_eta.pdf'))
plt.show()
# %%
