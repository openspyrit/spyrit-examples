# %% [markdown]
"""
This script reproduces the results given in Fig. 4 and study the impact of the interpolation
order for warping the patterns in dynamic single-pixel imaging.
"""

# %% Import bib
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import math

from pathlib import Path

import spyrit.core.torch as spytorch
from spyrit.core.meas import DynamicHadamSplit2d
from spyrit.core.prep import Unsplit
from spyrit.core.warp import AffineDeformationField

from spyrit.misc.disp import torch2numpy, imagesc, blue_box
from spyrit.misc.statistics import transform_norm, Cov2Var
import spyrit.misc.metrics as score
from spyrit.misc.load_data import download_girder, generate_synthetic_tumors



#%% LOAD IMAGE DATA
save_fig = True

img_size = 88  # full image side's size in pixels
meas_size = 64  # measurement pattern side's size in pixels (Hadamard matrix)
img_shape = (img_size, img_size)
meas_shape = (meas_size, meas_size)

amp_max = (img_shape[0] - meas_shape[0]) // 2

i = 0  # Image index (modify to change the image)
spyritPath = '../data/data_online/' #os.getcwd()
imgs_path = os.path.join(spyritPath, "spyrit/")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # avoid memory issues
print("Using device:", device)

dtype = torch.float64
simu_interp = 'bicubic'


# Create a transform for natural images to normalized grayscale image tensors
transform = transform_norm(img_size=img_size)

# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
# Reduce batch size to save memory
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)  # Changed from 7 to 1


# %% Select image
i = 1  # Image index (modify to change the image)

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
D2 = Dx.T @ Dx + Dy.T @ Dy
D2 = D2.to(dtype=dtype)

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

x_motion = aff_field(x, 0, (meas_size**2) * 2, mode=simu_interp)


# %% SIMULATE MEASUREMENT
meas_op = DynamicHadamSplit2d(time_dim=1, h=meas_size, M=meas_size**2, order=Ord, img_shape=img_shape, dtype=dtype, device=device)

y1 = meas_op(x_motion)

prep_op = Unsplit()
y2 = prep_op(y1)


# %% DYNAMIC MATRIX CONSTRUCTION
no_warp_interp = 'bilinear'

meas_op.build_dynamic_forward(aff_field, warping='image', mode=no_warp_interp)
H_dyn_diff = meas_op.H_dyn.cpu()


#%%
aff_field_inverse = AffineDeformationField(f_inverse, time_vector, img_shape, dtype=dtype, device=device)

warp_interp_list = ['nearest', 'bilinear', 'bicubic']

H_dyn_virtual_diff_tot = torch.zeros((len(warp_interp_list), meas_size ** 2, img_size ** 2), dtype=dtype)

for warp_interp_ind, warp_interp in enumerate(warp_interp_list):
    meas_op.build_dynamic_forward(aff_field_inverse, warping='pattern', mode=warp_interp)
    H_dyn_virtual_diff_tot[warp_interp_ind, :, :] = meas_op.H_dyn.cpu()


#%%
from tqdm import tqdm

eta_list = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]

psnr_array = np.zeros((len(warp_interp_list) + 1, len(eta_list)))
ssim_array = np.zeros((len(warp_interp_list) + 1, len(eta_list)))

x_rec_array = np.zeros((len(warp_interp_list) + 1, len(eta_list), img_size, img_size, n_wav))

D2_cpu = D2.cpu()
y2_cpu = y2.moveaxis(1, -1).cpu()


for eta_ind, eta in tqdm(enumerate(eta_list)):
    for warp_interp_ind, warp_interp in enumerate(warp_interp_list):
        sing_vals = torch.linalg.svdvals(H_dyn_virtual_diff_tot[warp_interp_ind])
        sigma_max = sing_vals[0]

        x_rec_virtual = torch.linalg.solve(
             H_dyn_virtual_diff_tot[warp_interp_ind].T @  H_dyn_virtual_diff_tot[warp_interp_ind] + eta * sigma_max ** 2 * D2_cpu,
             H_dyn_virtual_diff_tot[warp_interp_ind].T @ y2_cpu
        )
        x_rec_virtual_plot = x_rec_virtual.view(img_size, img_size, n_wav)

        x_rec_in_X = x_rec_virtual_plot[amp_max:-amp_max, amp_max:-amp_max]
        x_in_X = x[0, :, amp_max:-amp_max, amp_max:-amp_max].moveaxis(0, -1)
        psnr_array[warp_interp_ind, eta_ind] = score.psnr_(torch2numpy(x_rec_in_X), torch2numpy(x_in_X), r=1)
        ssim_array[warp_interp_ind, eta_ind] = score.ssim(torch2numpy(x_rec_in_X), torch2numpy(x_in_X))

        x_rec_array[warp_interp_ind, eta_ind] = x_rec_virtual_plot.clone().detach().cpu().numpy()

    sing_vals = torch.linalg.svdvals(H_dyn_diff)
    sigma_max = sing_vals[0]

    x_rec = torch.linalg.solve(
        H_dyn_diff.T @ H_dyn_diff + eta * sigma_max ** 2 * D2_cpu,
        H_dyn_diff.T @ y2_cpu
    )
    x_rec_plot = x_rec.view(img_size, img_size, n_wav)

    x_rec_array[3, eta_ind] = x_rec_plot.clone().detach().cpu().numpy()

    x_rec_in_X = x_rec_plot[amp_max:meas_size+amp_max, amp_max:meas_size+amp_max]
    x_in_X = x[0, :, amp_max:meas_size+amp_max, amp_max:meas_size+amp_max].moveaxis(0, -1)
    psnr_array[-1, eta_ind] = score.psnr_(torch2numpy(x_rec_in_X), torch2numpy(x_in_X), r=1)
    ssim_array[-1, eta_ind] = score.ssim(torch2numpy(x_rec_in_X), torch2numpy(x_in_X))



# %% Plot reconstructions (b), (c), (e), (f) from Fig. 4
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(blue_box(x_rec_array[0, 7, :, :], amp_max))
ax[0, 0].set_title('wh nearest', fontsize=16)
ax[0, 0].axis('off')

ax[0, 1].imshow(blue_box(x_rec_array[1, 6, :, :], amp_max))
ax[0, 1].set_title('wh bilinear', fontsize=16)
ax[0, 1].axis('off')

ax[1, 0].imshow(blue_box(x_rec_array[2, 5, :, :], amp_max))
ax[1, 0].set_title('wh bicubic', fontsize=16)
ax[1, 0].axis('off')

ax[1, 1].imshow(blue_box(x_rec_array[3, 5, :, :], amp_max))
ax[1, 1].set_title('wf bilinear', fontsize=16)
ax[1, 1].axis('off')

plt.tight_layout()
plt.show()

if save_fig:
    results_root = Path('/home/maitre/Images/images_thèse/2024_article/ablation_study/visual/rgb_scene/exp_1')
    results_root.mkdir(parents=True, exist_ok=True)

    plt.imsave(results_root / Path('wh_o0.pdf'), blue_box(x_rec_array[0, 7, :, :], amp_max).clip(0, 255))
    plt.imsave(results_root / Path('wh_o1.pdf'), blue_box(x_rec_array[1, 6, :, :], amp_max).clip(0, 255))
    plt.imsave(results_root / Path('wh_o3.pdf'), blue_box(x_rec_array[2, 5, :, :], amp_max).clip(0, 255))
    plt.imsave(results_root / Path('wf_o1.pdf'), blue_box(x_rec_array[3, 5, :, :], amp_max).clip(0, 255))



# %% PLOT PSNR % eta
methods_list = ['wf', 'wh']
colors = ['red', 'blue']

markers = ['^', 'o', 'D']
linestyles = ['dotted', 'dashed', 'dashdot']


fontsize = 35
plt.figure(figsize=(15, 15))
plt.rc('legend', fontsize=18)
plt.rc(('xtick', 'ytick'), labelsize=fontsize)

for ind, (warp_order, marker, linestyle) in enumerate(zip(warp_interp_list, markers, linestyles)):
    method_name = 'wh' + ' ' + warp_order
    plt.plot(np.array(eta_list), psnr_array[ind], marker=marker, markersize=16, 
                        color='blue', label=method_name, linewidth=3, linestyle=linestyle)
    
plt.plot(np.array(eta_list), psnr_array[3], marker='o', markersize=16, 
                        color='red', label='wf bilinear', linewidth=3, linestyle='dashed')
        

plt.xlabel(r'Regularization parameter $\eta$', fontsize=fontsize)
plt.ylabel('PSNR', fontsize=fontsize)
plt.xscale('log')
plt.legend(fontsize=fontsize)
plt.tight_layout()
if save_fig:
    plt.savefig(results_root / Path('psnr_vs_eta.pdf'))
plt.show()


# %% SSIM
fontsize = 35
plt.figure(figsize=(15, 15))
plt.rc('legend', fontsize=18)
plt.rc(('xtick', 'ytick'), labelsize=fontsize)

for ind, (warp_order, marker, linestyle) in enumerate(zip(warp_interp_list, markers, linestyles)):
    method_name = 'wh' + ' ' + warp_order
    plt.plot(np.array(eta_list), ssim_array[ind], marker=marker, markersize=16, 
                        color='blue', label=method_name, linewidth=3, linestyle=linestyle)
    
plt.plot(np.array(eta_list), ssim_array[3], marker='o', markersize=16, 
                        color='red', label='wf bilinear', linewidth=3, linestyle='dashed')
    
plt.xlabel(r'Regularization parameter $\eta$', fontsize=fontsize)
plt.ylabel('SSIM', fontsize=fontsize)
plt.xscale('log')
plt.legend(fontsize=fontsize)
plt.tight_layout()
if save_fig:
    plt.savefig(results_root / Path('ssim_vs_eta.pdf'))
plt.show()



# %%
