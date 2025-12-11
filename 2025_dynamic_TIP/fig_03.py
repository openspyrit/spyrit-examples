# %% [markdown]
"""
This script reproduces the results given in Fig. 3 to demonstrate the use of dynamic single-pixel imaging. 
"""

# %% Import bib
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import math

from pathlib import Path

import spyrit.core.torch as spytorch
from spyrit.core.warp import AffineDeformationField
from spyrit.core.prep import Unsplit
from spyrit.core.meas import HadamSplit2d, DynamicHadamSplit2d
from spyrit.misc.disp import torch2numpy, blue_box, save_motion_video
from spyrit.misc.statistics import Cov2Var, transform_norm
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
# x = x_healthy.repeat(1, n_wav, 1, 1).clone()
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

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

im = ax[0].imshow(tumors_plot[:, :, 0], cmap='gray')
ax[0].set_title('Red channel', fontsize=16)
ax[0].axis('off')
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

im = ax[1].imshow(tumors_plot[:, :, 1], cmap='gray')
ax[1].set_title('Green channel', fontsize=16)
ax[1].axis('off')
fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

im = ax[2].imshow(tumors_plot[:, :, 2], cmap='gray')
ax[2].set_title('Blue channel with tumor', fontsize=16)
ax[2].axis('off')
fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()



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


   
    # %% SAVE DEFORMATION 
    fps = int(x_motion.shape[1] / 10)
    video_path = 'fig_03_rgb_motion_2.mp4'
    save_motion_video(x_motion, video_path, amp_max, fps=fps)

    # video_path = 'fig_03_cmos_motion.mp4'
    # save_motion_video(x_motion.mean(dim=2, keepdim=True), video_path, amp_max, fps=fps)



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
    meas_op.build_dynamic_forward(def_field)
    
    H_dyn_diff = meas_op.H_dyn


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

    x_rec = torch.linalg.solve(H_dyn_diff.T @ H_dyn_diff + eta * sigma_max ** 2 * D2, (H_dyn_diff.T @ y2.moveaxis(1, -1)))

    # %% Plot reference, static and dynamic reconstructions side-by-side
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    x_ref_blue = blue_box(x_plot, amp_max)
    ax[0].imshow(x_ref_blue)
    ax[0].set_title('Reference image', fontsize=16)
    ax[0].axis('off')

    x_stat_wide = torch.zeros((img_size, img_size, n_wav), dtype=dtype, device=device)
    x_stat_wide[amp_max:-amp_max, amp_max:-amp_max] = x_stat.squeeze().moveaxis(0, -1)
    x_stat_wide_blue =  blue_box(torch2numpy(x_stat_wide), amp_max)
    ax[1].imshow(x_stat_wide_blue)
    ax[1].set_title('Static reconstruction', fontsize=16)
    ax[1].axis('off')

    x_rec_plot = x_rec.squeeze().reshape((img_size, img_size, n_wav))
    x_rec_blue = blue_box(torch2numpy(x_rec_plot), amp_max)
    ax[2].imshow(x_rec_blue)
    ax[2].set_title('Dynamic reconstruction', fontsize=16)
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()


    # %%
    x_cmos = x_plot.mean(axis=2)
    x_cmos_blue = blue_box(x_cmos, amp_max)
    plt.imshow(x_cmos_blue)
    plt.title(r"Simulated CMOS image", fontsize=16)
    plt.show()


    # %% save results as pdfs
    if save_fig:
        results_root = Path('/home/maitre/Images/images_thèse/2024_article/ablation_study/visual/rgb_scene/exp_0')
        results_root.mkdir(parents=True, exist_ok=True)

        plt.imsave(results_root / Path('reference_box.pdf'), x_ref_blue.clip(0, 255))
        plt.imsave(results_root / Path('reference_cmos.pdf'), x_cmos_blue.clip(0, 255))
        plt.imsave(results_root / Path('reco_static.pdf'), x_stat_wide_blue.clip(0, 255))
        plt.imsave(results_root / Path('reco_wf_eta_1e-3.pdf'), x_rec_blue.clip(0, 255))
        plt.imsave(results_root / Path('tumors.pdf'), tumors_plot.clip(0, 1))

# %%
