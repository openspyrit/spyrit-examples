#%% [markdown]
"""
This script is used to reproduce the results given in Fig. 5 for the motion strength study.
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

import spyrit.core.torch as spytorch
from spyrit.core.prep import Unsplit
from spyrit.core.meas import DynamicHadamSplit2d
from spyrit.core.noise import Poisson
from spyrit.core.warp import DeformationField

from spyrit.misc.disp import torch2numpy, imagesc, blue_box, save_field_video, save_motion_video
from spyrit.misc.statistics import transform_gray_norm, Cov2Var, data_loaders_stl10
import spyrit.misc.metrics as score
from spyrit.misc.load_data import download_girder



#%% LOAD IMAGE DATA
save_fig = False
save_deform = True

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
idx_img = 7667  # Image index (modify to change the image)

img, _ = dataloader.dataset[idx_img]
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

# %% Get three precomputed elastic deformations from Tomoradio warehouse
url_tomoradio = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
path_deform = Path.cwd() / Path('spyrit-examples/2025_dynamic_TIP/') / Path('deformations_elastic')  # replace with your path
id_files = [
    "693ae1b64c0b0d3d4bc700c0",  # def_field_index_72_comp_10.npz
    "693ae1b34c0b0d3d4bc700bd",  # def_field_index_56_comp_10.npz
    "693ae1be4c0b0d3d4bc700c6",  # def_field_index_250_comp_10.npz
    "693ae1b04c0b0d3d4bc700b4",   # amplitudes_index_50_100.npy
    "693ae1b04c0b0d3d4bc700ba"   # amplitudes_index_250_300.npy
]
try:
    download_girder(url_tomoradio, id_files, path_deform)

except Exception as e:
    print("Unable to download from the Tomoradio warehouse")
    print(e)



# %% DEFINE DEFORMATION
## We provide the random elastic deformations fields used in the paper's experiments. 
## The inverse deformation field was computed using scipy's griddata function (forward mapping).

deform_idxs = [72, 56, 250]  # min, med, max amplitude indices

results_root = Path('../../Images/images_thèse/2024_article/ablation_study/noise/image_bank/score_vs_motion/reco')

rec_array = np.zeros((len(deform_idxs), 4, img_size, img_size), dtype=np.float32)
psnr_array = np.zeros((len(deform_idxs), 4), dtype=np.float32)
ssim_array = np.zeros((len(deform_idxs), 4), dtype=np.float32)

with torch.no_grad():
    # load direct deform and inverse deform that were used in the paper's experiments
    path_deform = Path.cwd() / Path('spyrit-examples/2025_dynamic_TIP/') / Path('deformations_elastic')  # replace with your path

    scale_factor = (torch.tensor(img_shape) - 1).to(device=device)
    comp_factor = 10
    deform_batch_size = 50

    for i_deform in deform_idxs:
        # vocabulary was changed, here the saved 'direct' field is v_k; and the 'inverse' field is u_k
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

        time_dim = 1
        x_motion = def_field(x, 0, 2 * M, mode=simu_interp)
        x_motion = x_motion.moveaxis(time_dim, 1)
        print("x_motion.shape:", x_motion.shape)


        # Save deformation field as quiver plot video
        if save_deform:
            path_fig = path_deform
            Path(path_fig).mkdir(parents=True, exist_ok=True)
            video_path = path_fig / Path('deformation_index_%d_quiver.mp4' % i_deform)

            n_frames = 200
            step = 6  # subsampling for arrows
            fps = 30

            save_field_video(def_field, video_path, n_frames=n_frames, step=step, fps=fps, figsize=(6, 6), dpi=200, scale=1, fs=16,
                            amp_max=amp_max, box_color='blue', box_linewidth=2)

            video_path = path_fig / Path(f'motion_img_{idx_img}_deform_{i_deform}.mp4')
            fps = 216
            save_motion_video(x_motion, video_path, amp_max, fps=fps)

        # SIMULATE MEASUREMENT
        torch.manual_seed(100)  # for reproductible results

        alpha = 1000 # number of photons
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


        # Compute finite difference matrices in X and in X_{ext}
        Dx, Dy = spytorch.neumann_boundary(img_shape)
        D2_in_Xext = Dx.T @ Dx + Dy.T @ Dy
        D2_in_Xext = D2_in_Xext.to(dtype=dtype)

        Dx_in_X, Dy_in_X = spytorch.neumann_boundary(meas_shape)
        D2_in_X = Dx_in_X.T @ Dx_in_X + Dy_in_X.T @ Dy_in_X
        D2_in_X = D2_in_X.type(dtype=dtype)

        eta_in_X = 2e-1
        eta_in_Xext = 1e-1

        
        # Reconstruction with pattern warping (in X or in X_{ext})
        meas_op.build_dynamic_forward(def_field_inv, warping='pattern', mode=mode)
        H_dyn_diff = meas_op.H_dyn.to(device='cpu')
            
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



        # Reconstruction with image warping (in X or in X_{ext})
        meas_op.build_dynamic_forward(def_field, warping='image', mode=mode)
        H_dyn_diff = meas_op.H_dyn.to(device='cpu')
            
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


        # METRICS in X
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


        # save results
        rec_array[deform_idxs.index(i_deform), 0, amp_max:-amp_max, amp_max:-amp_max] = torch2numpy(x_wh_in_X.view((meas_size, meas_size)))
        rec_array[deform_idxs.index(i_deform), 1, amp_max:-amp_max, amp_max:-amp_max] = torch2numpy(x_wf_in_X.view((meas_size, meas_size)))
        rec_array[deform_idxs.index(i_deform), 2, :, :] = torch2numpy(x_wh_in_Xext.view((img_size, img_size)))
        rec_array[deform_idxs.index(i_deform), 3, :, :] = torch2numpy(x_wf_in_Xext.view((img_size, img_size)))

        psnr_array[deform_idxs.index(i_deform), 0] = psnr_wh_in_X
        psnr_array[deform_idxs.index(i_deform), 1] = psnr_wf_in_X
        psnr_array[deform_idxs.index(i_deform), 2] = psnr_wh_in_Xext
        psnr_array[deform_idxs.index(i_deform), 3] = psnr_wf_in_Xext

        ssim_array[deform_idxs.index(i_deform), 0] = ssim_wh_in_X
        ssim_array[deform_idxs.index(i_deform), 1] = ssim_wf_in_X
        ssim_array[deform_idxs.index(i_deform), 2] = ssim_wh_in_Xext
        ssim_array[deform_idxs.index(i_deform), 3] = ssim_wf_in_Xext


        if save_fig:
            path_fig = results_root / Path(f'img_{idx_img}') / Path(f'deform_idx_%d_amp_%d' % (i_deform, int(magnitude_amp)))
            Path(path_fig).mkdir(parents=True, exist_ok=True)

            # wh in X
            plt.imsave(path_fig / f'wh_in_X.pdf',
                    blue_box(rec_array[deform_idxs.index(i_deform), 0], amp_max=amp_max))

            # wf in X
            plt.imsave(path_fig / f'wf_in_X_.pdf',
                    blue_box(rec_array[deform_idxs.index(i_deform), 1], amp_max=amp_max))

            # wh in X ext
            plt.imsave(path_fig / f'wh_in_X_ext.pdf',
                    blue_box(rec_array[deform_idxs.index(i_deform), 2], amp_max=amp_max))

            # wf in X ext
            plt.imsave(path_fig / f'wf_in_X_ext.pdf',
                    blue_box(rec_array[deform_idxs.index(i_deform), 3], amp_max=amp_max))
            
        # keep track in arrays to display later
       
        

    # %% Plot reconstructions (a), (b), (c), (d) from Fig. 5
    fig, ax = plt.subplots(4, 3, figsize=(15, 20))

    method_names = ['wh in $X$', 'wf in $X$', 'wh in $X_{ext}$', 'wf in $X_{ext}$']
    rec_np = np.asarray(rec_array)  # shape: (n_deforms, 4, H, W)
    n_deforms = rec_np.shape[0]

    for r in range(4):  # rows = reconstruction methods
        for c in range(n_deforms):  # cols = deformation indices
            im = rec_np[c, r, :, :].astype(np.float32)
            ax[r, c].imshow(im, cmap='gray', vmin=0.0, vmax=1.0)
            psnr = psnr_array[c, r]
            ssim = ssim_array[c, r]
            ax[r, c].set_title('%s\nPSNR: %.2f dB | SSIM: %.4f' % (method_names[r], psnr, ssim), fontsize=12)
            ax[r, c].axis('off')

    plt.tight_layout()
    plt.show()

# %%
