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
from spyrit.misc.statistics import transform_gray_norm, Cov2Var
import spyrit.misc.metrics as score


#%% LOAD IMAGE DATA
img_size = 88  # full image side's size in pixels
meas_size = 64  # measurement pattern side's size in pixels (Hadamard matrix)
img_shape = (img_size, img_size)
meas_shape = (meas_size, meas_size)

i = 0  # Image index (modify to change the image)
spyritPath = '../data/data_online/' #os.getcwd()
imgs_path = os.path.join(spyritPath, "spyrit/")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # avoid memory issues
print("Using device:", device)

dtype = torch.float64
simu_interp = 'bicubic'


# Create a transform for natural images to normalized grayscale image tensors
transform = transform_gray_norm(img_size=img_size)

# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
# Reduce batch size to save memory
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)  # Changed from 7 to 1

x, _ = next(iter(dataloader))

x = x[i : i + 1, :, :, :].to(dtype=dtype, device=device)
x = x.detach().clone()

amp_max = (img_size - meas_size) // 2

x = (x - x.min()) / (x.max() - x.min())

# plot
x_plot = x.view(img_shape).cpu()
imagesc(x_plot, r"Original image $x$")# in [-1, 1]")


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

## EXP ORDER
stat_folder_acq = Path('./stats/')
cov_acq_file = stat_folder_acq / ('Cov_{}x{}'.format(meas_size, meas_size) + '.npy')

Cov_acq = np.load(cov_acq_file)
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
from tqdm import tqdm

# eta_list = [1e-10, 1e-9, 1e-8, 1e-6, 1e-4, 1e-1, 1, 1e1, 1e2, 1e3, 5e3, 1e4]
eta_list = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]

in_X_list = [False, True]
warping_list = [False, True]

psnr_array = np.zeros((len(in_X_list), len(warping_list), len(eta_list)))
ssim_array = np.zeros((len(in_X_list), len(warping_list), len(eta_list)))

x_rec_array = np.zeros((len(in_X_list), len(warping_list), len(eta_list), img_size, img_size))


for warping_ind, warping in enumerate(warping_list):
    if warping:
        meas_op.build_H_dyn(aff_field_inverse, warping=warping, mode='bilinear')
    else:
        meas_op.build_H_dyn(aff_field, warping=warping, mode='bilinear')

    H_dyn_diff = meas_op.H_dyn_diff.cpu()
    
    for in_X_ind, in_X in enumerate(in_X_list):
        if in_X:
            A_isornot_in_X = H_dyn_diff.reshape((H_dyn_diff.shape[0], img_size, img_size))[:, amp_max:-amp_max, amp_max:-amp_max].reshape((H_dyn_diff.shape[0], meas_size**2))
            D2 = D2_in_X
        else:
            A_isornot_in_X = H_dyn_diff
            D2 = D2_in_Xext

        sing_vals = torch.linalg.svdvals(A_isornot_in_X)
        sigma_max = sing_vals[0]

        print('warping patterns ', warping, ' | in X ', in_X, ' | sigma max %.3f' % sigma_max)

        for eta_ind, eta in tqdm(enumerate(eta_list)):
            x_rec = torch.linalg.solve(A_isornot_in_X.T @ A_isornot_in_X + eta * sigma_max ** 2 * D2, A_isornot_in_X.T @ y2.reshape(-1))

            if in_X:
                x_rec_in_X = x_rec.reshape((1, 1, meas_size, meas_size))
                x_rec_array[int(in_X), int(warping), eta_ind, amp_max:-amp_max, amp_max:-amp_max] = x_rec.view(meas_shape)
            else:
                x_rec = x_rec.reshape((1, 1, img_size, img_size))
                x_rec_in_X = x_rec[:, :, amp_max:-amp_max, amp_max:-amp_max]
                x_rec_array[int(in_X), int(warping), eta_ind, :, :] = x_rec.view(img_shape)

            x_in_X = x[0, 0, amp_max:-amp_max, amp_max:-amp_max]
            psnr_array[int(in_X), int(warping), eta_ind] = score.psnr_(torch2numpy(x_rec_in_X), torch2numpy(x_in_X), r=1).reshape((1, 1))
            ssim_array[int(in_X), int(warping), eta_ind] = score.ssim(torch2numpy(x_rec_in_X), torch2numpy(x_in_X)).reshape((1, 1))

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
        psnrs = psnr_array[int(is_in_X), int(warping), :]
        plt.plot(np.array(eta_list), psnrs, marker=marker, markersize=16, 
                        linestyle=linestyle, linewidth=3,
                        color=color, label=methods_list[int(warping)] + ' / ' + in_X_str)
    

plt.xlabel(r'Regularization parameter $\eta$', fontsize=fontsize)
plt.ylabel('PSNR', fontsize=fontsize)
plt.xscale('log')
plt.legend(ncol=2, loc='lower right', fontsize=fontsize)
plt.show()


# %% SSIM
plt.figure(figsize=(15, 15))
plt.rc('legend', fontsize=18)
plt.rc(('xtick', 'ytick'), labelsize=30)

for is_in_X, linestyle, marker in zip(in_X_list, linestyles, markers):
    in_X_str = '$X$' if is_in_X else '$X_{ext}$'
    for warping, color in zip(warping_list, colors):
        ssims = ssim_array[int(is_in_X), int(warping), :]
        plt.plot(np.array(eta_list), ssims, marker=marker, markersize=16, 
                        linestyle=linestyle, linewidth=3,
                        color=color, label=methods_list[int(warping)] + ' / ' + in_X_str)

plt.xlabel(r'Regularization parameter $\eta$', fontsize=fontsize)
plt.ylabel('SSIM', fontsize=fontsize)
plt.xscale('log')
plt.legend(ncol=2, fontsize=fontsize)
plt.show()
# %%
