# %% [markdown]
"""
This script is used to reproduce the results given in Fig. 5e.
It visualizes the results obtained with fig_05e_compute.py over the test set of STL-10. 
These results have been pre-computed and stored in the 'raw_fig_05' folder.
"""

# %% Import bib
import numpy as np
import matplotlib.pyplot as plt
import json

from tqdm import tqdm
from pathlib import Path

from spyrit.misc.load_data import download_girder


#%% LOAD IMAGE DATA
img_size = 88  # full image side's size in pixels
meas_size = 64  # measurement pattern side's size in pixels (Hadamard matrix)
img_shape = (img_size, img_size)
meas_shape = (meas_size, meas_size)

n_deform = 500  # number of deformations used for data simulation
batch_size = 16  # each batch was subjected to one deformation


url_tomoradio = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
local_folder = Path.cwd() / Path('spyrit-examples/2025_dynamic_TIP/') / Path('raw_fig_05')  # replace with your path
id_files = [
    "690b5f7304d23f6e964b1434",  # motion_params.json
    "690b5f7304d23f6e964b1437"   # scores.json
]
try:
    download_girder(url_tomoradio, id_files, local_folder)

except Exception as e:
    print("Unable to download from the Tomoradio warehouse")
    print(e)


# %% Instantiate scores arrays
alpha_list = [1000, 500, 300, 100, 50, 25, 10]
reg_list = ['H1']
is_in_X_list = [False, True]
warping_list = [False, True]

eta_nowarp_list_list = [[2e-2, 3e-2, 4e-2, 1e-1, 2e-1, 2e-1, 4e-1]]
eta_nowarp_inX_list_list = [[1e-1, 2e-1, 2e-1, 2e-1, 3e-1, 4e-1, 4e-1]]

eta_warp_list_list = [[5e-2, 5e-2, 7e-2, 1e-1, 2e-1, 2e-1, 4e-1]]
eta_warp_inX_list_list = [[1e-1, 2e-1, 2e-1, 2e-1, 3e-1, 4e-1, 4e-1]]

etas = np.zeros((len(is_in_X_list), len(warping_list), len(reg_list), len(alpha_list)))
for is_in_X in is_in_X_list:
    for warping in warping_list:
        if is_in_X and warping:
            etas[int(is_in_X), int(warping), :, :] = np.array(eta_warp_inX_list_list)
        elif is_in_X and not(warping):
            etas[int(is_in_X), int(warping), :, :] = np.array(eta_nowarp_inX_list_list)
        elif not(is_in_X) and warping:
            etas[int(is_in_X), int(warping), :, :] = np.array(eta_warp_list_list)
        else:
            etas[int(is_in_X), int(warping), :, :] = np.array(eta_nowarp_list_list)         

data_size = batch_size * n_deform

psnr_array = np.zeros((len(is_in_X_list), len(warping_list), len(reg_list), len(alpha_list), data_size))
ssim_array = np.zeros((len(is_in_X_list), len(warping_list), len(reg_list), len(alpha_list), data_size))
amp_array = np.zeros(n_deform)


# %% READ JSON
psnr_array = np.zeros((len(is_in_X_list), len(warping_list), len(reg_list), len(alpha_list), data_size))
ssim_array = np.zeros((len(is_in_X_list), len(warping_list), len(reg_list), len(alpha_list), data_size))

with open(str(local_folder / Path('scores.json')), "r") as infile:
    scores = json.load(infile)

for is_in_X in is_in_X_list:
    for warping in warping_list:
        for alpha_ind, alpha in enumerate(tqdm(alpha_list, desc='Iterating over noise level')):
            for reg_ind, reg in enumerate(reg_list):
                scores_alpha = scores['alpha_%d' % alpha]
                psnr_array[int(is_in_X), int(warping), reg_ind, alpha_ind, :] = np.array(scores_alpha[reg + '_warping_' + str(warping) +'_in_X_' + str(is_in_X)]['PSNRs'])
                ssim_array[int(is_in_X), int(warping), reg_ind, alpha_ind, :] = np.array(scores_alpha[reg + '_warping_' + str(warping) +'_in_X_' + str(is_in_X)]['SSIMs'])


with open(str(local_folder / Path('motion_params.json')), "r") as infile:
    motion_params = json.load(infile)

amp_array = np.array(motion_params['amp'])
smoothness = np.array(motion_params['smoothness'])

# %% PLOT MOTION PARAMS
fontsize = 18
plt.figure(figsize=(30, 10))
plt.rc(('xtick', 'ytick'), labelsize=fontsize)
plt.plot(amp_array, color='blue', marker='o')
plt.ylabel('Magnitude', fontsize=fontsize)
plt.xlabel('Batch', fontsize=fontsize)
plt.title('Motion magnitude used for data simulation', fontsize=fontsize)
plt.show()


# %% PLOT PSNR mean % alpha
colors = ['red', 'blue']
Z_score = 1.96  # 1.96 for 95% confidence interval
n_samples = psnr_array.shape[-1]

markers = ['o', 's']
linestyles = ['dashed', 'dotted']

fontsize = 46
plt.figure(figsize=(15, 15))
plt.rc('legend', fontsize=18)
plt.rc(('xtick', 'ytick'), labelsize=fontsize)

for is_in_X, linestyle, marker in zip(is_in_X_list, linestyles, markers):
    in_X_str = '$X$' if is_in_X else '$X_{ext}$'
    for warping, color in zip(warping_list, colors):
        warped_str = 'wh' if warping else 'wf'

        psnr_mean = psnr_array[int(is_in_X), int(warping), 0, :, :].mean(axis=1)
        psnr_std = psnr_array[int(is_in_X), int(warping), 0, :, :].std(axis=1)
        plt.plot(alpha_list, psnr_mean, marker=marker, markersize=16, 
                    linestyle=linestyle, linewidth=3,
                    color=color, label=warped_str + ' / ' + in_X_str)
        # plt.fill_between(alpha_list, psnr_mean + Z_score * psnr_std / n_samples ** 0.5, psnr_mean - Z_score * psnr_std / n_samples ** 0.5, color=color, alpha=0.1)
        # plt.fill_between(alpha_list, psnr_mean + psnr_std , psnr_mean - psnr_std, color=color, alpha=0.1)

plt.xlabel(r'Max intensity $\alpha$', fontsize=fontsize)
plt.ylabel('PSNR', fontsize=fontsize)
plt.legend(ncol=2, loc='lower right', fontsize=fontsize)
plt.show()

std_error = Z_score * psnr_std / n_samples ** 0.5
print("Standard error max : ", std_error.max())

# %% PLOT SSIM mean % alpha
colors = ['red', 'blue']
Z_score = 1.96  # 1.96 for 95% confidence interval
n_samples = ssim_array.shape[-1]

fontsize = 46
plt.figure(figsize=(15, 15))
plt.rc('legend', fontsize=18)
plt.rc(('xtick', 'ytick'), labelsize=fontsize)

for is_in_X, linestyle, marker in zip(is_in_X_list, linestyles, markers):
    in_X_str = '$X$' if is_in_X else '$X_{ext}$'
    for warping, color in zip(warping_list, colors):
        warped_str = 'wh' if warping else 'wf'

        ssim_mean = ssim_array[int(is_in_X), int(warping), 0, :, :].mean(axis=1)
        ssim_std = ssim_array[int(is_in_X), int(warping), 0, :, :].std(axis=1)
        plt.plot(alpha_list, ssim_mean ,marker=marker, markersize=16, 
                    linestyle=linestyle, linewidth=3,
                    color=color, label=warped_str + ' / ' + in_X_str)
        # plt.fill_between(alpha_list, ssim_mean + Z_score * ssim_std / n_samples ** 0.5, ssim_mean - Z_score * ssim_std / n_samples ** 0.5, color=color, alpha=0.1)
        # plt.fill_between(alpha_list, ssim_mean + ssim_std , ssim_mean - ssim_std, color=color, alpha=0.1)

plt.xlabel(r'Max intensity $\alpha$', fontsize=fontsize)
plt.ylabel('SSIM', fontsize=fontsize)
plt.legend(ncol=2, loc='lower right', fontsize=fontsize)
plt.show()

std_error = Z_score * ssim_std / n_samples ** 0.5
print("Standard error max : ", std_error.max())

# %%
