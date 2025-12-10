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


# %% 
# find indices of maximal, minimal and median amplitudes
idx_max = int(np.argmax(amp_array))
idx_min = int(np.argmin(amp_array))
median_val = float(np.median(amp_array))
idx_med = int(np.argmin(np.abs(amp_array - median_val)))

print("Max amplitude -> index:", idx_max, "value:", amp_array[idx_max])
print("Min amplitude -> index:", idx_min, "value:", amp_array[idx_min])
print("Median amplitude (closest) -> index:", idx_med, "value:", amp_array[idx_med], "median value:", median_val)


# %% plot scores % motion amplitude
save_fig = True
error_bars = False

path_fig = Path('../../Images/images_thèse/2024_article/ablation_study/noise/image_bank/score_vs_motion/')
if error_bars:
    path_fig = path_fig / Path('error_bars/')
Path(path_fig).mkdir(parents=True, exist_ok=True)

n_interpolation = 3
sigma_t = 3 * n_interpolation / 4
var_dz = 1 / 3
var_gdz = var_dz / (4 * np.pi * smoothness ** 2)
motion_std_array = amp_array * (var_gdz / (2 * np.pi ** 0.5 * sigma_t)) ** 0.5

# sort by motion standard deviation so the curve is monotonic in x
order_idx = np.argsort(motion_std_array)
ms_sorted = motion_std_array[order_idx]

n_bins = 25
assert len(ms_sorted) % n_bins == 0

motion_std_bin_array = (ms_sorted ** 2).reshape((n_bins, -1)).mean(axis=1) ** 0.5

fontsize = 18
plt.figure(figsize=(30, 10))
plt.rc(('xtick', 'ytick'), labelsize=fontsize)
plt.plot(ms_sorted, color='blue', marker='o')
plt.ylabel('Magnitude', fontsize=fontsize)
plt.xlabel('Batch', fontsize=fontsize)
plt.title('Motion std (sorted) used for simulation', fontsize=fontsize)
if save_fig:
    plt.savefig(path_fig / Path('motion_std_used_for_simulation_all.pdf'), dpi=100)
plt.show()

# %% PSNR
colors = ['red', 'blue']
Z_score = 1.96  # 1.96 for 95% confidence interval
n_samples = psnr_array.shape[-1]

markers = ['o', 's']
linestyles = ['dashed', 'dotted']

for alpha_ind, alpha in enumerate(alpha_list):

    fontsize = 46
    plt.figure(figsize=(15, 15))
    plt.rc('legend', fontsize=18)
    plt.rc(('xtick', 'ytick'), labelsize=fontsize)

    for is_in_X, linestyle, marker in zip(is_in_X_list, linestyles, markers):
        in_X_str = '$X$' if is_in_X else '$X_{ext}$'
        for warping, color in zip(warping_list, colors):
            warped_str = 'wh' if warping else 'wf'

            psnr_vals = psnr_array[int(is_in_X), int(warping), 0, alpha_ind, :]
            psnr_mean = psnr_vals.reshape(n_deform, batch_size).mean(axis=1).flatten()
            psnr_std = psnr_vals.reshape(n_deform, batch_size).std(axis=1).flatten()

            mean_sorted = psnr_mean[order_idx]
            std_sorted = psnr_std[order_idx]

            # Bin nearby motion-std values to reduce scatter and highlight trend.
            if n_bins < len(ms_sorted):
                mean_sorted_bins = mean_sorted.reshape((n_bins, -1)).mean(axis=1)
                std_sorted_bins = ((std_sorted ** 2).reshape((n_bins, -1)).mean(axis=1)) ** 0.5

                if error_bars:
                    plt.errorbar(motion_std_bin_array, mean_sorted_bins, yerr=std_sorted_bins, marker=marker, markersize=20,
                                linestyle=linestyle, linewidth=4, color=color,
                                label=warped_str + ' / ' + in_X_str, capsize=6, alpha=0.95)
                else:
                    plt.plot(motion_std_bin_array, mean_sorted_bins, marker=marker, markersize=20,
                        linestyle=linestyle, linewidth=4,
                        color=color, label=warped_str + ' / ' + in_X_str)
                    # plt.fill_between(motion_std_bin_array, mean_sorted_bins + std_sorted_bins , mean_sorted_bins - std_sorted_bins, color=color, alpha=0.1)

            else:
                plt.plot(ms_sorted, mean_sorted, marker=marker, markersize=20,
                        linestyle=linestyle, linewidth=4,
                        color=color, label=warped_str + ' / ' + in_X_str)

    plt.xlabel(r'Motion standard deviation', fontsize=fontsize)
    plt.ylabel('PSNR', fontsize=fontsize)
    plt.legend(ncol=2, loc='lower left', fontsize=fontsize)
    # plt.title(r'$\alpha$ = %d' % alpha, fontsize=fontsize)
    plt.tight_layout()
    if save_fig:
        plt.savefig(path_fig / Path('PSNR_vs_motion_std_alpha_%d.pdf' % alpha), dpi=100)

    plt.show()

# %% SSIM
for alpha_ind, alpha in enumerate(alpha_list):

    fontsize = 46
    plt.figure(figsize=(15, 15))
    plt.rc('legend', fontsize=18)
    plt.rc(('xtick', 'ytick'), labelsize=fontsize)

    for is_in_X, linestyle, marker in zip(is_in_X_list, linestyles, markers):
        in_X_str = '$X$' if is_in_X else '$X_{ext}$'
        for warping, color in zip(warping_list, colors):
            warped_str = 'wh' if warping else 'wf'

            ssim_vals = ssim_array[int(is_in_X), int(warping), 0, alpha_ind, :]
            ssim_mean = ssim_vals.reshape(n_deform, batch_size).mean(axis=1).flatten()
            ssim_std = ssim_vals.reshape(n_deform, batch_size).std(axis=1).flatten()

            mean_sorted = ssim_mean[order_idx]
            std_sorted = ssim_std[order_idx]

            # Bin nearby motion-std values to reduce scatter and highlight trend.
            if n_bins < len(ms_sorted):
                mean_sorted_bins = mean_sorted.reshape((n_bins, -1)).mean(axis=1)
                std_sorted_bins = ((std_sorted ** 2).reshape((n_bins, -1)).mean(axis=1)) ** 0.5

                if error_bars:
                    plt.errorbar(motion_std_bin_array, mean_sorted_bins, yerr=std_sorted_bins, marker=marker, markersize=20,
                                linestyle=linestyle, linewidth=4, color=color,
                                label=warped_str + ' / ' + in_X_str, capsize=6, alpha=0.95)
                else:
                    plt.plot(motion_std_bin_array, mean_sorted_bins, marker=marker, markersize=20,
                            linestyle=linestyle, linewidth=4,
                            color=color, label=warped_str + ' / ' + in_X_str)
                    # plt.fill_between(motion_std_bin_array, mean_sorted_bins + std_sorted_bins , mean_sorted_bins - std_sorted_bins, color=color, alpha=0.1)

            else:
                plt.plot(ms_sorted, mean_sorted, marker=marker, markersize=20,
                        linestyle=linestyle, linewidth=4,
                        color=color, label=warped_str + ' / ' + in_X_str)

    plt.xlabel(r'Motion standard deviation', fontsize=fontsize)
    plt.ylabel('SSIM', fontsize=fontsize)
    plt.legend(ncol=2, loc='lower left', fontsize=fontsize)
    # plt.title(r'$\alpha$ = %d' % alpha, fontsize=fontsize)

    if save_fig:
        plt.savefig(path_fig / Path('SSIM_vs_motion_std_alpha_%d.pdf' % alpha), dpi=100)

    plt.show()
# %%
