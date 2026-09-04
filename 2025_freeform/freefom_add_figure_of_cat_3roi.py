# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 16:48:48 2025

The script generates Fig. 6 of the paper.

It now iterates over the three freeform ROI datasets acquired on
2026-09-04 -- 'cat' (head+legs, 4096 px), 'cat-head' (2048 px) and
'cat-legs' (1024 px) -- producing one figure and one set of
SNR / MSNR / MSNRc tables per ROI. The FH2 (Walsh) acquisition and its
black reference are mask-independent full-frame measurements, so they
are shared by all three ROIs; H1/S1/RS integration times are scaled up
as the ROI shrinks so that N_pixel * ti (hence the acquisition time
budget) stays the same for all three.

@author: ducros
"""
# -*- coding: utf-8 -*-

#%% imports and global configuration
import json
import ast
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline
from PIL import Image
import os
os.chdir("d:/hspc/scripts")

from spyrit.misc.disp import imagesc, add_colorbar
from spyrit.misc.sampling import reindex
from spyrit.core.meas import HadamSplit2d, FreeformLinear, FreeformLinearSplit
from spyrit.core.prep import Unsplit
from spyrit.core.torch import walsh_matrix, ifwht
from spyrit.misc.walsh_hadamard import walsh_S_matrix, ifwalsh_S_torch

ti = 2      # base integration time (ms) -- FH2/MH2 are mask-independent
            # full-frame measurements, so they always use this value
h = 128     # image size hxh
norm = 32768 * ti  # time budget in ms, kept constant across the 3 ROIs
                   # so their SNR/MSNR are directly comparable
fig_folder = Path('../result/freeform/figures_cat_3roi')
fig_folder.mkdir(parents=True, exist_ok=True)
save_tag = True
plot_tag = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_folder = Path(r"../data/2026-09-04_freeform_publication")

# FH2 (Walsh) and every black reference are only acquired once, under the
# 'cat' object name, and reused for all three ROIs below.
walsh_title = r'obj_cat_source_white_LED_Walsh_im_128x128_ti_' + str(ti) + 'ms_zoom_x1'
black_obj_slug = 'cat'

method = 'substraction'    # 'classical'#
NR = 2                      # Number of repetitions in the "substraction" method
acq_list = ["FH2", "H1", "MH2", "S1", "RS"]
method_list = ["FH2", "RS", "MH2", "H1", "S1"]  # order used by the metric arrays below

# roi_scale = N_pixel(cat) / N_pixel(roi): H1 uses ti*4*roi_scale, S1/RS
# use ti*8*roi_scale, so that N_pixel*ti (i.e. the acquisition time
# budget of these ROI-adaptive methods) is the same for the 3 ROIs.
objects_cfg = [
    dict(label='cat',      obj_slug='cat',      mask_png='mask_head_&_legs.png', roi_scale=1),
    dict(label='cat-head', obj_slug='cat-head', mask_png='mask_head.png',        roi_scale=2),
    dict(label='cat-legs', obj_slug='cat-legs', mask_png='mask_legs.png',        roi_scale=4),
]

ref_method = 'S1'   # Choose one scan mode into the method_list, to compare MSNR from the other scan mode
ref_idx = method_list.index(ref_method)

#%% Plot options
fs = 9                  # Font size
dpi_fig = 600
cbar_pos = 'bottom'     # colorbar position
plot_tag = False
print_metric = 'PSNR'   # 'SNR'#
print_value = False
dark_plot_tag = True
spl_plot_tag = True

lambda_central_list = [515, 515, 1800, 1800]  # no signal below 15 and above 2038
nc_list = [16, 3, 16, 3]
results = {}



def load_spihim(data_folder, data_title):

    suffix = {"data": "_spectraldata.npz", "metadata": "_metadata.json"}

    # Spectral data in numpy
    exp_data = [
        np.load(data_folder / title / (title + suffix["data"]))["spectral_data"]
        for title in data_title
        ]

    # Metadata
    patterns = [[] for _ in range(len(data_title))]
    wavelengths = [[] for _ in range(len(data_title))]

    for ii, title in enumerate(data_title):

        file = open(data_folder / title / (title + suffix["metadata"]), "r")
        json_metadata = json.load(file)[4]
        file.close()

        # Pattern order
        # replace "np.int32(" with an empty string and ")" with an empty string
        tmp = json_metadata["patterns"]
        tmp = tmp.replace("np.int32(", "").replace(")", "")
        patterns[ii] = ast.literal_eval(tmp)

        # Wavelength
        wavelengths[ii] = ast.literal_eval(json_metadata["wavelengths"])

    return exp_data, wavelengths, patterns


def compute_roi_snr(img_im1, img_im2, roi_mask):
    """Mean over both repetitions within roi_mask, and std of their
    difference (bias-corrected for the variance increase caused by the
    subtraction), used to derive the SNR of that ROI."""
    v1 = img_im1[roi_mask]
    v2 = img_im2[roi_mask]
    roi_mean = torch.mean((v1 + v2) / 2)
    roi_std = torch.std((v1 - v2) / 2**.5)
    roi_max = ((v1 + v2) / 2).max()
    return roi_mean, roi_std, roi_max


#%% Dark measurement / stray light reference
# This is a diagnostic-only, ROI-independent measurement (mu_dark is not
# actually subtracted below -- see the commented-out line), so it is
# loaded once and shared by all three ROI iterations.
dark_folder = Path(r"../data/2025-09-11_freeform_SNR")
dark_title = [r'obj_black_source_No source_raster_cat_4096_im_128x128_ti_8ms_zoom_x1']

data_dark, _, _ = load_spihim(dark_folder, dark_title)
mu_dark = data_dark[0].mean(axis=0)

if dark_plot_tag:
    plt.figure()
    plt.plot(mu_dark[100:700])
    plt.title('µ dark')
#%% MAIN LOOP over the 3 ROI datasets (cat, cat-head, cat-legs)
for cfg in objects_cfg:

    label = cfg['label']
    obj_slug = cfg['obj_slug']
    print()
    print('#' * 60)
    print(f'### ROI = {label} ###')
    print('#' * 60)

    ti_H1 = int(ti * 4 * cfg['roi_scale'])
    ti_S1 = int(ti * 8 * cfg['roi_scale'])
    ti_RS = ti_S1

    # ---- reload the mask in full size (128x128) ----
    # Acquisitions save the ready-to-use mask directly as a plain (h x h)
    # binary PNG in the shared Walsh folder -- just read it, no metadata
    # involved.
    mask_png_path = data_folder / walsh_title / cfg['mask_png']
    mask_full = np.array(Image.open(mask_png_path))
    mask_full = mask_full > mask_full.min()

    mask = torch.from_numpy(mask_full).to(device=device)

    ind_array = np.where(mask_full == True)
    ind_array = (torch.from_numpy(ind_array[0]).to(device=device),
                 torch.from_numpy(ind_array[1]).to(device=device))

    N_pixel = len(ind_array[0])

    if plot_tag:
        imagesc(mask.cpu())
        plt.title(f'mask -- {label}')

    # ---- read data ----
    data_title = [
        walsh_title,
        r'obj_' + obj_slug + r'_source_white_LED_hadam1d_8192_im_128x128_ti_' + str(ti_H1) + 'ms_zoom_x1',
        r'obj_' + obj_slug + r'_source_white_LED_hadam2d_32768_im_128x128_ti_' + str(ti) + 'ms_zoom_x1',
        r'obj_' + obj_slug + r'_source_white_LED_smatrix_4095_im_128x128_ti_' + str(ti_S1) + 'ms_zoom_x1',
        r'obj_' + obj_slug + r'_source_white_LED_Raster_im_128x128_ti_' + str(ti_RS) + 'ms_zoom_x1']

    black_title = [
        r'obj_' + black_obj_slug + r'_source_white_LED_black_4096_im_128x128_ti_' + str(ti) + 'ms_zoom_x1',
        r'obj_' + black_obj_slug + r'_source_white_LED_black_4096_im_128x128_ti_' + str(ti_H1) + 'ms_zoom_x1',
        r'obj_' + black_obj_slug + r'_source_white_LED_black_4096_im_128x128_ti_' + str(ti) + 'ms_zoom_x1',
        r'obj_' + black_obj_slug + r'_source_white_LED_black_4096_im_128x128_ti_' + str(ti_S1) + 'ms_zoom_x1',
        r'obj_' + black_obj_slug + r'_source_white_LED_black_4096_im_128x128_ti_' + str(ti_RS) + 'ms_zoom_x1']

    black_exp, _, _ = load_spihim(data_folder, black_title)
    data_exp, wavelength, patterns = load_spihim(data_folder, data_title)

    # ---- denoise black (stray light) ----
    spl_arr = np.zeros((len(black_exp), len(wavelength[0])))

    for j in range(len(black_exp)):

        y = black_exp[j].mean(axis=0)  # - mu_dark

        if method == 'substraction':
            spl = make_smoothing_spline(wavelength[0], y, lam=1e3)
        else:  # method == 'classical':
            if j <= 2:
                spl = make_smoothing_spline(wavelength[0], black_exp[j][1, :], lam=100)
            else:
                RS_mean = black_exp[4].mean(axis=1)
                ind_min = np.argmin(RS_mean)
                spl = make_smoothing_spline(wavelength[0], black_exp[4][ind_min, :], lam=100)

        spl_np = spl(wavelength[0])
        spl_arr[j, :] = spl_np

        if spl_plot_tag:
            plt.figure()
            plt.plot(wavelength[0], y, marker="o", color="blue")
            plt.plot(wavelength[0], spl_arr[j, :], color="red")
            plt.title(acq_list[j] + f' - spline fit for black patterns ({label})')

    # Dark noise per method/band, saved as a diagnostic
    lambda_central_list_dark = [515, 515, 1800, 1800]
    nc_list_dark = [16, 3, 16, 3]
    ind_expe_list = [0, 4, 2, 1, 3]  # reorder

    sigma_m = np.zeros((len(black_exp), len(lambda_central_list_dark)))

    for j, jj in enumerate(ind_expe_list):
        for ll in range(len(lambda_central_list_dark)):
            lambda_central = lambda_central_list_dark[ll]
            nc = nc_list_dark[ll]
            lambda_min = lambda_central - nc + 1
            lambda_max = lambda_central + nc

            yi = black_exp[jj][:, lambda_min:lambda_max]
            sigma_m[j, ll] = np.sqrt(np.var(yi, axis=0).mean())

    np.save(fig_folder / f'sigma_dark_{label}', sigma_m)

    # ---- substract stray light ----
    sub_plot_tag = False
    acqui_size = []
    for j in range(len(data_exp)):

        acqui_size.append(int(data_exp[j].shape[0] / NR))
        for nM in range(acqui_size[j] * NR):

            data_exp[j][nM, :] = data_exp[j][nM, :] - spl_arr[j, :]

            if sub_plot_tag:
                if nM == 0 or nM == acqui_size[j]:
                    Color = 'blue' if nM == 0 else 'red'
                    plt.figure()
                    plt.plot(wavelength[0], data_exp[j][nM, :], color=Color)
                    plt.title(acq_list[j] + f' - first spectrum of each repetiton ({label})')

    # ---- reorder measurements for full 2D Hadamard ----
    for nR in range(NR):
        data_exp[0][acqui_size[0] * nR:acqui_size[0] * (nR + 1), :] = reindex(
                data_exp[0][acqui_size[0] * nR:acqui_size[0] * (nR + 1), :],
                np.array(patterns[0]),
                axis="rows",
                inverse_permutation=True
                )

    # --- Rebuild ind_array from the acquisition metadata (native scan order) ---
    # `ind_array`, as built above, comes from np.where(mask==True) on the
    # mask PNG -- that recovers the right *set* of pixels, but not their
    # acquisition order: the mask PNG is saved 180deg-rotated relative to
    # the DMD/scan convention used to build the H1/S1/RS measurement
    # matrices (acquisition_parameters.mask_index in the metadata), so
    # np.where ends up traversing that set in *reverse* scan order. This
    # doesn't matter for the 2D Hadamard methods (FH2/MH2), which only
    # use mask as an order-independent boolean selection, but it breaks
    # the arbitrary-shape 1D methods (H1/S1/RS): their k-th measurement
    # must land on the k-th pixel of the original scan order.
    # Fix: read the scan order from metadata (mask_index/x_mask_coord/
    # y_mask_coord) and flip it 180deg (row -> h-1-row, col -> h-1-col) to
    # land in the mask PNG's frame -- this preserves the scan order while
    # matching mask's pixel positions exactly (checked below).
    roi_meta_title = data_title[4]  # any ROI-adaptive acquisition works; they all share the same ROI
    with open(data_folder / roi_meta_title / (roi_meta_title + '_metadata.json'), 'r') as file:
        roi_acquisition_parameters = json.load(file)[4]

    x_mask_coord = ast.literal_eval(roi_acquisition_parameters['x_mask_coord'])
    y_mask_coord = ast.literal_eval(roi_acquisition_parameters['y_mask_coord'])
    mask_index = np.array(ast.literal_eval(roi_acquisition_parameters['mask_index']), dtype=int)
    x0, y0 = int(x_mask_coord[0]), int(y_mask_coord[0])
    w_len = int(x_mask_coord[1] - x0)

    row = h - 1 - (y0 + mask_index // w_len)
    col = h - 1 - (x0 + mask_index % w_len)
    ind_array = (torch.from_numpy(row).to(device=device), torch.from_numpy(col).to(device=device))
    N_pixel = len(ind_array[0])

    assert set(zip(row.tolist(), col.tolist())) == set(zip(*np.where(mask.cpu().numpy()))), \
        f'metadata-based ind_array does not match mask PNG for {label} -- check x_mask_coord/y_mask_coord/mask_index'

    # --- Combined ROI, for the SNR measurement ---
    # For the 'cat' ROI the mask is made of two disjoint blobs (head and
    # legs), but since they partition the whole freeform mask, their sum
    # is simply the mask itself -- so a single ROI covering both blobs is
    # just `mask`. For 'cat-head'/'cat-legs', mask is already the
    # single-blob ROI.
    roi_mask = mask

    f, ax = plt.subplots(5, len(lambda_central_list),
                          figsize=(len(lambda_central_list) * 2, 10))

    # axes: [method, wavelength band]
    std = np.empty([5, len(lambda_central_list)])
    moy = np.empty([5, len(lambda_central_list)])
    snr = np.empty([5, len(lambda_central_list)])
    maxi = np.empty([5, len(lambda_central_list)])
    psnr = np.empty([5, len(lambda_central_list)])

    for ll in range(1):#len(lambda_central_list)):
        print("================================= Lambda = " + str(lambda_central_list[ll]) + " nm / band = " + str(nc_list[ll]) + " =================================")
        # Spectral binning
        lambda_central = lambda_central_list[ll]
        nc = nc_list[ll]
        lambda_min = lambda_central - nc + 1
        lambda_max = lambda_central + nc
        lambda_n = lambda_max - lambda_min

        data_bin = [[data_exp[ii][acqui_size[ii] * nR:acqui_size[ii] * (nR + 1), lambda_min:lambda_max].sum(axis=1)
                    for ii in range(len(data_title))]
                    for nR in range(NR)]

        # Convert to torch tensors
        data_bin = [[torch.from_numpy(data_bin[nR][ii]).to(device=device, dtype=torch.float32)
                    for ii in range(len(data_title))]
                    for nR in range(NR)]

        #--------------------------------------------------------------------------
        # 2D Hadamard full
        #--------------------------------------------------------------------------
        if print_value:
            print('== Hadamard 2D full ==')

        indx_dataset = 0
        indx_graph = 0

        y = data_bin[0][indx_dataset]
        y2 = data_bin[1][indx_dataset]

        # integration time scaling
        y = y * (h * h * 2) / norm
        y2 = y2 * (h * h * 2) / norm
        
        if print_value:
            print('max of meas:', y.max())
            print('min of meas:', y.min())

        meas_op = HadamSplit2d(h, device=device)
        prep = Unsplit()
        prep = prep.to(device=device)

        # NB: fast_pinv's output frame is 180deg-flipped relative to
        # mask/ind_array's frame -- rotate it back so ROI selection below
        # (roi = x_H2dF[mask]) picks the right pixels.
        x_H2dF = torch.rot90(meas_op.fast_pinv(prep(y)), 2, dims=(0, 1))
        x_H2dF_2 = torch.rot90(meas_op.fast_pinv(prep(y2)), 2, dims=(0, 1))

        if plot_tag:
            plt.figure()
            plt.imshow(x_H2dF.cpu())
            plt.colorbar()
            plt.title(f'H2F, image 0 ({label})')

            plt.figure()
            plt.imshow(x_H2dF_2.cpu())
            plt.colorbar()
            plt.title(f'H2F, image 1 ({label})')

            x_H2dF_sub = x_H2dF - x_H2dF_2
            plt.figure()
            plt.imshow(x_H2dF_sub.cpu())
            plt.colorbar()
            plt.title(f'H2F, diff ({label})')

        moy[indx_graph, ll], std[indx_graph, ll], maxi[indx_graph, ll] = \
            compute_roi_snr(x_H2dF, x_H2dF_2, roi_mask)

        snr[indx_graph, ll] = moy[indx_graph, ll] / std[indx_graph, ll]

        if print_value:
            print(f'std = {std[indx_graph, ll]}')
            print(f'moy = {moy[indx_graph, ll]}')
            print(f'snr = {snr[indx_graph, ll]}')

        #--------------------------------------------------------------------------
        # RASTER SCAN
        #--------------------------------------------------------------------------
        if print_value:
            print('== Raster Scan ==')

        indx_dataset = 4
        indx_graph = 1

        y = data_bin[0][indx_dataset]
        y2 = data_bin[1][indx_dataset]

        y = y * N_pixel / norm
        y2 = y2 * N_pixel / norm
        
        if print_value:
            print('max of meas:', y.max())
            print('min of meas:', y.min())

        H = torch.eye(N_pixel)
        meas_1d = FreeformLinear(H,
                        meas_shape=(h, h),
                        index_mask=torch.stack(ind_array),
                        device=device
                        )

        x_rec_2 = y
        x_I1d = torch.zeros_like(x_H2dF)
        x_I1d[ind_array[0], ind_array[1]] = x_rec_2
        x_I1d[~mask] = x_rec_2.min()

        x_rec_2 = y2
        x_I1d_2 = torch.zeros_like(x_H2dF)
        x_I1d_2[ind_array[0], ind_array[1]] = x_rec_2
        x_I1d_2[~mask] = x_rec_2.min()

        if plot_tag:
            plt.figure()
            plt.imshow(x_I1d.cpu())
            plt.colorbar()
            plt.title(f'RS ({label})')

            plt.figure()
            plt.imshow(x_I1d_2.cpu())
            plt.colorbar()
            plt.title(f'RS image 1 ({label})')

            x_I1d_sub = x_I1d - x_I1d_2
            plt.figure()
            plt.imshow(x_I1d_sub.cpu())
            plt.colorbar()
            plt.title(f'RS image sub ({label})')

        moy[indx_graph, ll], std[indx_graph, ll], maxi[indx_graph, ll] = \
            compute_roi_snr(x_I1d, x_I1d_2, roi_mask)

        if moy[indx_graph, ll] < 0:
            moy[indx_graph, ll] = 0
            print('!!!!! Warning, mean < 0 !!!!!!!!!!!!!!!')

        snr[indx_graph, ll] = moy[indx_graph, ll] / std[indx_graph, ll]
        psnr[indx_graph, ll] = 20 * np.log10(maxi[indx_graph, ll] / std[indx_graph, ll])

        if print_value:
            print(f'std = {std[indx_graph, ll]}')
            print(f'moy = {moy[indx_graph, ll]}')
            print(f'snr = {snr[indx_graph, ll]}')

        #--------------------------------------------------------------------------
        # Masked 2D Hadamard
        #--------------------------------------------------------------------------
        if print_value:
            print('== Hadamard 2D masked ==')

        indx_dataset = 2
        indx_graph = 2

        y = data_bin[0][indx_dataset]
        y2 = data_bin[1][indx_dataset]

        y = y * (h * h * 2) / norm
        y2 = y2 * (h * h * 2) / norm

        if print_value:
            print('max of meas:', y.max())
            print('min of meas:', y.min())

        meas_op = HadamSplit2d(h, device=device)
        prep = Unsplit().to(device=device)

        # NB: same 180deg flip as FH2 -- fast_pinv's output frame does not
        # match mask/ind_array's frame otherwise (see FH2 section above).
        x_H2dM = torch.rot90(meas_op.fast_pinv(prep(y)), 2, dims=(0, 1))
        x_H2dM[~mask] = x_H2dM[mask].min()

        x_H2dM_2 = torch.rot90(meas_op.fast_pinv(prep(y2)), 2, dims=(0, 1))
        x_H2dM_2[~mask] = x_H2dM_2.min()

        if plot_tag:
            plt.figure()
            plt.imshow(x_H2dM.cpu())
            plt.colorbar()
            plt.title(f'H2M, image 0 ({label})')

            plt.figure()
            plt.imshow(x_H2dM_2.cpu())
            plt.colorbar()
            plt.title(f'H2dM, image 1 ({label})')

            x_H2dM_sub = x_H2dM - x_H2dM_2
            plt.figure()
            plt.imshow(x_H2dM_sub.cpu())
            plt.colorbar()
            plt.title(f'H2dM, diff ({label})')

        moy[indx_graph, ll], std[indx_graph, ll], maxi[indx_graph, ll] = \
            compute_roi_snr(x_H2dM, x_H2dM_2, roi_mask)

        snr[indx_graph, ll] = moy[indx_graph, ll] / std[indx_graph, ll]

        if print_value:
            print(f'std = {std[indx_graph, ll]}')
            print(f'moy = {moy[indx_graph, ll]}')
            print(f'snr = {snr[indx_graph, ll]}')

        #--------------------------------------------------------------------------
        # Arbitrary shape -- Hadamard matrix 1D
        #--------------------------------------------------------------------------
        if print_value:
            print('== Hadamard 1D ==')

        indx_dataset = 1
        indx_graph = 3

        y = data_bin[0][indx_dataset]
        y2 = data_bin[1][indx_dataset]

        y = y * (N_pixel * 2) / norm
        y2 = y2 * (N_pixel * 2) / norm
        
        if print_value:
            print('max of meas:', y.max())
            print('min of meas:', y.min())

        H = walsh_matrix(N_pixel)
        meas_1d = FreeformLinearSplit(H,
                            meas_shape=(h, h),
                            index_mask=torch.stack(ind_array),
                            device=device)
        prep = Unsplit()

        x_rec_2 = ifwht(prep(y))
        x_H1d = torch.zeros_like(x_H2dF)
        x_H1d[ind_array[0], ind_array[1]] = x_rec_2
        x_H1d[~mask] = x_rec_2.min()

        x_rec_2 = ifwht(prep(y2))
        x_H1d_2 = torch.zeros_like(x_H2dF)
        x_H1d_2[ind_array[0], ind_array[1]] = x_rec_2
        x_H1d_2[~mask] = x_rec_2.min()

        if plot_tag:
            plt.figure()
            plt.imshow(x_H1d.cpu())
            plt.colorbar()
            plt.title(f'H1, image 0 ({label})')

            plt.figure()
            plt.imshow(x_H1d_2.cpu())
            plt.colorbar()
            plt.title(f'H1, image 1 ({label})')

            x_H1d_sub = x_H1d - x_H1d_2
            plt.figure()
            plt.imshow(x_H1d_sub.cpu())
            plt.colorbar()
            plt.title(f'H1, diff ({label})')

        moy[indx_graph, ll], std[indx_graph, ll], maxi[indx_graph, ll] = \
            compute_roi_snr(x_H1d, x_H1d_2, roi_mask)

        snr[indx_graph, ll] = moy[indx_graph, ll] / std[indx_graph, ll]

        if print_value:
            print(f'std = {std[indx_graph, ll]}')
            print(f'moy = {moy[indx_graph, ll]}')
            print(f'snr = {snr[indx_graph, ll]}')

        #--------------------------------------------------------------------------
        # Arbitrary shape -- S matrix
        #--------------------------------------------------------------------------
        if print_value:
            print('== S-matrix 1D ==')

        indx_dataset = 3
        indx_graph = 4

        y = data_bin[0][indx_dataset]
        y2 = data_bin[1][indx_dataset]

        # NB: we use N_pixel here, not N_pixel-1, in accordance with the experiment
        y = y * N_pixel / norm
        y2 = y2 * N_pixel / norm

        if print_value:
            print('max of meas:', y.max())
            print('min of meas:', y.min())

        H = torch.from_numpy(walsh_S_matrix(N_pixel - 1))

        ind_array_0_S = ind_array[0][:-1]
        ind_array_1_S = ind_array[1][:-1]

        meas_1d = FreeformLinear(H,
                            meas_shape=(h, h),
                            index_mask=torch.stack((ind_array_0_S, ind_array_1_S)),
                            device=device)
        # NB: pixel index 0 (i.e. ind_array_0_S[0], ind_array_1_S[0]) is a
        # structural artifact of the cyclic S-matrix construction, not a
        # real measurement: ifwalsh_S_torch effectively reconstructs it as
        # a residual/"missing-mode" bin, so it swings far outside the
        # range of every other pixel regardless of wavelength band. Left
        # as is, that one bad pixel also contaminates the whole
        # out-of-ROI background below, since it's set to x_rec_2.min().
        # Exclude it from the background-fill value, then zero it out for
        # display.

        x_rec_2 = ifwalsh_S_torch(y)
        x_bg = x_rec_2[1:].min()
        x_rec_2[0] = 0
        x_S1d = torch.zeros_like(x_H2dF)
        x_S1d[ind_array_0_S, ind_array_1_S] = x_rec_2
        x_S1d[~mask] = x_bg

        x_rec_2 = ifwalsh_S_torch(y2)
        x_bg = x_rec_2[1:].min()
        x_rec_2[0] = 0
        x_S1d_2 = torch.zeros_like(x_H2dF)
        x_S1d_2[ind_array_0_S, ind_array_1_S] = x_rec_2
        x_S1d_2[~mask] = x_bg

        if plot_tag:
            plt.figure()
            plt.imshow(x_S1d.cpu())
            plt.colorbar()
            plt.title(f'SM ({label})')

            plt.figure()
            plt.imshow(x_S1d_2.cpu())
            plt.colorbar()
            plt.title(f'SM, image 1 ({label})')

            x_S1d_sub = x_S1d - x_S1d_2
            plt.figure()
            plt.imshow(x_S1d_sub.cpu())
            plt.colorbar()
            plt.title(f'SM, diff ({label})')

        moy[indx_graph, ll], std[indx_graph, ll], maxi[indx_graph, ll] = \
            compute_roi_snr(x_S1d, x_S1d_2, roi_mask)

        snr[indx_graph, ll] = moy[indx_graph, ll] / std[indx_graph, ll]

        if print_value:
            print(f'std = {std[indx_graph, ll]}')
            print(f'moy = {moy[indx_graph, ll]}')
            print(f'snr = {snr[indx_graph, ll]}')

        #--------------------------------------------------------------------------
        # Plot all images on same figure
        #--------------------------------------------------------------------------
        im = ax[0, ll].imshow(x_H2dF.cpu(), cmap="gray")
        cbar = add_colorbar(im, cbar_pos)
        cbar.ax.tick_params(labelsize=fs - 3)

        im = ax[1, ll].imshow(x_I1d.cpu(), cmap="gray")
        cbar = add_colorbar(im, cbar_pos)
        cbar.ax.tick_params(labelsize=fs - 3)

        im = ax[2, ll].imshow(x_H2dM.cpu(), cmap="gray")
        cbar = add_colorbar(im, cbar_pos)
        cbar.ax.tick_params(labelsize=fs - 3)

        im = ax[3, ll].imshow(x_H1d.cpu(), cmap="gray")
        cbar = add_colorbar(im, cbar_pos)
        cbar.ax.tick_params(labelsize=fs - 3)

        im = ax[4, ll].imshow(x_S1d.cpu(), cmap="gray")
        cbar = add_colorbar(im, cbar_pos)
        cbar.ax.tick_params(labelsize=fs - 2)

        for j in range(5):
            ax[j, ll].set_xticks([])
            ax[j, ll].set_yticks([])

    # ---- End of main loop: labels, save figure ----
    for j in range(len(method_list)):
        ax[j, 0].set_ylabel(method_list[j], fontsize=fs)

    for j in range(4):
        lambda_central = lambda_central_list[j]
        nc = nc_list[j]
        lambda_min = lambda_central - nc + 1
        lambda_max = lambda_central + nc
        lambda_n = lambda_max - lambda_min

        ax[0, j].set_title(
            f'{lambda_n} channel' +
                ('s\n' if lambda_n > 1 else ' \n') +
                f'in [{wavelength[0][lambda_min]:0.0f}, {wavelength[0][lambda_max]:0.0f}) nm',
            fontsize=fs)

    f.suptitle(f'ROI: {label}')
    plt.tight_layout()

    if save_tag:
        # Save the actual grid figure object `f`, not whatever plt considers
        # the "current" figure -- otherwise a stale/unrelated figure could
        # silently end up saved instead, producing a near-empty file.
        fil_name = f'figure_{label}_ti_{ti}ms.pdf'
        f.savefig(fig_folder / fil_name, bbox_inches='tight', dpi=dpi_fig)

    # ---- MSNR ----
    save_array = True

    # moy == 0 (defensive clamp applied when the raw mean came out negative,
    # e.g. RS at low signal -- see "!!!!! Warning, ... mean < 0" above) is
    # not a real signal estimate. Replace it with the average moy of the
    # other scan modes at the same wavelength band (i.e. average over the
    # method axis, excluding the zeroed-out method(s)) *before* computing
    # MSNR, so the imputed moy is combined with that method's own (valid,
    # non-zero) std.
    zeroed = (moy == 0)
    for ll in range(moy.shape[1]):
        zeroed_methods = np.where(zeroed[:, ll])[0]
        if len(zeroed_methods) == 0:
            continue
        valid_methods = np.where(~zeroed[:, ll])[0]
        if len(valid_methods) == 0:
            print(f'Warning: all methods are zero at band={ll} -- cannot impute moy')
            continue
        replacement = moy[valid_methods, ll].mean()
        moy[zeroed_methods, ll] = replacement
        for m in zeroed_methods:
            print(f'moy[{method_list[m]}, band={ll}] was undefined (=0), '
                  f'replaced with the average of the other scan modes: {replacement:.3f}')

    MSNR = 20 * np.log10(moy / std)

    fbar_ref_581 = moy[-1, 0] * 1e3 / 31 / (17**2 / 32.768)
    fbar_ref_726 = moy[-1, 2] * 1e3 / 31 / (17**2 / 32.768)

    if save_array:
        np.save(fig_folder / f'std_{label}', std)
        np.save(fig_folder / f'moy_{label}', moy)
        np.save(fig_folder / f'MSNR_{label}', MSNR)

    # ---- MSNR relative to a reference scan mode (contrast, dB) ----
    # MSNRc[m, ll] = MSNR[m, ll] - MSNR[ref_idx, ll]: how many dB each scan
    # mode gains (>0) or loses (<0) compared to the reference scan mode,
    # per wavelength band. Change ref_method (above, outside the loop) to
    # compare against a different scan mode.
    MSNRc = MSNR - MSNR[ref_idx:ref_idx + 1, :]

    # Band labels, e.g. "579-583nm (31ch)", built the same way as the
    # column titles above
    band_labels = []
    for j in range(len(lambda_central_list)):
        lambda_central = lambda_central_list[j]
        nc = nc_list[j]
        lambda_min = lambda_central - nc + 1
        lambda_max = lambda_central + nc
        lambda_n = lambda_max - lambda_min
        band_labels.append(
            f'{wavelength[0][lambda_min]:.0f}-{wavelength[0][lambda_max]:.0f}nm ({lambda_n}ch)')

    # Nice, dependency-free text table: rows = scan modes, columns = wavelength band
    col_w = 16
    row_label_w = 10
    gap = '  '

    header1 = ' ' * row_label_w + gap.join(f'{lbl:>{col_w}}' for lbl in band_labels)
    sep = '-' * len(header1)

    # print the average
    print('moy=')
    print(moy)
    print('-' * 60)
    # Absolute SNR (linear, = moy/std) for all 5 scan modes (FH2 included
    # this time -- there is no reference to subtract out)
    SNR_abs = moy / std

    print()
    print(f'SNR (linear, = moy / std -- {label})')
    print(sep)
    print(header1)
    print(sep)
    for m in range(len(method_list)):
        row = f'{method_list[m]:<{row_label_w}}'
        row += gap.join(f'{SNR_abs[m, ll]:>{col_w}.2f}' for ll in range(len(lambda_central_list)))
        print(row)
    print(sep)

    # Same table layout, absolute MSNR (dB, = 20*log10(moy/std)) for all 5
    # scan modes (FH2 included -- there is no reference to subtract out)
    print()
    print(f'MSNR (dB, absolute, = 20*log10(moy/std) -- {label})')
    print(sep)
    print(header1)
    print(sep)
    for m in range(len(method_list)):
        row = f'{method_list[m]:<{row_label_w}}'
        row += gap.join(f'{MSNR[m, ll]:>{col_w}.2f}' for ll in range(len(lambda_central_list)))
        print(row)
    print(sep)

    print()
    print(f'MSNRc (dB, relative to {ref_method} -- {label})')
    print(sep)
    print(header1)
    print(sep)
    for m in range(len(method_list)):
        if m == ref_idx:
            continue  # MSNRc is 0 by construction for the reference
        row = f'{method_list[m]:<{row_label_w}}'
        row += gap.join(f'{MSNRc[m, ll]:>{col_w}.2f}' for ll in range(len(lambda_central_list)))
        print(row)
    print(sep)

    results[label] = dict(std=std, moy=moy, snr=snr, psnr=psnr, SNR_abs=SNR_abs,
                           MSNR=MSNR, MSNRc=MSNRc, band_labels=band_labels,
                           header1=header1, sep=sep)

    # ---- compare to SiemensStar (only meaningful for the full, 4096px ROI) ----
    if label == 'cat':
        HF2_mat = [18.58, 10.93, -0.14, -5.89]
        RS_mat = [5.36, 0.11, -20.28, -24.97]
        MH2_mat = [23.38, 16.16, 1.26, -3.93]
        H1_mat = [24.34, 16.59, 5.89, -0.06]
        S1_mat = [24.52, 16.95, 7.25, 0.86]

        # Rows follow the same order as method_list ("FH2", "RS", "MH2", "H1", "S1")
        MSNR_mat = np.array([HF2_mat, RS_mat, MH2_mat, H1_mat, S1_mat])

        # MSNRc_mat[m, ll] = MSNR_mat[m, ll] - MSNR_mat[ref_idx, ll]: same
        # "relative to ref_method" contrast as above, using the same
        # ref_method / ref_idx, but for the SiemensStar reference measurements.
        MSNRc_mat = MSNR_mat - MSNR_mat[ref_idx:ref_idx + 1, :]

        print()
        print(f'MSNRc (dB, relative to {ref_method}) -- SiemensStar')
        print(sep)
        print(header1)
        print(sep)
        for m in range(len(method_list)):
            if m == ref_idx:
                continue  # MSNRc_mat is 0 by construction for the reference
            row = f'{method_list[m]:<{row_label_w}}'
            row += gap.join(f'{MSNRc_mat[m, ll]:>{col_w}.2f}' for ll in range(len(lambda_central_list)))
            print(row)
        print(sep)

#%% Display the 3 ROI figures
plt.show()
