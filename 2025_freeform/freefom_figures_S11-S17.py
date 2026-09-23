# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 2026

The script generates Supplementary Figures S11-S17 of the paper.

Adapted from freefom_add_figure_of_cat_3roi.py for the freeform N-scan
dataset acquired on 2026-09-11 ('cat' object, ROI masks of increasing size
N = 256, 512, 1024, 2048, 4096, 8192 px, plus the full 128x128 frame,
N=16384) -- producing one figure and one set of SNR / MSNR / MSNRc tables
per ROI size N. Each ROI is named 'ROI_N_<N>'. The list of available N
values, and each ROI-adaptive method's actual integration time, are
discovered automatically from the data folder (rather than hardcoded), so
this script keeps working unchanged if more N values are added later. The
FH2 (Walsh) acquisition and its black reference are mask-independent
full-frame measurements, so they are shared by every N. H1/S1/RS
integration times are scaled up as N shrinks so that N_pixel * ti (hence
the acquisition time budget) stays the same for every ROI size. The mask.png
are stored in the Walsh acquisition folder.

@author: ducros
"""
#%% imports and global configuration
import json
import ast
import re
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline
from PIL import Image

from spyrit.misc.disp import imagesc, add_colorbar
from spyrit.misc.sampling import reindex
from spyrit.core.meas import HadamSplit2d, FreeformLinear, FreeformLinearSplit
from spyrit.core.prep import Unsplit
from spyrit.core.torch import walsh_matrix, ifwht
from spyrit.misc.walsh_hadamard import walsh_S_matrix, ifwalsh_S_torch

# One ROI per available N, named 'ROI_N_<N>', from the largest (the full
# 128x128 frame) down to the smallest mask on disk.
only_N = None  # e.g. 512 to process only that ROI, or a list e.g. [512, 1024];
               # None (default) processes every N found on disk.

ti = 2      # base integration time (ms) -- FH2/MH2 are mask-independent
            # full-frame measurements, so they always use this value
h = 128     # image size hxh
norm = 32768 * ti  # time budget in ms, kept constant across every ROI
                   # size so their SNR/MSNR are directly comparable
fig_folder = Path('figures/figures_S11-S17')
fig_folder.mkdir(parents=True, exist_ok=True)
save_tag = True
plot_tag = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_folder = Path(r"data/2026-09-11_freeform_publication")

# FH2 (Walsh) and every black reference are only acquired once, under the
# 'cat' object name, and reused for every ROI size N below.
walsh_title = r'obj_cat_source_white_LED_Walsh_im_128x128_ti_' + str(ti) + 'ms_zoom_x1'
black_obj_slug = 'cat'

method = 'substraction'    # 'classical'#
NR = 2                      # Number of repetitions in the "substraction" method
acq_list = ["FH2", "H1", "MH2", "S1", "RS"]
method_list = ["FH2", "RS", "MH2", "H1", "S1"]  # order used by the metric arrays below

ref_method = 'S1'   # Choose one scan mode into the method_list, to compare MSNR from the other scan mode
ref_idx = method_list.index(ref_method)

#%% Plot options
fs = 9                  # Font size
dpi_fig = 600
cbar_pos = 'bottom'     # colorbar position
plot_tag = False
print_metric = 'PSNR'   # 'SNR'#
print_value = False
dark_plot_tag = False    # => to plot µdark
spl_plot_tag = False     # => to plot spline fit of the dark_expe
sub_plot_tag = False     # => to plot the data_expe / spline substraction

lambda_central_list = [515, 515, 1800, 1800]  # no signal below 15 and above 2038
nc_list = [16, 3, 16, 3]
results = {}


def tex_escape(text):
    """Escape LaTeX-special characters so `text` (e.g. an ROI label such as
    'ROI_N_4096') can be dropped into a plot title/suptitle even when
    text.usetex is enabled below -- raw '_'/'&'/etc. otherwise make LaTeX
    fail (e.g. '&' is a misplaced alignment tab outside a tabular). Only
    use this for display strings (titles); filenames/prints should keep
    the raw label."""
    conv = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
            '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'}
    return ''.join(conv.get(c, c) for c in text)


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


def find_one(data_folder, pattern):
    """Glob `pattern` (relative to data_folder) and return the single
    matching folder's name -- asserts there is exactly one match, so a
    naming-convention mismatch fails loudly instead of silently picking
    the wrong dataset."""
    matches = sorted(data_folder.glob(pattern))
    assert len(matches) == 1, (
        f'expected exactly one match for pattern {pattern!r}, got '
        f'{len(matches)}: {[m.name for m in matches]}')
    return matches[0].name


def extract_ti_ms(folder_name):
    """Pull the integration time (ms) out of a `..._ti_<n>ms_...` dataset
    folder name."""
    m = re.search(r'_ti_(\d+)ms_', folder_name)
    assert m, f'could not find "_ti_<n>ms_" in {folder_name!r}'
    return int(m.group(1))


def discover_N_list(data_folder, walsh_title, h):
    """Every ROI size this campaign has data for: every mask_N_*.png saved
    in the shared Walsh folder, plus N = h*h (the full frame, which has no
    mask file since there is nothing to mask)."""
    mask_files = sorted((data_folder / walsh_title).glob('mask_N_*.png'))
    N_list = [int(re.search(r'mask_N_(\d+)\.png$', f.name).group(1)) for f in mask_files]
    N_list.append(h * h)
    return sorted(set(N_list), reverse=True)



N_list = discover_N_list(data_folder, walsh_title, h)
if only_N is not None:
    only_N_list = [only_N] if isinstance(only_N, int) else list(only_N)
    N_list = [N for N in N_list if N in only_N_list]

# Fixed N -> supplementary-figure-number tag, independent of `only_N`
# filtering above: N=16384 (full frame) is S11, down to N=256 as S17,
# matching the "figure_S11_ROI_N_16384.pdf" ... "figure_S17_ROI_N_256.pdf"
# naming used in the paper.
N_TO_S_TAG = {16384: 'S11', 8192: 'S12', 4096: 'S13', 2048: 'S14',
              1024: 'S15', 512: 'S16', 256: 'S17'}
objects_cfg = [dict(label=f'ROI_N_{N}', N=N,
                     s_tag=N_TO_S_TAG.get(N, f'N{N}'))
               for N in N_list]

#%% Dark measurement / stray light reference
# This is a diagnostic-only, ROI-independent measurement (mu_dark is not
# actually subtracted below -- see the commented-out line), so it is
# loaded once and shared by all ROI iterations.
dark_folder = Path(r"data/2025-09-11_freeform_SNR")
dark_title = [r'obj_black_source_No source_raster_cat_4096_im_128x128_ti_8ms_zoom_x1']

data_dark, _, _ = load_spihim(dark_folder, dark_title)
mu_dark = data_dark[0].mean(axis=0)

if dark_plot_tag:
    plt.figure()
    plt.plot(mu_dark[100:700])
    plt.title('µ dark')
#%% MAIN LOOP over every ROI size N (ROI_N_256 ... ROI_N_16384)
for cfg in objects_cfg:

    label = cfg['label']
    label_tex = tex_escape(label)  # display-safe version of label, for plot
                                    # titles only (text.usetex is turned on
                                    # further below) -- keep using `label`
                                    # (raw) for filenames/prints.
    N = cfg['N']
    s_tag = cfg['s_tag']
    print()
    print('#' * 60)
    print(f'### ROI = {label} (N = {N}, {s_tag}) ###')
    print('#' * 60)

    # ---- reload the mask in full size (128x128) ----
    # Acquisitions save the ready-to-use mask directly as a plain (h x h)
    # binary PNG in the shared Walsh folder -- just read it, no metadata
    # involved. N = h*h (the full frame) has no mask file since there is
    # nothing to mask.
    if N == h * h:
        mask_full = np.ones((h, h), dtype=bool)
    else:
        mask_png_path = data_folder / walsh_title / f'mask_N_{N}.png'
        mask_full = np.array(Image.open(mask_png_path))
        mask_full = mask_full > mask_full.min()

    mask = torch.from_numpy(mask_full).to(device=device)

    ind_array = np.where(mask_full == True)
    ind_array = (torch.from_numpy(ind_array[0]).to(device=device),
                 torch.from_numpy(ind_array[1]).to(device=device))

    N_pixel = len(ind_array[0])

    if plot_tag:
        imagesc(mask.cpu())
        plt.title(f'mask -- {label_tex}')

    # ---- locate this ROI's H1/MH2/S1/RS acquisitions and their actual
    # integration times -- each dataset folder name already encodes its
    # own ti_XXXms (scaled so N_pixel * ti stays constant across N), so
    # read it back from disk instead of recomputing it here ----
    h1_name = find_one(data_folder, f'obj_cat_source_white_LED_hadam1d_*_N_{N}_im_128x128_ti_*ms_zoom_x1')
    mh2_name = find_one(data_folder, f'obj_cat_source_white_LED_hadam2d_*_N_{N}_im_128x128_ti_*ms_zoom_x1')
    s1_name = find_one(data_folder, f'obj_cat_source_white_LED_smatrix_*_N_{N}_im_128x128_ti_*ms_zoom_x1')
    rs_name = find_one(data_folder, f'obj_cat_source_white_LED_raster_*_N_{N}_im_128x128_ti_*ms_zoom_x1')

    ti_H1 = extract_ti_ms(h1_name)
    ti_MH2 = extract_ti_ms(mh2_name)
    ti_S1 = extract_ti_ms(s1_name)
    ti_RS = extract_ti_ms(rs_name)

    # ---- read data ----
    data_title = [walsh_title, h1_name, mh2_name, s1_name, rs_name]

    black_title = [
        r'obj_' + black_obj_slug + r'_source_white_LED_black_4096_im_128x128_ti_' + str(ti) + 'ms_zoom_x1',
        r'obj_' + black_obj_slug + r'_source_white_LED_black_4096_im_128x128_ti_' + str(ti_H1) + 'ms_zoom_x1',
        r'obj_' + black_obj_slug + r'_source_white_LED_black_4096_im_128x128_ti_' + str(ti_MH2) + 'ms_zoom_x1',
        r'obj_' + black_obj_slug + r'_source_white_LED_black_4096_im_128x128_ti_' + str(ti_S1) + 'ms_zoom_x1',
        r'obj_' + black_obj_slug + r'_source_white_LED_black_4096_im_128x128_ti_' + str(ti_RS) + 'ms_zoom_x1']

    # ---- skip this ROI if any required acquisition is not finished yet
    # (its folder exists -- e.g. created by the acquisition GUI on start --
    # but the actual _spectraldata.npz has not been written), rather than
    # crashing the whole run and losing every ROI already processed ----
    missing = [t for t in data_title + black_title
               if not (data_folder / t / (t + '_spectraldata.npz')).exists()]
    if missing:
        print(f'!!! Skipping {label}: acquisition not finished for {missing} !!!')
        continue

    black_exp, _, _ = load_spihim(data_folder, black_title)
    data_exp, wavelength, patterns = load_spihim(data_folder, data_title)

    # ---- denoise black (stray light) ----
    spl_arr = np.zeros((len(black_exp), len(wavelength[0])))

    for j in range(len(black_exp)):

        y = black_exp[j].mean(axis=0) #- mu_dark

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
            plt.title(acq_list[j] + f' - spline fit for black patterns ({label_tex})')

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
                    plt.title(acq_list[j] + f' - first spectrum of each repetiton ({label_tex})')

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
    if N == h * h:
        # Full-frame ROI: there is nothing to crop, so the acquisition
        # metadata's x_mask_coord/y_mask_coord/mask_index are left empty
        # (saved as the single character "]", not a valid Python literal --
        # confirmed on this dataset's raster/hadam1d/hadam2d/smatrix
        # metadata) instead of a real bounding box. Reconstruct the
        # (implicit) scan order directly instead, using the same
        # row-major-within-bounding-box convention as every masked ROI,
        # but with a bounding box spanning the whole frame (x0=y0=0,
        # w_len=h).
        x0, y0, w_len = 0, 0, h
        mask_index = np.arange(h * h, dtype=int)
    else:
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

    # --- ROI used for the SNR measurement ---
    # Each ROI_N_* mask is a single connected region (the full frame for
    # the largest N) -- roi_mask is simply that mask.
    roi_mask = mask

    f, ax = plt.subplots(5, len(lambda_central_list),
                          figsize=(len(lambda_central_list) * 2, 10))

    # axes: [method, wavelength band]
    std = np.empty([5, len(lambda_central_list)])
    moy = np.empty([5, len(lambda_central_list)])
    snr = np.empty([5, len(lambda_central_list)])
    maxi = np.empty([5, len(lambda_central_list)])
    psnr = np.empty([5, len(lambda_central_list)])

    for ll in range(len(lambda_central_list)):
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
            plt.title(f'H2F, image 0 ({label_tex})')

            plt.figure()
            plt.imshow(x_H2dF_2.cpu())
            plt.colorbar()
            plt.title(f'H2F, image 1 ({label_tex})')

            x_H2dF_sub = x_H2dF - x_H2dF_2
            plt.figure()
            plt.imshow(x_H2dF_sub.cpu())
            plt.colorbar()
            plt.title(f'H2F, diff ({label_tex})')

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
            plt.title(f'RS ({label_tex})')

            plt.figure()
            plt.imshow(x_I1d_2.cpu())
            plt.colorbar()
            plt.title(f'RS image 1 ({label_tex})')

            x_I1d_sub = x_I1d - x_I1d_2
            plt.figure()
            plt.imshow(x_I1d_sub.cpu())
            plt.colorbar()
            plt.title(f'RS image sub ({label_tex})')

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
            plt.title(f'H2M, image 0 ({label_tex})')

            plt.figure()
            plt.imshow(x_H2dM_2.cpu())
            plt.colorbar()
            plt.title(f'H2dM, image 1 ({label_tex})')

            x_H2dM_sub = x_H2dM - x_H2dM_2
            plt.figure()
            plt.imshow(x_H2dM_sub.cpu())
            plt.colorbar()
            plt.title(f'H2dM, diff ({label_tex})')

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
            plt.title(f'H1, image 0 ({label_tex})')

            plt.figure()
            plt.imshow(x_H1d_2.cpu())
            plt.colorbar()
            plt.title(f'H1, image 1 ({label_tex})')

            x_H1d_sub = x_H1d - x_H1d_2
            plt.figure()
            plt.imshow(x_H1d_sub.cpu())
            plt.colorbar()
            plt.title(f'H1, diff ({label_tex})')

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
            plt.title(f'SM ({label_tex})')

            plt.figure()
            plt.imshow(x_S1d_2.cpu())
            plt.colorbar()
            plt.title(f'SM, image 1 ({label_tex})')

            x_S1d_sub = x_S1d - x_S1d_2
            plt.figure()
            plt.imshow(x_S1d_sub.cpu())
            plt.colorbar()
            plt.title(f'SM, diff ({label_tex})')

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

    # NB: no tex_escape here -- this script never turns text.usetex on, so
    # an escaped underscore ('\_') would render as a literal backslash
    # instead of being interpreted by LaTeX.
    f.suptitle(f'Freeform region: N = {N}')
    plt.tight_layout()

    if save_tag:
        # Save the actual grid figure object `f`, not whatever plt considers
        # the "current" figure -- otherwise a stale/unrelated figure could
        # silently end up saved instead, producing a near-empty file.
        fil_name = f'figure_{s_tag}_{label}.pdf'
        f.savefig(fig_folder / fil_name, bbox_inches='tight', dpi=dpi_fig)

    # ---- MSNR ----
    save_array = True

    # Two situations mark a raw moy[m, ll] as unreliable. Both are replaced
    # the same way -- with the average moy of the other, valid scan modes
    # at that band -- *before* MSNR is computed, so the imputed moy still
    # gets combined with that method's own (valid) std:
    #  - moy == 0: a defensive clamp applied when the raw reconstructed
    #    mean came out negative (see "!!!!! Warning, ... mean < 0" above).
    #  - moy far above what the other methods agree on: e.g. RS's moy can
    #    be contaminated by a stray-light-fit residual that the shared
    #    "black" reference (one acquisition, reused via a spline fit by
    #    every scan mode) does not fully capture for RS's own scan
    #    geometry -- confirmed at N=512, 726nm, where RS's raw moy came
    #    out ~8x higher than FH2/MH2/H1/S1, which agree closely with each
    #    other there.
    OUTLIER_FACTOR = 3.0  # flag moy[m, ll] if it exceeds OUTLIER_FACTOR
                           # times the median moy of the OTHER, non-zeroed
                           # methods at that band

    zeroed = (moy == 0)
    outlier = np.zeros_like(zeroed)
    for ll in range(moy.shape[1]):
        for m in range(moy.shape[0]):
            if zeroed[m, ll]:
                continue
            others = [mm for mm in range(moy.shape[0]) if mm != m and not zeroed[mm, ll]]
            if not others:
                continue
            median_others = np.median(moy[others, ll])
            if median_others > 0 and moy[m, ll] > OUTLIER_FACTOR * median_others:
                outlier[m, ll] = True

    invalid = zeroed | outlier
    for ll in range(moy.shape[1]):
        invalid_methods = np.where(invalid[:, ll])[0]
        if len(invalid_methods) == 0:
            continue
        valid_methods = np.where(~invalid[:, ll])[0]
        if len(valid_methods) == 0:
            print(f'Warning: all methods are unreliable at band={ll} -- cannot impute moy')
            continue
        replacement = moy[valid_methods, ll].mean()
        moy[invalid_methods, ll] = replacement
        for m in invalid_methods:
            reason = 'undefined (=0)' if zeroed[m, ll] else f'> {OUTLIER_FACTOR:g}x the other methods\' median'
            print(f'moy[{method_list[m]}, band={ll}] was {reason}, '
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

    # # print the average
    # print('moy=')
    # print(moy)
    # print('-' * 60)
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

    # ---- compare to SiemensStar (SiemensStar reference values below were
    # measured on an N=4096 ROI, so only compare at the matching ROI size) ----
    if N == 4096:
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

#%% Display the ROI figures
plt.show()
