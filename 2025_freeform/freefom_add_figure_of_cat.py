# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 16:48:48 2025

The script generates Fig. 6 of the paper

@author: ducros
"""
# -*- coding: utf-8 -*-

#%%
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline
import os
os.chdir("d:/hspc/scripts")

h = 128  # image size hxh
norm = 32768 # time budget in ms
fig_folder = Path('../result/freeform/figures_cat')
save_tag = True
plot_tag = True#False 
#%% reload the mask in full size (128x128)
# Acquisitions save the ready-to-use mask directly as a plain (h x h)
# binary PNG named 'mask.png' in their own folder (mode 'L', 128x128,
# only 2 grey levels) -- just read it, no metadata involved.
from PIL import Image
from spyrit.misc.disp import imagesc

ti = 2.5 #2

if ti == 2:
    mask_data_folder = Path(r"../data/2026-08-24_freeform_publication")
elif ti == 2.5:
    mask_data_folder = Path(r"../data/2026-08-27_freeform_publication")

mask_source_title = r'obj_cat_source_white_LED_Walsh_im_128x128_ti_' + str(ti) + 'ms_zoom_x1'    
mask_name = mask_source_title

mask_png_path = mask_data_folder / mask_source_title / 'mask.png'

mask_full = np.array(Image.open(mask_png_path))
mask_full = mask_full > mask_full.min()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mask = torch.from_numpy(mask_full).to(device=device)

ind_array = np.where(mask_full == True)
ind_array = (torch.from_numpy(ind_array[0]).to(device=device),
             torch.from_numpy(ind_array[1]).to(device=device))

N_pixel = len(ind_array[0])

# shape and plot
imagesc(mask.cpu())

#%% read data
# Load experimental data
# ====================================================================
import json
import ast

data_folder = mask_data_folder

method = 'substraction'     # 'classical'#
NR = 2                   # Number of repetition in the case of "substraction" method
acq_list = ["FH2", "H1", "MH2", "S1", "RS"]

# acquisitions
data_title = [
r'obj_cat_source_white_LED_Walsh_im_128x128_ti_' + str(ti) + 'ms_zoom_x1',
r'obj_cat_source_white_LED_hadam1d_8192_im_128x128_ti_' + str(int(ti*4)) + 'ms_zoom_x1',
r'obj_cat_source_white_LED_hadam2d_32768_im_128x128_ti_' + str(ti) + 'ms_zoom_x1',
r'obj_cat_source_white_LED_smatrix_4095_im_128x128_ti_' + str(int(ti*8)) + 'ms_zoom_x1',
r'obj_cat_source_white_LED_Raster_im_128x128_ti_' + str(int(ti*8)) + 'ms_zoom_x1']

# black patterns
black_title = [
r'obj_cat_source_white_LED_black_32768_im_128x128_ti_' + str(ti) + 'ms_zoom_x1',
r'obj_cat_source_white_LED_black_8192_im_128x128_ti_' + str(int(ti*4)) + 'ms_zoom_x1',
r'obj_cat_source_white_LED_black_32768_im_128x128_ti_' + str(ti) + 'ms_zoom_x1',
r'obj_cat_source_white_LED_black_4096_im_128x128_ti_' + str(int(ti*8)) + 'ms_zoom_x1',
r'obj_cat_source_white_LED_black_4096_im_128x128_ti_' + str(int(ti*8)) + 'ms_zoom_x1'
]
        
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

black_exp, _, _ = load_spihim(data_folder, black_title)
data_exp, wavelength, patterns = load_spihim(data_folder, data_title)

#%% Dark measurement / stray light
plot_tag = True

dark_folder = Path(r"../data/2025-09-11_freeform_SNR")
dark_title = [r'obj_black_source_No source_raster_cat_4096_im_128x128_ti_8ms_zoom_x1']

# Load dark measurements
data_dark, _, _ = load_spihim(dark_folder, dark_title)
mu_dark = data_dark[0].mean(axis=0)

# Plot mean
if plot_tag:
    plt.figure()
    plt.plot(mu_dark[100:700])
    plt.title('µ dark')    

# Denoise
spl_arr = np.zeros((len(black_exp), len(wavelength[0])))
lc = [515, 1800] # central wavelength (in pixel) chosen in the main loop, varaible: lambda_central_list
lambda_central_list = [515, 515, 1800, 1800] # no signal below 15 and above 2038
nc_list = [16, 3, 16, 3]
sigma_m = np.zeros((len(black_exp), len(lc)))
sigma2_m = np.zeros((len(black_exp), len(lambda_central_list), 16))
for j in range(len(black_exp)):
    y = black_exp[j].mean(axis=0)# - mu_dark
    if method == 'substraction':
        spl = make_smoothing_spline(wavelength[0], y, lam=1e3)
    else: #method == 'classical':
        if j <= 2:
            spl = make_smoothing_spline(wavelength[0], black_exp[j][1,:], lam=100)
        else:
            RS_mean = black_exp[4].mean(axis=1)
            ind_min = np.argmin(RS_mean)
            spl = make_smoothing_spline(wavelength[0], black_exp[4][ind_min,:], lam=100)
        
    spl_np = spl(wavelength[0])
    spl_arr[j,:] = spl_np
    
    if plot_tag:
        plt.figure()
        plt.plot(wavelength[0], y, marker = "o", color = "blue")
        plt.plot(wavelength[0], spl_arr[j,:], color = "red")
        plt.title(acq_list[j] + ' - spline fit for black patterns')

    # measure of sigma dark
    yi = black_exp[j]
    
    for i, lci in enumerate(lc):
        yi_lc = yi[:, lci]
        sigma_m[j, i] = np.std(yi_lc)
    
    for i, lci in enumerate(lambda_central_list):
        for k in range(nc_list[i]):
            yi_lc = yi[:, lci + k - 8]
            sigma2_m[j, i, k] = np.std(yi_lc)
            
    # sigma22_m = sigma2_m.mean(axis=2)
    
    sigma2_m[sigma2_m == 0] = np.nan
    means = np.nanmean(sigma2_m, axis=2)

#%% substract stray light

plot_tag = True
  
acqui_size = []
for j in range(len(data_exp)):

    # substract parasite light
    acqui_size.append(int(data_exp[j].shape[0] / NR))
    for nM in range(acqui_size[j]*NR):

        
        data_exp[j][nM, :] = data_exp[j][nM, :] - spl_arr[j,:]
        
        if plot_tag:
            if nM == 0 or nM == acqui_size[j]:
                plt.figure(j)
                if nM == 0:
                    Color = 'blue'
                elif nM == acqui_size[j]:
                    Color = 'red'
                plt.figure()
                plt.plot(wavelength[0], data_exp[j][nM, :], color=Color)
                plt.title(acq_list[j] + ' - first spectrum of each repetiton')

#%% delete offset | Why ? Keep ?

# for k in range(data_exp[4].shape[0]):    
#     y=data_exp[4][k,:]
    
#     ym=y[1950:].mean()
#     data_exp[4][k,:] = data_exp[4][k,:] - ym
    
#     print(ym)

# %% Reorder measurements for full 2D Hadamard
from spyrit.misc.sampling import reindex

for nR in range(NR):
    data_exp[0][acqui_size[0]*nR:acqui_size[0]*(nR+1), :] = reindex(
            data_exp[0][acqui_size[0]*nR:acqui_size[0]*(nR+1), :], 
            np.array(patterns[0]), 
            axis = "rows",
            inverse_permutation = True
            )
    
#%% MAIN LOOP
from spyrit.misc.disp import add_colorbar

# --- Rebuild ind_array from the acquisition metadata (native scan order) ---
# `ind_array`, as built in the "reload the mask" cell, comes from
# np.where(mask==True) on mask.png -- that recovers the right *set* of
# pixels, but not their acquisition order: mask.png is saved 180deg-
# rotated relative to the DMD/scan convention used to build the H1/S1/RS
# measurement matrices (acquisition_parameters.mask_index in the
# metadata), so np.where ends up traversing that set in *reverse* scan
# order. This doesn't matter for the 2D Hadamard methods (FH2/MH2), which
# only use mask as an order-independent boolean selection, but it breaks
# the arbitrary-shape 1D methods (H1/S1/RS): their k-th measurement must
# land on the k-th pixel of the original scan order.
# Fix: read the scan order from metadata (mask_index/x_mask_coord/
# y_mask_coord) and flip it 180deg (row -> h-1-row, col -> h-1-col) to
# land in mask.png's frame -- this preserves the scan order while
# matching mask's pixel positions exactly (checked below).

# Plot options  
fs = 9                 # Font size
dpi_fig = 600
cbar_pos = 'bottom'     #'right'# colorbar position
plot_tag = False
print_metric = 'PSNR' # 'SNR'#
print_value = True


roi_meta_title = data_title[4]  # any ROI-adaptive acquisition works; they all share the same ROI
with open(data_folder / roi_meta_title / (roi_meta_title + '_metadata.json'), 'r') as file:
    roi_acquisition_parameters = json.load(file)[4]

x_mask_coord = ast.literal_eval(roi_acquisition_parameters['x_mask_coord'])
y_mask_coord = ast.literal_eval(roi_acquisition_parameters['y_mask_coord'])
mask_index   = np.array(ast.literal_eval(roi_acquisition_parameters['mask_index']), dtype=int)
x0, y0 = int(x_mask_coord[0]), int(y_mask_coord[0])
w_len  = int(x_mask_coord[1] - x0)

row = h - 1 - (y0 + mask_index // w_len)
col = h - 1 - (x0 + mask_index % w_len)
ind_array = (torch.from_numpy(row).to(device=device), torch.from_numpy(col).to(device=device))
N_pixel = len(ind_array[0])

assert set(zip(row.tolist(), col.tolist())) == set(zip(*np.where(mask.cpu().numpy()))), \
    'metadata-based ind_array does not match mask.png -- check x_mask_coord/y_mask_coord/mask_index'

# --- Split the freeform mask into its two ROIs, for the SNR measurement ---
# The mask is made of two disjoint blobs: roi1, on the left, hugging the
# cat's head outline, and roi2, on the right, roughly rectangular.
from scipy import ndimage

labeled_mask, n_components = ndimage.label(mask.cpu().numpy(), structure=np.ones((3, 3)))
assert n_components == 2, f'expected 2 disjoint ROIs in the mask, found {n_components}'

centroids_x = ndimage.center_of_mass(mask.cpu().numpy(), labeled_mask, [1, 2])
label_roi1, label_roi2 = sorted([1, 2], key=lambda lbl: centroids_x[lbl - 1][1])  # left first

roi1_mask = torch.from_numpy(labeled_mask == label_roi1).to(device=device)
roi2_mask = torch.from_numpy(labeled_mask == label_roi2).to(device=device)
roi_masks = [roi1_mask, roi2_mask]
roi_names = ['roi1', 'roi2']


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

# lambda_central_list = [512, 512, 1900, 1900] # no signal below 15 and above 2038
# nc_list = [1, 15, 1, 15]

lambda_central_list = [515, 515, 1800, 1800] # no signal below 15 and above 2038
nc_list = [16, 3, 16, 3]
# lambda_central_list = [1800] # no signal below 15 and above 2038
# nc_list = [16]

f, ax = plt.subplots(5,len(lambda_central_list),
                     figsize=(len(lambda_central_list)*2,10))

# axes: [method, roi1/roi2, wavelength band]
std = np.empty([5, 2, len(lambda_central_list)])
moy = np.empty([5, 2, len(lambda_central_list)])
snr = np.empty([5, 2, len(lambda_central_list)])
maxi = np.empty([5, 2, len(lambda_central_list)])
psnr = np.empty([5, 2, len(lambda_central_list)])

for ll in range(len(lambda_central_list)):
    print("================================= Lambda = " + str(lambda_central_list[ll]) + " nm / band = " + str(nc_list[ll]) + " =================================")
    #% Spectral binning
    lambda_central = lambda_central_list[ll]
    nc = nc_list[ll]
    lambda_min = lambda_central - nc + 1      
    lambda_max = lambda_central + nc
    lambda_n = lambda_max - lambda_min 
    
    data_bin = [[data_exp[ii][acqui_size[ii]*nR:acqui_size[ii]*(nR+1),lambda_min:lambda_max].sum(axis=1)
                for ii in range(len(data_title))]
                for nR in range(NR)]
    
    #% Convert to torch tensors
    data_bin = [[torch.from_numpy(data_bin[nR][ii]).to(device=device,dtype=torch.float32)
                for ii in range(len(data_title))]
                for nR in range(NR)]

    #--------------------------------------------------------------------------  
    #% 2D Hadamard full
    #--------------------------------------------------------------------------
    print('== Hadamard 2D full ==')
    
    # Select dataset
    indx_dataset = 0
    indx_graph = 0
    
    y = data_bin[0][indx_dataset]
    y2 = data_bin[1][indx_dataset]

    # integration time scaling
    y = y * (h*h*2) / norm
    y2 = y2 * (h*h*2) / norm

    print('max of meas:', y.max())
    print('min of meas:', y.min())
    
    from spyrit.core.meas import HadamSplit2d
    from spyrit.core.prep import Unsplit
    
    meas_op = HadamSplit2d(h, device=device)
    prep = Unsplit()
    prep = prep.to(device=device)
    
    # NB: fast_pinv's output frame is 180deg-flipped relative to mask
    # (mask.png)/ind_array's frame -- rotate it back so ROI selection
    # below (roi = x_H2dF[mask]) picks the right pixels.
    x_H2dF   = torch.rot90(meas_op.fast_pinv(prep(y)), 2, dims=(0, 1))
    x_H2dF_2 = torch.rot90(meas_op.fast_pinv(prep(y2)), 2, dims=(0, 1))
    
    # Plot
    if plot_tag:
        plt.figure()
        plt.imshow(x_H2dF.cpu())
        plt.colorbar()
        plt.title('H2F, image 0')

        plt.figure()
        plt.imshow(x_H2dF_2.cpu())
        plt.colorbar()
        plt.title('H2F, image 1')     

        x_H2dF_sub = x_H2dF - x_H2dF_2
        plt.figure()
        plt.imshow(x_H2dF_sub.cpu())
        plt.colorbar()
        plt.title('H2F, diff')
            
    # Compute metrics in each of the two ROIs
    for r, roi_mask in enumerate(roi_masks):
        moy[indx_graph, r, ll], std[indx_graph, r, ll], maxi[indx_graph, r, ll] = \
            compute_roi_snr(x_H2dF, x_H2dF_2, roi_mask)

        snr[indx_graph, r, ll] = moy[indx_graph, r, ll] / std[indx_graph, r, ll]

        if print_value:
            print(f'{roi_names[r]}: std = {std[indx_graph, r, ll]}')
            print(f'{roi_names[r]}: moy = {moy[indx_graph, r, ll]}')
            print(f'{roi_names[r]}: snr = {snr[indx_graph, r, ll]}')

    #--------------------------------------------------------------------------
    #% RASTER SCAN
    #--------------------------------------------------------------------------
    print('== Raster Scan ==')

    indx_dataset = 4
    indx_graph = 1
    
    y = data_bin[0][indx_dataset]
    y2 = data_bin[1][indx_dataset]
    
    # integration time scaling
    y = y * N_pixel / norm
    y2 = y2 * N_pixel / norm
    
    print('max of meas:', y.max())
    print('min of meas:', y.min())
    
    # Hadamard
    from spyrit.core.meas import FreeformLinear
    H = torch.eye(N_pixel)
    meas_1d = FreeformLinear(H,
                    meas_shape = (h,h), 
                    index_mask = torch.stack(ind_array),
                    device = device
                    )
    
    # Direct reconstruction
    x_rec_2 = y
    x_I1d = torch.zeros_like(x_H2dF)
    x_I1d[ind_array[0],ind_array[1]] = x_rec_2
    x_I1d[~mask] = x_rec_2.min()    # Set out-of-ROI pixels to minimum value
    
    # Direct reconstruction
    x_rec_2 = y2
    x_I1d_2 = torch.zeros_like(x_H2dF)
    x_I1d_2[ind_array[0],ind_array[1]] = x_rec_2
    x_I1d_2[~mask] = x_rec_2.min()  # Set out-of-ROI pixels to minimum value
    
    # Plot
    if plot_tag:
        plt.figure()
        plt.imshow(x_I1d.cpu())
        plt.colorbar()
        plt.title('RS')

        plt.figure()
        plt.imshow(x_I1d_2.cpu())
        plt.colorbar()
        plt.title('RS image 1')

        x_I1d_sub = x_I1d - x_I1d_2
        plt.figure()
        plt.imshow(x_I1d_sub.cpu())
        plt.colorbar()
        plt.title('RS image sub')
        
    # Compute metrics in each of the two ROIs
    for r, roi_mask in enumerate(roi_masks):
        moy[indx_graph, r, ll], std[indx_graph, r, ll], maxi[indx_graph, r, ll] = \
            compute_roi_snr(x_I1d, x_I1d_2, roi_mask)

        if moy[indx_graph, r, ll] < 0:
            moy[indx_graph, r, ll] = 0
            print(f'!!!!! Warning, {roi_names[r]} mean < 0 !!!!!!!!!!!!!!!')

        snr[indx_graph, r, ll] = moy[indx_graph, r, ll] / std[indx_graph, r, ll]
        psnr[indx_graph, r, ll] = 20*np.log10(maxi[indx_graph, r, ll] / std[indx_graph, r, ll])

        if print_value:
            print(f'{roi_names[r]}: std = {std[indx_graph, r, ll]}')
            print(f'{roi_names[r]}: moy = {moy[indx_graph, r, ll]}')
            print(f'{roi_names[r]}: snr = {snr[indx_graph, r, ll]}')

    #--------------------------------------------------------------------------
    #% Masked 2D Hadamard
    #--------------------------------------------------------------------------
    print('== Hadamard 2D masked ==')
    
    # Select dataset
    indx_dataset = 2
    indx_graph = 2
        
    y = data_bin[0][indx_dataset]
    y2 = data_bin[1][indx_dataset]
    
    # integration time scaling
    y = y * (h*h*2) / norm
    y2 = y2 * (h*h*2) / norm
    
    print('max of meas:', y.max())
    print('min of meas:', y.min())
    
    #  Init operators
    meas_op = HadamSplit2d(h, device=device)
    prep = Unsplit().to(device=device)
    
    # Pseudo inverse reconstruction
    # NB: same 180deg flip as FH2 -- fast_pinv's output frame does not
    # match mask/ind_array's frame otherwise (see FH2 section above).
    x_H2dM = torch.rot90(meas_op.fast_pinv(prep(y)), 2, dims=(0, 1))
    x_H2dM[~mask] = x_H2dM[mask].min()  # Set out-of-ROI pixels to minimum value

    x_H2dM_2 = torch.rot90(meas_op.fast_pinv(prep(y2)), 2, dims=(0, 1))
    x_H2dM_2[~mask] = x_H2dM_2.min()
    
    # Plot
    if plot_tag:
        plt.figure()
        plt.imshow(x_H2dM.cpu())
        plt.colorbar()
        plt.title('H2M, image 0')

        plt.figure()
        plt.imshow(x_H2dM_2.cpu())
        plt.colorbar()
        plt.title('H2dM, image 1')

        x_H2dM_sub = x_H2dM - x_H2dM_2
        plt.figure()
        plt.imshow(x_H2dM_sub.cpu())
        plt.colorbar()
        plt.title('H2dM, diff')

    # Compute metrics in each of the two ROIs
    for r, roi_mask in enumerate(roi_masks):
        moy[indx_graph, r, ll], std[indx_graph, r, ll], maxi[indx_graph, r, ll] = \
            compute_roi_snr(x_H2dM, x_H2dM_2, roi_mask)

        snr[indx_graph, r, ll] = moy[indx_graph, r, ll] / std[indx_graph, r, ll]

        if print_value:
            print(f'{roi_names[r]}: std = {std[indx_graph, r, ll]}')
            print(f'{roi_names[r]}: moy = {moy[indx_graph, r, ll]}')
            print(f'{roi_names[r]}: snr = {snr[indx_graph, r, ll]}')

    #--------------------------------------------------------------------------
    #% Arbitrary shape -- Hadamard matrix 1D
    #--------------------------------------------------------------------------
    print('== Hadamard 1D ==')
    
    # Select dataset
    indx_dataset = 1
    indx_graph = 3
    
    y = data_bin[0][indx_dataset]
    y2 = data_bin[1][indx_dataset]
    
    # integration time scaling
    y = y * (N_pixel*2) / norm
    y2 = y2 * (N_pixel*2) / norm
    
    print('max of meas:', y.max())
    print('min of meas:', y.min())
    
    # Init operators
    from spyrit.core.torch import walsh_matrix
    from spyrit.core.meas import FreeformLinearSplit
    
    H = walsh_matrix(N_pixel)
    meas_1d = FreeformLinearSplit(H, 
                        meas_shape = (h,h), 
                        index_mask = torch.stack(ind_array),
                        device = device)
    # Reconstruction
    from spyrit.core.torch import ifwht
    prep = Unsplit()
    
    x_rec_2 = ifwht(prep(y)) 
    x_H1d = torch.zeros_like(x_H2dF)
    x_H1d[ind_array[0],ind_array[1]] = x_rec_2
    x_H1d[~mask] = x_rec_2.min()      # Set out-of-ROI pixels to minimum value
    
    x_rec_2 = ifwht(prep(y2))         
    x_H1d_2 = torch.zeros_like(x_H2dF)
    x_H1d_2[ind_array[0],ind_array[1]] = x_rec_2
    x_H1d_2[~mask] = x_rec_2.min()    # Set out-of-ROI pixels to minimum value  
    
    # Plot
    if plot_tag:
        plt.figure()
        plt.imshow(x_H1d.cpu())
        plt.colorbar()
        plt.title('H1, image 0')
        
        plt.figure()
        plt.imshow(x_H1d_2.cpu())
        plt.colorbar()
        plt.title('H1, image 1')

        x_H1d_sub = x_H1d - x_H1d_2
        plt.figure()
        plt.imshow(x_H1d_sub.cpu())
        plt.colorbar()
        plt.title('H1, diff')
    
    # Compute metrics in each of the two ROIs
    for r, roi_mask in enumerate(roi_masks):
        moy[indx_graph, r, ll], std[indx_graph, r, ll], maxi[indx_graph, r, ll] = \
            compute_roi_snr(x_H1d, x_H1d_2, roi_mask)

        snr[indx_graph, r, ll] = moy[indx_graph, r, ll] / std[indx_graph, r, ll]

        if print_value:
            print(f'{roi_names[r]}: std = {std[indx_graph, r, ll]}')
            print(f'{roi_names[r]}: moy = {moy[indx_graph, r, ll]}')
            print(f'{roi_names[r]}: snr = {snr[indx_graph, r, ll]}')

    #--------------------------------------------------------------------------
    #% Arbitrary shape -- S matrix
    #--------------------------------------------------------------------------
    print('== S-matrix 1D ==')
    
    # Select dataset
    # y_dark = lambda_n*mu_dark
    indx_dataset = 3
    indx_graph = 4
    
    y  = data_bin[0][indx_dataset]
    y2 = data_bin[1][indx_dataset]
    
    # Integration time scaling
    # NB: we use N_pixel here, not N_pixel-1, in accordance with the experiment
    y  = y * N_pixel / norm
    y2 = y2 * N_pixel / norm
    
    print('max of meas:', y.max())
    print('min of meas:', y.min())
    
    # Init operators
    from spyrit.misc.walsh_hadamard import walsh_S_matrix, ifwalsh_S_torch
    
    H = torch.from_numpy(walsh_S_matrix(N_pixel-1))
    
    ind_array_0_S = ind_array[0][:-1]
    ind_array_1_S = ind_array[1][:-1]
    
    meas_1d = FreeformLinear(H, 
                        meas_shape = (h,h), 
                        index_mask = torch.stack((ind_array_0_S, ind_array_1_S)),
                        device = device) 
    # NB: pixel index 0 (i.e. ind_array_0_S[0], ind_array_1_S[0]) is a
    # structural artifact of the cyclic S-matrix construction, not a real
    # measurement: ifwalsh_S_torch effectively reconstructs it as a
    # residual/"missing-mode" bin, so it swings far outside the range of
    # every other pixel regardless of wavelength band (checked: it's the
    # single highest pixel -- ~2x brighter than the next one -- at 581nm,
    # and the single lowest -- as low as -20 vs. ~-3.5 for every other
    # pixel -- at 726nm, for both repetitions). Left as is, that one bad
    # pixel also contaminates the whole out-of-ROI background below, since
    # it's set to x_rec_2.min(). Exclude it from the background-fill value,
    # then zero it out for display.

    # Reconstruction
    x_rec_2 = ifwalsh_S_torch(y)
    x_bg = x_rec_2[1:].min()  # background value, excluding the artifact pixel
    x_rec_2[0] = 0
    x_S1d = torch.zeros_like(x_H2dF)
    x_S1d[ind_array_0_S,ind_array_1_S] = x_rec_2
    x_S1d[~mask] = x_bg    # Set out-of-ROI pixels to minimum value

    # Reconstruction
    x_rec_2 = ifwalsh_S_torch(y2)
    x_bg = x_rec_2[1:].min()  # background value, excluding the artifact pixel
    x_rec_2[0] = 0
    x_S1d_2 = torch.zeros_like(x_H2dF)
    x_S1d_2[ind_array_0_S,ind_array_1_S] = x_rec_2
    x_S1d_2[~mask] = x_bg   # Set out-of-ROI pixels to minimum value
    
    # Plot
    if plot_tag:
        plt.figure()
        plt.imshow(x_S1d.cpu())
        plt.colorbar()
        plt.title('SM')

        plt.figure()
        plt.imshow(x_S1d_2.cpu())
        plt.colorbar()
        plt.title('SM, image 1')

        x_S1d_sub = x_S1d - x_S1d_2
        plt.figure()
        plt.imshow(x_S1d_sub.cpu())
        plt.colorbar()
        plt.title('SM, diff')

    # Compute metrics in each of the two ROIs
    for r, roi_mask in enumerate(roi_masks):
        moy[indx_graph, r, ll], std[indx_graph, r, ll], maxi[indx_graph, r, ll] = \
            compute_roi_snr(x_S1d, x_S1d_2, roi_mask)

        snr[indx_graph, r, ll] = moy[indx_graph, r, ll] / std[indx_graph, r, ll]

        if print_value:
            print(f'{roi_names[r]}: std = {std[indx_graph, r, ll]}')
            print(f'{roi_names[r]}: moy = {moy[indx_graph, r, ll]}')
            print(f'{roi_names[r]}: snr = {snr[indx_graph, r, ll]}')

    #--------------------------------------------------------------------------
    #% Plot all images on same figure
    #--------------------------------------------------------------------------
    im = ax[0,ll].imshow(x_H2dF.cpu(), cmap="gray")
    cbar = add_colorbar(im, cbar_pos)
    cbar.ax.tick_params(labelsize=fs-3)
    
    im = ax[1,ll].imshow(x_I1d.cpu(), cmap="gray")
    cbar = add_colorbar(im, cbar_pos)
    cbar.ax.tick_params(labelsize=fs-3)
    
    im = ax[2,ll].imshow(x_H2dM.cpu(), cmap="gray")
    cbar = add_colorbar(im, cbar_pos)
    cbar.ax.tick_params(labelsize=fs-3)
    
    im = ax[3,ll].imshow(x_H1d.cpu(), cmap="gray")
    cbar = add_colorbar(im, cbar_pos)
    cbar.ax.tick_params(labelsize=fs-3)
    
    im = ax[4,ll].imshow(x_S1d.cpu(), cmap="gray")
    cbar = add_colorbar(im, cbar_pos)
    cbar.ax.tick_params(labelsize=fs-2)
    
    for j in range(5):
        ax[j,ll].set_xticks([])
        ax[j,ll].set_yticks([]) 
        # if measure == 'noise':
        #     ax[j,ll].set_title('std = ' + str(round(std[j,ll]*100)/100), fontsize = fs-3, x = 0.5, y = 0.95)
        # elif measure == 'signal':
        #     if print_metric == 'SNR': 
        #         ax[j,ll].set_title('SNR = ' + str(round(snr[j,ll]*100)/100), fontsize = fs-3, x = 0.5, y = 0.95)
        #     elif print_metric == 'PSNR':
        #         ax[j,ll].set_title('PSNR = ' + str(round(psnr[j,ll]*100)/100) + ' dB', fontsize = fs-3, x = 0.5, y = 0.95)

#%% End of main loop
save_tag = True
# Row labels  
method_list = ["FH2", "RS", "MH2", "H1", "S1"]

for j in range(len(method_list)):
    ax[j,0].set_ylabel(method_list[j], fontsize=fs)
    
# Column labels  
for j in range(4):
    
    lambda_central = lambda_central_list[j]
    nc = nc_list[j]
    lambda_min = lambda_central - nc + 1      
    lambda_max = lambda_central + nc
    lambda_n = lambda_max - lambda_min 
    
    ax[0,j].set_title(
        f'{lambda_n} channel' + 
            ('s\n' if lambda_n>1 else ' \n') +
            f'in [{wavelength[0][lambda_min]:0.0f}, {wavelength[0][lambda_max]:0.0f}) nm',
        fontsize=fs)

plt.tight_layout()

if save_tag:
    # Save the actual grid figure object `f`, not whatever plt considers
    # the "current" figure -- otherwise a stale/unrelated figure (e.g.
    # from re-running an earlier plot_tag=True cell) can silently end up
    # saved instead, producing a near-empty file.
    f.savefig(fig_folder / 'figure_cat_ti_' + str(ti) + 'ms.pdf', bbox_inches='tight', dpi=dpi_fig)
#%%
save_array = True

# moy == 0 (defensive clamp applied when the raw mean came out negative,
# e.g. RS at low signal -- see "!!!!! Warning, ... mean < 0" in MAIN LOOP)
# is not a real signal estimate. Replace it with the average moy of the
# other scan modes at the same ROI and wavelength band (i.e. average over
# the method axis, excluding the zeroed-out method(s))  *before* computing
# MSNR, so the imputed moy is combined with that method's own (valid,
# non-zero) std.
zeroed = (moy == 0)
for r in range(moy.shape[1]):
    for ll in range(moy.shape[2]):
        zeroed_methods = np.where(zeroed[:, r, ll])[0]
        if len(zeroed_methods) == 0:
            continue
        valid_methods = np.where(~zeroed[:, r, ll])[0]
        if len(valid_methods) == 0:
            print(f'Warning: all methods are zero at roi={roi_names[r]}, band={ll} -- cannot impute moy')
            continue
        replacement = moy[valid_methods, r, ll].mean()
        moy[zeroed_methods, r, ll] = replacement
        for m in zeroed_methods:
            print(f'moy[{method_list[m]}, {roi_names[r]}, band={ll}] was undefined (=0), '
                  f'replaced with the average of the other scan modes: {replacement:.3f}')

MSNR = 20*np.log10(moy/std)

fbar_ref_581 = moy[-1,:,0]*1e3/31/(17**2/32.768)  # one value per ROI
fbar_ref_726 = moy[-1,:,2]*1e3/31/(17**2/32.768)  # one value per ROI

if save_array:
    np.save(fig_folder / 'std', std)
    np.save(fig_folder / 'moy', moy)
    np.save(fig_folder / 'MSNR', MSNR)

#%% MSNR relative to FH2 (contrast, dB) -- all other scan modes vs. FH2
# MSNRc[m, r, ll] = MSNR[m, r, ll] - MSNR[FH2, r, ll]: how many dB each
# scan mode gains (>0) or loses (<0) compared to the full 2D Hadamard
# reference, per ROI and wavelength band.
MSNRc = MSNR - MSNR[0:1, :, :]

# Band labels, e.g. "579-583nm (31ch)", built the same way as the
# MAIN LOOP column titles
band_labels = []
for j in range(len(lambda_central_list)):
    lambda_central = lambda_central_list[j]
    nc = nc_list[j]
    lambda_min = lambda_central - nc + 1
    lambda_max = lambda_central + nc
    lambda_n = lambda_max - lambda_min
    band_labels.append(
        f'{wavelength[0][lambda_min]:.0f}-{wavelength[0][lambda_max]:.0f}nm ({lambda_n}ch)')

# Nice, dependency-free text table: rows = scan modes (FH2 excluded,
# since MSNRc[FH2] is 0 by construction), columns = band x ROI, with a
# blank column between wavelength bands for readability
col_w = 9
row_label_w = 6
group_w = col_w * 2  # width of one (roi1, roi2) group
gap = '  '

header1 = ' ' * row_label_w + gap.join(f'{lbl:^{group_w}}' for lbl in band_labels)
header2 = ' ' * row_label_w + gap.join(
    ''.join(f'{roi_names[r]:>{col_w}}' for r in range(2)) for _ in band_labels)
sep = '-' * len(header1)

print('MSNRc (dB, relative to FH2)')
print(sep)
print(header1)
print(header2)
print(sep)
for m in range(1, len(method_list)):  # skip FH2 (index 0)
    row = f'{method_list[m]:<{row_label_w}}'
    row += gap.join(
        ''.join(f'{MSNRc[m, r, ll]:>{col_w}.2f}' for r in range(2))
        for ll in range(len(lambda_central_list)))
    print(row)
print(sep)

# Same table layout, absolute SNR (linear, = moy/std) for all 5 scan
# modes (FH2 included this time -- there is no reference to subtract out)
SNR_abs = moy / std

print()
print('SNR (linear, = moy / std)')
print(sep)
print(header1)
print(header2)
print(sep)
for m in range(len(method_list)):
    row = f'{method_list[m]:<{row_label_w}}'
    row += gap.join(
        ''.join(f'{SNR_abs[m, r, ll]:>{col_w}.2f}' for r in range(2))
        for ll in range(len(lambda_central_list)))
    print(row)
print(sep)
