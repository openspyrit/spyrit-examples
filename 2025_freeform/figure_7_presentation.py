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

h = 128  # image size hxh

mask_type = None        # 'skew' or None to load a PNG
mask_filename = 'cat_roi.png'  # only if mask_type is not 'skew'


norm = 32768 # time budget in ms


fig_folder = Path('./figures/presentation')

save_tag = True

# plt.rcParams['text.usetex'] = True

#%% Define mask
from matplotlib.pyplot import imread
from spyrit.misc.disp import imagesc

# Build skew mask
mask = imread(mask_filename)
mask = mask[:,:,0]
mask = mask.astype(np.bool)
mask = torch.from_numpy(mask)

ind_array = np.where(mask == True)
ind_array = (torch.from_numpy(ind_array[0]),torch.from_numpy(ind_array[1]))

mask_name = mask_filename[:-8]
    
N_pixel = len(ind_array[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mask = mask.to(device=device)

# shape and plot
imagesc(mask.cpu())

#%% read data
# Load experimental data
# ====================================================================
import json
import ast

data_folder = Path(r"./data/2025-09-25_freeform_publication")

measure = 'signal'      # 'noise' #     ArithmeticError
ti = 1
moy_meas = False        # True#

method = 'substraction'     # 'classical'#
NR = 2                   # Number of repetition in the case of "substraction" method
acq_list = ["FH2", "H1", "MH2", "S1", "RS"]

    
if ti == 1 and moy_meas == False:
    data_title = [
    r'obj_StarSector_source_white_LED_Walsh_im_128x128_ti_1ms_zoom_x1',
    r'obj_StarSector_source_white_LED_hadam1d_cat_8192_im_128x128_ti_4ms_zoom_x1',
    r'obj_StarSector_source_white_LED_hadam2d_cat_32768_im_128x128_ti_1ms_zoom_x1',
    r'obj_StarSector_source_white_LED_smatrix_cat_4095_im_128x128_ti_8ms_zoom_x1',
    r'obj_StarSector_source_white_LED_raster_cat_4096_im_128x128_ti_8ms_zoom_x1']

elif ti == 1 and moy_meas == True:
    data_title = [
    r'obj_Nothing_source_white_LED_Walsh_im_128x128_ti_1ms_zoom_x1',
    r'obj_Nothing_source_white_LED_hadam1d_cat_8192_im_128x128_ti_4ms_zoom_x1',
    r'obj_Nothing_source_white_LED_hadam2d_cat_32768_im_128x128_ti_1ms_zoom_x1',
    r'obj_Nothing_source_white_LED_smatrix_cat_4095_im_128x128_ti_8ms_zoom_x1',
    r'obj_Nothing_source_white_LED_raster_cat_4096_im_128x128_ti_8ms_zoom_x1']

elif ti == 2 and moy_meas == False:
    data_title = [
    r'obj_StarSector_source_white_LED_Walsh_im_128x128_ti_2ms_zoom_x1',
    r'obj_StarSector_source_white_LED_hadam1d_cat_8192_im_128x128_ti_8ms_zoom_x1',
    r'obj_StarSector_source_white_LED_hadam2d_cat_32768_im_128x128_ti_2ms_zoom_x1',
    r'obj_StarSector_source_white_LED_smatrix_cat_4095_im_128x128_ti_16ms_zoom_x1',
    r'obj_StarSector_source_white_LED_raster_cat_4096_im_128x128_ti_16ms_zoom_x1']

# black patterns
if ti == 1 and moy_meas == False:
    black_title = [
    r'obj_StarSector_source_white_LED_black_32768_im_128x128_ti_1ms_zoom_x1',
    r'obj_StarSector_source_white_LED_black_8192_im_128x128_ti_4ms_zoom_x1',
    r'obj_StarSector_source_white_LED_black_32768_im_128x128_ti_1ms_zoom_x1',
    r'obj_StarSector_source_white_LED_black_4096_im_128x128_ti_8ms_zoom_x1',
    r'obj_StarSector_source_white_LED_black_4096_im_128x128_ti_8ms_zoom_x1'
    ]

elif ti == 1 and moy_meas == True:
    black_title = [
    r'obj_Nothing_source_white_LED_black_32768_im_128x128_ti_1ms_zoom_x1',
    r'obj_Nothing_source_white_LED_black_8192_im_128x128_ti_4ms_zoom_x1',
    r'obj_Nothing_source_white_LED_black_32768_im_128x128_ti_1ms_zoom_x1',
    r'obj_Nothing_source_white_LED_black_4096_im_128x128_ti_8ms_zoom_x1',
    r'obj_Nothing_source_white_LED_black_4096_im_128x128_ti_8ms_zoom_x1'
    ]
    
elif ti == 2 and moy_meas == False:
    black_title = [
    r'obj_StarSector_source_white_LED_black_32768_im_128x128_ti_2ms_zoom_x1',
    r'obj_StarSector_source_white_LED_black_8192_im_128x128_ti_8ms_zoom_x1',
    r'obj_StarSector_source_white_LED_black_32768_im_128x128_ti_2ms_zoom_x1',
    r'obj_StarSector_source_white_LED_black_4096_im_128x128_ti_16ms_zoom_x1',
    r'obj_StarSector_source_white_LED_black_4096_im_128x128_ti_16ms_zoom_x1'
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

dark_folder = Path(r"./data/2025-09-11_freeform_SNR")
dark_title = [r'obj_black_source_No source_raster_cat_4096_im_128x128_ti_8ms_zoom_x1']

# Load dark measurements (For info, not used later)
data_dark, _, _ = load_spihim(dark_folder, dark_title)
mu_dark = data_dark[0].mean(axis=0)

# Denoise
spl_arr = np.zeros((len(black_exp), len(wavelength[0])))

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
    
#%% substract stray light
  
acqui_size = []
for j in range(len(data_exp)):
    
    # substract parasite light
    acqui_size.append(int(data_exp[j].shape[0] / NR))
    for nM in range(acqui_size[j]*NR):
        data_exp[j][nM, :] = data_exp[j][nM, :] - spl_arr[j,:]


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

# lambda_central_list = [512, 512, 1900, 1900] # no signal below 15 and above 2038
# nc_list = [1, 15, 1, 15]

lambda_central_list = [515, 515, 1800, 1800] # no signal below 15 and above 2038
nc_list = [16, 3, 16, 3]
# lambda_central_list = [512] # no signal below 15 and above 2038
# nc_list = [20]

# Plot options  
fs = 12                 # Font size
dpi_fig = 600
cbar_pos = 'right'     #'right'# colorbar position

roi_x_coord = [90,117]
roi_y_coord = [51,82]

print_metric = 'PSNR' # 'SNR'# 

for ll in range(len(lambda_central_list)):
    
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
    
    x_H2dF   = meas_op.fast_pinv(prep(y))  
    x_H2dF_2 = meas_op.fast_pinv(prep(y2))
    
    #--------------------------------------------------------------------------
    #% Arbitrary shape -- Identity matrix 1D
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
    # Reconstruction
    x_rec_2 = ifwalsh_S_torch(y)  
    x_S1d = torch.zeros_like(x_H2dF)
    x_S1d[ind_array_0_S,ind_array_1_S] = x_rec_2
    
    # Delete hot pixel
    mm = (x_S1d[117,83] + x_S1d[118,82] + x_S1d[117,82]) / 3
    x_S1d[118,83] = mm
    
    x_S1d[~mask] = x_rec_2.min()    # Set out-of-ROI pixels to minimum value
    
    # Reconstruction
    x_rec_2 = ifwalsh_S_torch(y2)
    x_S1d_2 = torch.zeros_like(x_H2dF)
    x_S1d_2[ind_array_0_S,ind_array_1_S] = x_rec_2    
    
    # Delete hot pixel
    mm = (x_S1d_2[117,83] + x_S1d_2[118,82] + x_S1d_2[117,82]) / 3
    x_S1d_2[118,83] = mm
    
    x_S1d_2[~mask] = x_rec_2.min()   # Set out-of-ROI pixels to minimum value
    
    # Compute metrics in freeform region
    roi = x_S1d[mask]
    roi_2 = x_S1d_2[mask]
    
    roi_sub  = (roi - roi_2) / 2**.5  # division compensates for std increase due to difference
    roi_mean = (roi + roi_2) / 2
    
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
    x_H2dM = meas_op.fast_pinv(prep(y))
    x_H2dM[~mask] = x_H2dM[mask].min()  # Set out-of-ROI pixels to minimum value
    
    x_H2dM_2 = meas_op.fast_pinv(prep(y2))
    x_H2dM_2[~mask] = x_H2dM_2.min()
           
    #--------------------------------------------------------------------------
    #% Plot all images individualy
    #--------------------------------------------------------------------------
    plt.figure()
    im = plt.imshow(x_H2dF.cpu(), cmap="gray")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar = add_colorbar(im, cbar_pos)
    cbar.ax.tick_params(labelsize=fs)
    if save_tag:
        plt.savefig(fig_folder / f'FH2_{lambda_central_list[ll]}_nm.png', bbox_inches='tight', dpi=dpi_fig)


    plt.figure()
    im = plt.imshow(x_I1d.cpu(), cmap="gray")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar = add_colorbar(im, cbar_pos)
    cbar.ax.tick_params(labelsize=fs)
    if save_tag:
        plt.savefig(fig_folder / f'RS_{lambda_central_list[ll]}_nm.png', bbox_inches='tight', dpi=dpi_fig)
    
    plt.figure()
    im = plt.imshow(x_H2dM.cpu(), cmap="gray")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar = add_colorbar(im, cbar_pos)
    cbar.ax.tick_params(labelsize=fs)
    if save_tag:
        plt.savefig(fig_folder / f'MH2_{lambda_central_list[ll]}_nm.png', bbox_inches='tight', dpi=dpi_fig)
    
    plt.figure()
    im = plt.imshow(x_H1d.cpu(), cmap="gray")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar = add_colorbar(im, cbar_pos)
    cbar.ax.tick_params(labelsize=fs)
    if save_tag:
        plt.savefig(fig_folder / f'H1_{lambda_central_list[ll]}_nm.png', bbox_inches='tight', dpi=dpi_fig)
    
    plt.figure()
    im = plt.imshow(x_S1d.cpu(), cmap="gray")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar = add_colorbar(im, cbar_pos)
    cbar.ax.tick_params(labelsize=fs)
    if save_tag:
        plt.savefig(fig_folder / f'S1_{lambda_central_list[ll]}_nm.png', bbox_inches='tight', dpi=dpi_fig)  
        plt.close('all')
