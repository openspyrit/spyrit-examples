#%% [markdown]
"""
This script is used to reproduce the results given in Fig. 5.
It compares reconstruction methods using pattern warping (wh) and image warping (wf), 
and the influence of the extended FOV, for a given random elastic deformation field,
on a the whole test set of STL-10.

/!\ This script is long to run (several hours). Results are provided in 'raw_fig_05' folder. /!\
"""

# %% Import bib
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

from pathlib import Path

import spyrit.core.torch as spytorch
from spyrit.core.prep import Unsplit
from spyrit.core.meas import DynamicHadamSplit2d
from spyrit.core.noise import Poisson
from spyrit.core.warp import DeformationField

from spyrit.misc.statistics import Cov2Var, data_loaders_stl10
from spyrit.misc.load_data import download_girder
import spyrit.misc.metrics as score



#%% LOAD IMAGE DATA
paths_params = json.load(open("spyrit-examples/2025_dynamic_TIP/paths.json"))

save_fig = paths_params.get("save_fig")
results_root = Path(paths_params.get("results_root")) / Path('simu/exp_0')
data_root = Path(paths_params.get("data_root"))

img_size = 88  # full image side's size in pixels
meas_size = 64  # measurement pattern side's size in pixels (Hadamard matrix)

time_dim = 1
M = meas_size ** 2
img_shape = (img_size, img_size)
meas_shape = (meas_size, meas_size)
imgs_path = os.path.join(data_root, "spyrit/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dtype = torch.float32
simu_interp = 'bicubic'
matrix_order = 'bilinear'

amp_max = (img_size - meas_size) // 2

path_fig = results_root / Path('fig_05') 
os.makedirs(path_fig, exist_ok=True)
path_deform = results_root / Path('deformations')


batch_size = 16

# Dataloader for STL-10 dataset
mode_run = True
if mode_run:
    dataloaders = data_loaders_stl10(
        data_root,
        img_size=img_size,
        batch_size=batch_size,
        seed=7,
        shuffle=True,
        download=False,
        normalize=True,
    )


#%% Calc finite diff matrix
Dx, Dy = spytorch.neumann_boundary(img_shape)
D2_H1 = Dx.T @ Dx + Dy.T @ Dy
D2_H1 = D2_H1.to(dtype=dtype)  # keep on cpu for linalg operations

Dx_in_X, Dy_in_X = spytorch.neumann_boundary(meas_shape)
D2_H1_in_X = Dx_in_X.T @ Dx_in_X + Dy_in_X.T @ Dy_in_X
D2_H1_in_X = D2_H1_in_X.to(dtype=dtype)

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


# %% Initialize run parameters
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

data_size = len(dataloaders['val'].dataset)
# data_size = batch_size * 2
psnr_array = np.zeros((len(is_in_X_list), len(warping_list), len(reg_list), len(alpha_list), data_size))
ssim_array = np.zeros((len(is_in_X_list), len(warping_list), len(reg_list), len(alpha_list), data_size))

amp_array = np.zeros(len(dataloaders['val']))

#%%
torch.manual_seed(100)  # for reproductible results
N = meas_size ** 2
scale_factor = (torch.tensor(img_shape).to(device=device) - 1)

comp_factor = 10
deform_batch_size = 50

prep_op = Unsplit()

with torch.no_grad():
    for index, (x, _) in enumerate(tqdm(dataloaders['val'], desc='Iterating through dataset.')):    
        x = x.to(device=device, dtype=dtype)
        x = (x + 1) / 2  # to [0, 1]  

        beg_batch, end_batch = index * batch_size, (index + 1) * batch_size        
        beg_deform_batch, end_deform_batch = (index // deform_batch_size) * deform_batch_size, (index // deform_batch_size + 1) * deform_batch_size
        amp_array_dist = np.load(path_deform / Path('amplitudes_index_%d_%d.npy' % (beg_deform_batch, end_deform_batch)))
        amp_array[index] = amp_array_dist[index - beg_deform_batch]

        def_field_compressed = np.load(path_deform / Path('def_field_index_%d_comp_%d.npz' % (index, comp_factor)))

        def_field_torch = torch.from_numpy(def_field_compressed['direct']).to(dtype=dtype, device=device) / comp_factor
        def_field_torch = def_field_torch * 2 / scale_factor - 1 

        def_field = DeformationField(def_field_torch)

        del def_field_torch
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()

        x_motion = def_field(x, 0, (meas_size**2) * 2, mode=simu_interp)

        ## SIMULATE MEASUREMENT
        meas_op = DynamicHadamSplit2d(time_dim=time_dim, h=meas_size, M=M, order=Ord,
                                fast=True, reshape_output=False, img_shape=img_shape,
                                noise_model=torch.nn.Identity(), white_acq=None,
                                dtype=dtype, device=device)

        ## DYNAMIC MATRIX CONSTRUCTION
        for warping in warping_list:
            if warping:
                def_field_torch_inv = torch.from_numpy(def_field_compressed['inverse']).to(dtype=dtype, device=device) / comp_factor
                def_field_torch_inv = def_field_torch_inv * 2 / scale_factor - 1 
                def_field_inv = DeformationField(def_field_torch_inv)
                
                del def_field_torch_inv
                if device == torch.device("cuda"):
                    torch.cuda.empty_cache()

                meas_op.build_H_dyn(def_field_inv, warping=warping, mode=matrix_order)
            else:
                meas_op.build_H_dyn(def_field, warping=warping, mode=matrix_order)

            H_dyn_diff = meas_op.H_dyn_diff

            ## ADD NOISE (POISSON)    
            for alpha_ind, alpha in enumerate(alpha_list):
                meas_op.noise_model = Poisson(alpha)
                
                y1 = meas_op(x_motion)  # before: x = (x + 1) / 2 is done inside. To visualize img in [0,1], the only thing to preprocess measurements is divide by alpha
                y2 = prep_op(y1)
                
                m2 = y2 / alpha
                m2 = m2.cpu()  # send to cpu for linalg operations

                for is_in_X in is_in_X_list:  
                    if is_in_X:
                        A_isornot_in_X = H_dyn_diff.reshape((H_dyn_diff.shape[0], img_size, img_size))[:, amp_max:-amp_max, amp_max:-amp_max].reshape((H_dyn_diff.shape[0], meas_size**2))
                    else:
                        A_isornot_in_X = H_dyn_diff
                    A_isornot_in_X = A_isornot_in_X.cpu()  # send to cpu for linalg operations

                    ## COND
                    sing_vals = torch.linalg.svdvals(A_isornot_in_X)
                    sigma_max = sing_vals[0]

                    for reg_ind, reg in enumerate(reg_list): 
                        eta = etas[int(is_in_X), int(warping), reg_ind, alpha_ind]

                        ## RECO 
                        if reg == 'H1':
                            if is_in_X:
                                x_rec = torch.linalg.solve(A_isornot_in_X.T @ A_isornot_in_X + eta * sigma_max ** 2 * D2_H1_in_X, A_isornot_in_X.T @ m2.reshape((batch_size, meas_size**2, 1)))
                            else:
                                x_rec = torch.linalg.solve(A_isornot_in_X.T @ A_isornot_in_X + eta * sigma_max ** 2 * D2_H1, A_isornot_in_X.T @ m2.reshape((batch_size, meas_size**2, 1)))

                        ## METRICS in X
                        if is_in_X:
                            x_rec = x_rec.reshape((batch_size, 1, meas_size, meas_size))
                            x_rec_in_X = x_rec.reshape((batch_size, 1, meas_size, meas_size))
                        else:
                            x_rec = x_rec.reshape((batch_size, 1, img_size, img_size))
                            x_rec_in_X = x_rec[:, :, amp_max:-amp_max, amp_max:-amp_max]

                        x_in_X = x[:, :, amp_max:-amp_max, amp_max:-amp_max]
                        psnr_list = score.batch_psnr_(x_rec_in_X, x_in_X, r=1)
                        ssim_list = score.batch_ssim(x_rec_in_X, x_in_X)

                        psnr_array[int(is_in_X), int(warping), reg_ind, alpha_ind, beg_batch:end_batch] = np.array(psnr_list)
                        ssim_array[int(is_in_X), int(warping), reg_ind, alpha_ind, beg_batch:end_batch] = np.array(ssim_list)
                        
                        if index % 100 == 0:
                            (path_fig / Path('batch_%d' % index)).mkdir(parents=True, exist_ok=True)
                            plt.imsave(path_fig / Path('batch_%d' % index) / ('mode_' + matrix_order + '_warping_' + str(warping) + '_in_X_' + str(is_in_X) +'_alpha_%d_reg_' % alpha + reg + '_eta_%.2e.pdf' %  eta), x_rec[0, 0].cpu().numpy(), cmap='gray')


        del def_field, x_motion, meas_op, H_dyn_diff, sing_vals, sigma_max, A_isornot_in_X 
        torch.cuda.empty_cache()
        
        if index % 100 == 0:
            # SAVE SCORES EVERY 100 BATCH
            scores = {}

            for alpha_ind, alpha in enumerate(tqdm(alpha_list, desc='Iterating over noise level')):
                scores_alpha = {}
                for reg_ind, reg in enumerate(reg_list):
                    for is_in_X in is_in_X_list:
                        for warping in warping_list:
                            scores_alpha[reg + '_warping_' + str(warping) +'_in_X_' + str(is_in_X)] = {'eta': etas[int(is_in_X), int(warping), reg_ind, alpha_ind], 'PSNRs': psnr_array[int(is_in_X), int(warping), reg_ind, alpha_ind, :].tolist(), 'SSIMs': ssim_array[int(is_in_X), int(warping), reg_ind, alpha_ind, :].tolist()}
                            scores['alpha_%d' % alpha] = scores_alpha

            Path(path_fig).mkdir(parents=True, exist_ok=True)
            with open(str(path_fig / Path('scores.json')), "w") as outfile:
                json.dump(scores, outfile, indent=4)

            motion_params = {}
            motion_params['amp'] = amp_array.tolist()
            motion_params['smoothness'] = 5
            with open(str(path_fig / Path('motion_params.json')), "w") as outfile:
                json.dump(motion_params, outfile, indent=4)

            
    #%% SAVE AS JSON
    scores = {}

    for alpha_ind, alpha in enumerate(tqdm(alpha_list, desc='Iterating over noise level')):
        scores_alpha = {}
        for reg_ind, reg in enumerate(reg_list):
            for is_in_X in is_in_X_list:
                for warping in warping_list:
                    scores_alpha[reg + '_warping_' + str(warping) +'_in_X_' + str(is_in_X)] = {'eta': etas[int(is_in_X), int(warping), reg_ind, alpha_ind], 'PSNRs': psnr_array[int(is_in_X), int(warping), reg_ind, alpha_ind, :].tolist(), 'SSIMs': ssim_array[int(is_in_X), int(warping), reg_ind, alpha_ind, :].tolist()}
                    scores['alpha_%d' % alpha] = scores_alpha

    Path(path_fig).mkdir(parents=True, exist_ok=True)
    with open(str(path_fig / Path('scores.json')), "w") as outfile:
        json.dump(scores, outfile, indent=4)

    motion_params = {}
    motion_params['amp'] = amp_array.tolist()
    motion_params['smoothness'] = 5
    with open(str(path_fig / Path('motion_params.json')), "w") as outfile:
        json.dump(motion_params, outfile, indent=4)
