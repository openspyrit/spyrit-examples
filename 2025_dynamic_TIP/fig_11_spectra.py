# %% Import bib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

from pathlib import Path

from spyrit.misc.disp import torch2numpy
import spyrit.core.torch as spytorch
from spyrit.misc.statistics import Cov2Var
from spyrit.core.meas import HadamSplit2d, DynamicHadamSplit2d
from spyrit.core.prep import Unsplit

from spyrit.core.dual_arm import ComputeHomography, recalibrate, MotionFieldProjector
from spyrit.misc.load_data import read_acquisition, download_girder


# %% Set params and download data if needed
paths_params = json.load(open("spyrit-examples/2025_dynamic_TIP/paths.json"))

save_fig = paths_params.get("save_fig")
results_root = Path(paths_params.get("results_root")) / Path('expe')
data_root = Path(paths_params.get("data_root")) / Path('2025-12-05_motion_color')

homo_folder = Path('homography/')  # folder where the homography files are saved/loaded

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

url_pilot = "https://pilot-warehouse.creatis.insa-lyon.fr/api/v1"
id_files = [
    "6932b461655717d26a63a7d1",  # obj_cat_bicolor_motion_rot_source_white_LED_Walsh_im_64x64_ti_4ms_zoom_x1
    "6932b240655717d26a63a794",  # obj_cat_bicolor_no_motion_source_white_LED_Walsh_im_64x64_ti_4ms_zoom_x1
    "6932af51655717d26a63a774"  # obj_nothing_source_white_LED_Walsh_im_64x64_ti_1ms_zoom_x1
]
try:
    download_girder(url_pilot, id_files, data_root, gc_type="folder")
except Exception as e:
    print("Unable to download from the Pilot warehouse")
    print(e)


# %% DETERMINE HOMOGRAPHY
n = 64
n_acq = 64
data_folder = Path('obj_cat_bicolor_no_motion_source_white_LED_Walsh_im_64x64_ti_4ms_zoom_x1')
data_file_prefix = 'obj_cat_bicolor_no_motion_source_white_LED_Walsh_im_64x64_ti_4ms_zoom_x1'

read_homography = True
save_homography = False

snapshot = True  # with new acquisitions

kp_method = "sift"
read_hand_kp = False

homography_finder = ComputeHomography(data_root, data_folder, data_file_prefix, n, n_acq)

homography_x1 = homography_finder(kp_method, homo_folder=homo_folder, read_homography=read_homography, 
                               save_homography=save_homography, read_hand_kp=read_hand_kp, snapshot=snapshot,
                               show_calib=True)

homography_x1 = homography_x1.to(dtype=dtype, device=device)


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

Cov_acq = np.load(local_folder / ('Cov_{}x{}'.format(n_acq, n_acq) + '.npy'))
Ord_acq = Cov2Var(Cov_acq)
Ord = torch.from_numpy(Ord_acq)


# %% STATIC ACQUISITION
n = 64
n_acq = 64

N = n ** 2
M = N

data_folder = Path('obj_cat_bicolor_no_motion_source_white_LED_Walsh_im_64x64_ti_4ms_zoom_x1')
data_file_prefix = 'obj_cat_bicolor_no_motion_source_white_LED_Walsh_im_64x64_ti_4ms_zoom_x1'

metadata, meas = read_acquisition(data_root, data_folder, data_file_prefix)

acquisition_parameters = metadata[1]
spectro_parameters = metadata[2]

t_stat = spectro_parameters.integration_time_ms

wavelengths = torch.from_numpy(acquisition_parameters.wavelengths)

n_wav = 128  # number of spectral channels to reconstruct
strategy = 'bin'  # 'slice' or 'bin'

meas2_torch = torch.from_numpy(meas).to(dtype=dtype, device=device)

if strategy == 'bin':
    # Bin spectral channels, with each bin covering consecutive slices
    bin_size = meas2_torch.shape[1] // n_wav
    y1_exp = torch.stack([
        meas2_torch[:, i * bin_size : (i + 1) * bin_size].mean(dim=1)
        for i in range(n_wav)
    ], dim=1).moveaxis(-1, 0)

    wav = torch.stack([
        wavelengths[i * bin_size : (i + 1) * bin_size].mean()
        for i in range(n_wav)
    ], dim=0).numpy()

elif strategy == 'slice':
    # Select specific slices for each spectral channel
    bin_size = meas2_torch.shape[1] // n_wav
    y1_exp = torch.stack([
        meas2_torch[:, i * bin_size : (i + 1) * bin_size].median(dim=1).values
        for i in range(n_wav)
    ], dim=1).moveaxis(-1, 0)

    wav = torch.stack([
        wavelengths[i * bin_size : (i + 1) * bin_size].median()
        for i in range(n_wav)
    ], dim=0).numpy()

else:
    raise ValueError("Invalid strategy. Choose 'bin' or 'slice'.")


# %%
meas_op_stat = HadamSplit2d(M=M, h=n, order=Ord, dtype=dtype, device=device) 

prep_op = Unsplit()

y2_exp = prep_op(y1_exp)


x_stat = meas_op_stat.fast_pinv(y2_exp) 
x_stat /= t_stat


# %% DYNAMIC ACQUISITION
n = 64
n_acq = 64

N = n ** 2
M = N

data_folder = Path('obj_cat_bicolor_motion_rot_source_white_LED_Walsh_im_64x64_ti_4ms_zoom_x1')
data_file_prefix = 'obj_cat_bicolor_motion_rot_source_white_LED_Walsh_im_64x64_ti_4ms_zoom_x1'

metadata, meas = read_acquisition(data_root, data_folder, data_file_prefix)

acquisition_parameters = metadata[1]
spectro_parameters = metadata[2]
cam_parameters = metadata[-1]


wavelengths = torch.from_numpy(acquisition_parameters.wavelengths)
patterns = np.array(acquisition_parameters.patterns)
T = acquisition_parameters.total_callback_acquisition_time_s
zoom_factor = int(data_file_prefix[-1])  # set like this for now but ideally it would be in the metadata
# zoom_factor = acquisition_parameters.zoom  
t_dyn = spectro_parameters.integration_time_ms

## UPDATE homography according to compression factor
c1 = (n - 1) / 2
c2 = (n - 1) / 2
zoom_homography = torch.tensor([[zoom_factor, 0, (1 - zoom_factor) * c1],
                                [0, zoom_factor, (1 - zoom_factor) * c2],
                                [0, 0, 1]], dtype=dtype, device=device)

homography = zoom_homography @ homography_x1

meas2_torch = torch.from_numpy(meas).to(dtype=dtype, device=device)

if strategy == 'bin':
    # Bin spectral channels, with each bin covering consecutive slices
    bin_size = meas2_torch.shape[1] // n_wav
    y1_exp = torch.stack([
        meas2_torch[:, i * bin_size : (i + 1) * bin_size].mean(dim=1)
        for i in range(n_wav)
    ], dim=1).moveaxis(-1, 0)

    wav = torch.stack([
        wavelengths[i * bin_size : (i + 1) * bin_size].mean()
        for i in range(n_wav)
    ], dim=0).numpy()

elif strategy == 'slice':
    # Select specific slices for each spectral channel
    bin_size = meas2_torch.shape[1] // n_wav
    y1_exp = torch.stack([
        meas2_torch[:, i * bin_size : (i + 1) * bin_size].median(dim=1).values
        for i in range(n_wav)
    ], dim=1).moveaxis(-1, 0)

    wav = torch.stack([
        wavelengths[i * bin_size : (i + 1) * bin_size].median()
        for i in range(n_wav)
    ], dim=0).numpy()

else:
    raise ValueError("Invalid strategy. Choose 'bin' or 'slice'.")



# %% PREP OP
y2_exp = prep_op(y1_exp)


# %% STATIC RECO
f_stat = meas_op_stat.fast_pinv(y2_exp)
f_stat /= t_dyn


#%% load empty acquisition for dynamic flat-field correction
data_folder_white = 'obj_nothing_source_white_LED_Walsh_im_64x64_ti_1ms_zoom_x1'
data_file_prefix_white = 'obj_nothing_source_white_LED_Walsh_im_64x64_ti_1ms_zoom_x1'

_, meas_empty = read_acquisition(data_root, data_folder_white, data_file_prefix_white)

# Process measurements
meas_empty_torch = torch.from_numpy(meas_empty).to(dtype=dtype, device=device)
y_empty = torch.mean(meas_empty_torch, axis=1).view(1, -1)
y_empty_diff = prep_op(y_empty)

# Reconstruct SP image
w = meas_op_stat.fast_pinv(y_empty_diff)

zoom_homography_inv = torch.linalg.inv(zoom_homography)
w = recalibrate(w.unsqueeze(0), (n, n), zoom_homography_inv, amp_max=0)


# %% stat acq flat field
x_stat_ff = x_stat / w.squeeze(0)

x_stat_ff_np = torch2numpy(torch.rot90(x_stat_ff, k=2, dims=(-2, -1)))


# %% DEFORMATION FILES
deform_path = Path('../omigod_res/color_cat')

deform_folder = Path('rot_x1')
deform_prefix = 'rot'


# %% ESTIM MOTION FROM CMOS CAMERA
amp_max = 10
l = n + 2 * amp_max
translation = (0, 0)
frame_ref = 0 #17 #570 #278 #28   # frame ref in CMOS referential
n_ppg = 16  # TODO: download updated data
# n_ppg = cam_parameters.gate_period

projector = MotionFieldProjector(deform_path / deform_folder, deform_prefix, n, M, n_ppg, T, 
                        frame_ref=frame_ref, homography=homography, translation=translation,
                        dtype=dtype, device=torch.device('cpu'))  # cpu to avoid cuda OOM



# %%
forme_interp = 'bilinear'

l = n + 2 * amp_max
L = l ** 2

time_dim = 1


# %% 
y2_exp = y2_exp.to('cpu')  # send to cpu for linalg operations

meas_op = DynamicHadamSplit2d(time_dim=time_dim, h=n, M=M, order=Ord, img_shape=(l, l),
                              white_acq=w, dtype=dtype, device=device)

# %% CALC FINITE DIFFERENCE MATRIX
Dx, Dy = spytorch.neumann_boundary((n, n))
D2_in_X = Dx.T @ Dx + Dy.T @ Dy

Dx, Dy = spytorch.neumann_boundary((l, l))
D2 = Dx.T @ Dx + Dy.T @ Dy

D2_in_X, D2 = D2_in_X.type(dtype=dtype), D2.type(dtype=dtype)


# %% ######### IMAGE WARPING DYNAMIC MEASUREMENT MATRIX #########
warping = 'image'
def_field = projector(warping=warping, amp_max=amp_max).to(device)
meas_op.build_dynamic_forward(def_field, warping=warping, mode=forme_interp)

H_dyn = meas_op.H_dyn


# %% send to cpu for linalg operations
H_dyn = H_dyn.to('cpu') # send to cpu for linalg operations

s = torch.linalg.svdvals(H_dyn)
print(f'H_dyn spectrum is [{s[-1]:.2f}, {s[0]:.2f}] (condition number: {s[0]/s[-1]:.2e})')
norm = s[0]


# %% RECO WITH LAPACK (LU factorization)
eta = 1e-2

f_wf_Xext = torch.linalg.solve(H_dyn.T @ H_dyn + eta * norm ** 2 * D2, H_dyn.T @ y2_exp.T)

f_wf_Xext_np = np.rot90(torch2numpy(f_wf_Xext.view((l, l, n_wav)).moveaxis(-1, 0)), 2, axes=(1, 2))
f_wf_Xext_np /= t_dyn


# %% compare static and dynamic reconstructions for two spectra: one in red, one in green

vmin, vmax = f_wf_Xext_np.mean(axis=0).min(), f_wf_Xext_np.mean(axis=0).max()

x_stat_ff_np_wide = np.zeros((n_wav, l, l))
x_stat_ff_np_wide[:, amp_max : amp_max + n, amp_max : amp_max + n] = x_stat_ff_np

# points on the dynamic
q1_x, q1_y = 39, 65  # green
q2_x, q2_y = 33, 34  # red

# points on the static
p1_x, p1_y = q1_x + 1, q1_y - 3  # green
p2_x, p2_y = q2_x, q2_y - 3  # red



fig, ax = plt.subplots(1, 3, figsize=(15, 5))


ax[0].imshow(x_stat_ff_np_wide.mean(axis=0), cmap='gray', vmin=vmin, vmax=vmax)
ax[0].plot(p1_x, p1_y, marker='o', markersize=8, color='green')
ax[0].plot(p2_x, p2_y, marker='o', markersize=8, color='red')
rect_fov = Rectangle((amp_max, amp_max), n, n, linewidth=3, edgecolor='blue', facecolor='none')
ax[0].add_patch(rect_fov)
ax[0].set_title(f"Stat rec ff / {t_stat} ms", fontsize=20)


ax[1].imshow(f_wf_Xext_np.mean(axis=0), cmap='gray', vmin=vmin, vmax=vmax)
ax[1].plot(q1_x, q1_y, marker='o', markersize=8, color='gold')
ax[1].plot(q2_x, q2_y, marker='o', markersize=8, color='orange')
rect_fov = Rectangle((amp_max, amp_max), n, n, linewidth=2, edgecolor='blue', facecolor='none')
ax[1].add_patch(rect_fov)
ax[1].set_title(f"Dyn rec ff / {t_dyn} ms", fontsize=20)

spec1_stat = x_stat_ff_np_wide[:, p1_y, p1_x]
spec2_stat = x_stat_ff_np_wide[:, p2_y, p2_x]

spec1_dyn = f_wf_Xext_np[:, q1_y, q1_x]
spec2_dyn = f_wf_Xext_np[:, q2_y, q2_x]

ax[2].plot(wav, spec1_stat, label='no motion 1', color='green', lw=1) #, linestyle=(3, (1, 1)))
ax[2].plot(wav, spec2_stat, label='no motion 2', color='red', lw=1) #, linestyle=(3, (1, 1)))

ax[2].plot(wav, spec1_dyn, label='dynamic 1', color='gold', lw=1) #, linestyle=(3, (1, 1)))
ax[2].plot(wav, spec2_dyn, label='dynamic 2', color='orange', lw=1) #, linestyle=(3, (1, 1)))

ax[2].set_title("Spectra", fontsize=20)
ax[2].set_xlabel("Wavelength (nm)", fontsize=18)
plt.legend(fontsize=12)
plt.show()


# %% save only subplot 1
plt.figure(figsize=(5, 5))
plt.imshow(x_stat_ff_np_wide.mean(axis=0), cmap='gray', vmin=vmin, vmax=vmax)
plt.plot(p1_x, p1_y, marker='o', markersize=8, color='green')
plt.plot(p2_x, p2_y, marker='o', markersize=8, color='red')
rect_fov = Rectangle((amp_max, amp_max), n, n, linewidth=4, edgecolor='blue', facecolor='none')
plt.gca().add_patch(rect_fov)
plt.axis('off')
plt.tight_layout()
if save_fig:
    path_fig = results_root / Path('fig_spec_color_cat') / Path('rot_x1')
    path_fig.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_fig / f'fig_spec_stat_rec.pdf', bbox_inches='tight', pad_inches=0, transparent=True)
plt.show()

# %% save only subplot 2
plt.figure(figsize=(5, 5))
plt.imshow(f_wf_Xext_np.mean(axis=0), cmap='gray', vmin=vmin, vmax=vmax)
plt.plot(q1_x, q1_y, marker='o', markersize=8, color='gold')
plt.plot(q2_x, q2_y, marker='o', markersize=8, color='orange')
rect_fov = Rectangle((amp_max, amp_max), n, n, linewidth=4, edgecolor='blue', facecolor='none')
plt.gca().add_patch(rect_fov)
plt.axis('off')
plt.tight_layout()
if save_fig:
    path_fig = results_root / Path('fig_spec_color_cat') / Path('rot_x1')
    path_fig.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_fig / f'fig_spec_dyn_rec.pdf', bbox_inches='tight', pad_inches=0, transparent=True)
plt.show()

# %% save only the spectrums, but better
plt.figure(figsize=(5, 5))
plt.plot(wav, spec1_stat, label='no motion 1', color='green', lw=1) #, linestyle=(3, (1, 1)))
plt.plot(wav, spec2_stat, label='no motion 2', color='red', lw=1) #, linestyle=(3, (1, 1)))

plt.plot(wav, spec1_dyn, label='dynamic 1', color='gold', lw=1) #, linestyle=(3, (1, 1)))
plt.plot(wav, spec2_dyn, label='dynamic 2', color='orange', lw=1) #, linestyle=(3, (1, 1)))

# plt.title("Spectra", fontsize=20)
plt.xlabel("Wavelength (nm)", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.tight_layout()
if save_fig:
    path_fig = results_root / Path('fig_spec_color_cat') / Path('rot_x1')
    path_fig.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_fig / f'fig_spec_spectra.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()


# %% give errors in the green region
mse_roi = np.mean((spec1_stat - spec1_dyn) ** 2)
print(f'RMSE between static and dynamic ROI spectrums in the green region: {mse_roi:.2e}')

mse_ref = np.mean(spec1_stat ** 2)
rel_mse_roi = mse_roi / mse_ref
print(f'Relative RMSE between static and dynamic ROI spectrums in the green region: {rel_mse_roi:.4%}')


# %% give errors in the red region
mse_roi = np.mean((spec2_stat - spec2_dyn) ** 2)
print(f'RMSE between static and dynamic ROI spectrums in the red region: {mse_roi:.2e}')

mse_ref = np.mean(spec2_stat ** 2)
rel_mse_roi = mse_roi / mse_ref
print(f'Relative RMSE between static and dynamic ROI spectrums in the red region: {rel_mse_roi:.4%}')

# %%
