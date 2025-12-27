# %% Import bib
import torch
import numpy as np
import matplotlib.pyplot as plt
import json

from pathlib import Path

from spyrit.misc.disp import torch2numpy
import spyrit.core.torch as spytorch
from spyrit.misc.statistics import Cov2Var
from spyrit.core.meas import HadamSplit2d, DynamicHadamSplit2d
from spyrit.core.prep import Unsplit

from spyrit.core.dual_arm import ComputeHomography, recalibrate, MotionFieldProjector
from spyrit.misc.load_data import read_acquisition, download_girder
from spyrit.misc.disp import get_frame, save_motion_video, save_field_video


# %% DETERMINE HOMOGRAPHY
paths_params = json.load(open("spyrit-examples/2025_dynamic_TIP/paths.json"))

save_fig = paths_params.get("save_fig")
results_root = Path(paths_params.get("results_root"))
data_root = Path(paths_params.get("data_root")) / Path('2025-12-05_motion_color')

homo_folder = Path('homography/')  # folder where the homography files are saved/loaded

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n = 64
n_acq = 64
data_folder = Path('obj_no_motion_cat_DoF-811_source_white_LED_f80mm-P2_Walsh_im_64x64_ti_1ms_zoom_x1')
data_file_prefix = 'obj_no_motion_cat_DoF-811_source_white_LED_f80mm-P2_Walsh_im_64x64_ti_1ms_zoom_x1'

read_homography = True
save_homography = False
snapshot = True
kp_method = "hand"
read_hand_kp = True

homography_finder = ComputeHomography(data_root, data_folder, data_file_prefix, n, n_acq)

homography_x1 = homography_finder(kp_method, homo_folder=homo_folder, read_homography=read_homography, 
                               save_homography=save_homography, read_hand_kp=read_hand_kp, snapshot=snapshot,
                               show_calib=True)

homography_x1 = homography_x1.to(dtype=dtype, device=device)



# %% READ DATA
n = 64
n_acq = 64

N = n ** 2
M = N

data_folder = Path('obj_motion_Diag-UL-BR_starSector_DoF-811_source_white_LED_f80mm-P2_Walsh_im_64x64_ti_1.5ms_zoom_x1')
data_file_prefix = 'obj_motion_Diag-UL-BR_starSector_DoF-811_source_white_LED_f80mm-P2_Walsh_im_64x64_ti_1.5ms_zoom_x1'

metadata, meas = read_acquisition(data_root, data_folder, data_file_prefix)

acquisition_parameters = metadata[1]
cam_parameters = metadata[-1]

wavelengths = torch.from_numpy(acquisition_parameters.wavelengths)
patterns = np.array(acquisition_parameters.patterns)
T = acquisition_parameters.total_callback_acquisition_time_s
# zoom_factor = acquisition_parameters.pattern_compression 
zoom_factor = int(data_file_prefix[-1])  # set like this for now but ideally it would be in the metadata
# zoom_factor = acquisition_parameters.zoom  

## UPDATE homography according to compression factor
c1 = (n - 1) / 2
c2 = (n - 1) / 2
zoom_homography = torch.tensor([[zoom_factor, 0, (1 - zoom_factor) * c1],
                                [0, zoom_factor, (1 - zoom_factor) * c2],
                                [0, 0, 1]], dtype=dtype, device=device)

homography = zoom_homography @ homography_x1

n_wav = 4  # number of spectral channels to reconstruct
strategy = 'slice'  # 'slice' or 'bin'

meas2_torch = torch.from_numpy(meas).to(dtype=dtype, device=device)

y1_pan = meas2_torch.mean(dim=1).view(1, -1)

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
    indices = [127, 639, 940, 1407]  # example indices for 4 channels
    y1_exp = meas2_torch[:, indices].moveaxis(-1, 0)
    wav = wavelengths[indices].numpy()

else:
    raise ValueError("Invalid strategy. Choose 'bin' or 'slice'.")


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


# %% PREP OP
prep_op = Unsplit()
y2_exp = prep_op(y1_exp)
y2_pan = prep_op(y1_pan)


# %% STATIC RECO
meas_op_stat = HadamSplit2d(M=M, h=n, order=Ord, dtype=dtype, device=device) 

f_stat = meas_op_stat.fast_pinv(y2_pan)


#%% load empty acquisition for dynamic flat-field correction
data_folder_white = 'obj_Empty_DoF-811_source_white_LED_f80mm-P2_Walsh_im_64x64_ti_1ms_zoom_x1'
data_file_prefix_white = 'obj_Empty_DoF-811_source_white_LED_f80mm-P2_Walsh_im_64x64_ti_1ms_zoom_x1'

_, meas_empty = read_acquisition(data_root, data_folder_white, data_file_prefix_white)

# Process measurements
meas_empty_torch = torch.from_numpy(meas_empty).to(dtype=dtype, device=device)
y_empty = torch.mean(meas_empty_torch, axis=1).view(1, -1)
y_empty_diff = prep_op(y_empty)

# Reconstruct SP image
w = meas_op_stat.fast_pinv(y_empty_diff)

zoom_homography_inv = torch.linalg.inv(zoom_homography)
w = recalibrate(w.unsqueeze(0), (n, n), zoom_homography_inv, amp_max=0)



# %% DEFORMATION FILES
deform_path = Path('../omigod_res/extended_FOV2')

deform_folder = Path('star_diag')
deform_prefix = 'star'


# %% ESTIM MOTION FROM CMOS CAMERA
amp_max = 20
l = n + 2 * amp_max
translation = (0, 0)
frame_ref = 0 #17 #570 #278 #28   # frame ref in CMOS referential
n_ppg = 16  # TODO: download updated data
# n_ppg = cam_parameters.gate_period

projector = MotionFieldProjector(deform_path / deform_folder, deform_prefix, n, M, n_ppg, T, 
                        frame_ref=frame_ref, homography=homography, translation=translation,
                        dtype=dtype, device=torch.device('cpu'))



# %% CMOS REFERENCE FRAME
full_path = data_root / data_folder / (data_file_prefix + '_video.avi')
movie = str(full_path)
g_frame0 = get_frame(movie, frame_ref)

homography_inv = torch.linalg.inv(homography)
g_frame0 = torch.from_numpy(g_frame0).unsqueeze(0).unsqueeze(0)
g_frame0 = g_frame0.to(dtype=dtype, device=device)
img_cmos_calibrated = recalibrate(g_frame0, (l, l), homography_inv, amp_max=amp_max)
img_cmos_calibrated_np = np.rot90(torch2numpy(img_cmos_calibrated[0, 0]), 2)

plt.imshow(img_cmos_calibrated_np, cmap='gray')
plt.title('CMOS reference frame')


# %%
forme_interp = 'bilinear'

l = n + 2 * amp_max
L = l ** 2

time_dim = 1

eta = 1e-2


# %% 
y2_pan = y2_pan.to('cpu')  # send to cpu for linalg operations

Dx, Dy = spytorch.neumann_boundary((n, n))
D2_in_X = Dx.T @ Dx + Dy.T @ Dy

Dx, Dy = spytorch.neumann_boundary((l, l))
D2 = Dx.T @ Dx + Dy.T @ Dy

D2_in_X, D2 = D2_in_X.type(dtype=dtype), D2.type(dtype=dtype)


#%% Save deformation field as quiver plot video
if save_fig:
    warping = 'pattern'
    def_field = projector(warping=warping, amp_max=amp_max).to(device)

    path_fig = results_root / data_folder
    Path(path_fig).mkdir(parents=True, exist_ok=True)
    video_path = path_fig / 'deformation_quiver.mp4'

    n_frames = 200
    step = 6  # subsampling for arrows
    fps = 30

    save_field_video(def_field, video_path, n_frames=n_frames, step=step, fps=fps, figsize=(6, 6), dpi=200, scale=1, fs=16,
                     amp_max=amp_max, box_color='blue', box_linewidth=2)

# %% get deformation field
warping = 'image'
def_field = projector(warping=warping, amp_max=amp_max).to(device)


# %% Without dynamic flat field correction
meas_op_no_gain = DynamicHadamSplit2d(time_dim=time_dim, h=n, M=M, order=Ord, img_shape=(l, l),
                              white_acq=None, dtype=dtype, device=device)


# %% ######### IMAGE WARPING DYNAMIC MEASUREMENT MATRIX #########
meas_op_no_gain.build_dynamic_forward(def_field, warping=warping, mode=forme_interp)

H_dyn = meas_op_no_gain.H_dyn

# %% send to cpu for linalg operations
H_dyn = H_dyn.to('cpu')

s = torch.linalg.svdvals(H_dyn)
print(f'H_dyn spectrum is [{s[-1]:.2f}, {s[0]:.2f}] (condition number: {s[0]/s[-1]:.2e})')
norm = s[0]

# RECO WITH LAPACK (LU factorization)
f_wf_Xext_no_gain = torch.linalg.solve(H_dyn.T @ H_dyn + eta * norm ** 2 * D2, H_dyn.T @ y2_pan.T)


# %% With dynamic flat field correction
meas_op = DynamicHadamSplit2d(time_dim=time_dim, h=n, M=M, order=Ord, img_shape=(l, l),
                              white_acq=w, dtype=dtype, device=device)


# %% ######### IMAGE WARPING DYNAMIC MEASUREMENT MATRIX #########
meas_op.build_dynamic_forward(def_field, warping=warping, mode=forme_interp)

H_dyn = meas_op.H_dyn

# %% send to cpu for linalg operations
H_dyn = H_dyn.to('cpu')

s = torch.linalg.svdvals(H_dyn)
print(f'H_dyn spectrum is [{s[-1]:.2f}, {s[0]:.2f}] (condition number: {s[0]/s[-1]:.2e})')
norm = s[0]

# RECO WITH LAPACK (LU factorization)
f_wf_Xext = torch.linalg.solve(H_dyn.T @ H_dyn + eta * norm ** 2 * D2, H_dyn.T @ y2_pan.T)


# %% Fig. 12
f_white_np = np.zeros((l, l))

f_cmos_ref = np.rot90(torch2numpy(img_cmos_calibrated.squeeze()), 2)
f_white_np[amp_max:n+amp_max, amp_max:n+amp_max] = np.rot90(torch2numpy(w.squeeze()), 2)
f_wf_Xext_np_no_gain = np.rot90(torch2numpy(f_wf_Xext_no_gain.mean(axis=1)).reshape((l, l)), 2)
f_wf_Xext_np = np.rot90(torch2numpy(f_wf_Xext.mean(axis=1)).reshape((l, l)), 2)

fs = 30

fig, ax = plt.subplots(1, 4,  figsize=(20, 10))

ax[0].imshow(f_cmos_ref, cmap='gray')
ax[0].set_title('(a) CMOS', fontsize=fs)
ax[0].axis('off')

ax[1].imshow(f_white_np, cmap="gray")
ax[1].set_title('(b) g_0', fontsize=fs)
ax[1].axis('off')

ax[2].imshow(f_wf_Xext_np_no_gain, cmap="gray")
ax[2].set_title('(d) g = 1', fontsize=fs)
ax[2].axis('off')

ax[3].imshow(f_wf_Xext_np, cmap="gray")
ax[3].set_title('(f) g = g_0', fontsize=fs)
ax[3].axis('off')

plt.tight_layout()
if save_fig:
    path_fig = results_root / data_folder
    Path(path_fig).mkdir(parents=True, exist_ok=True)
    plt.savefig(path_fig / f'overview.png', dpi=300)
plt.show()

# %% Recover video from reference frame and deformation field
x_rec_video = def_field(f_wf_Xext.mean(axis=1).view((1, 1, l, l)).to(dtype=dtype, device=device), 0, 2 * M, mode='bilinear')
x_rec_video = torch.rot90(x_rec_video, k=2, dims=(-2, -1))

out_dir = results_root / data_folder
out_dir.mkdir(parents=True, exist_ok=True)
video_path = out_dir / f'video_rec.mp4'

total_time_acq = acquisition_parameters.total_callback_acquisition_time_s
fps = int(x_rec_video.shape[time_dim] / total_time_acq)

if save_fig:
    save_motion_video(x_rec_video, video_path, amp_max, fps=fps)



# %% save as pdfs.
if save_fig:
    path_fig = results_root / data_folder
    Path(path_fig).mkdir(parents=True, exist_ok=True)

    # (a) CMOS
    plt.imsave(path_fig / f'fig12_a_cmos_amp{amp_max}_frame{frame_ref}.pdf', 
               f_cmos_ref, cmap='gray')

    # (b) g0
    plt.imsave(path_fig / f'fig12_b_g0_amp{amp_max}_frame{frame_ref}.pdf',
               f_white_np, cmap='gray')

    # (c) wf g=1
    plt.imsave(path_fig / f'fig12_c_wf_g=1_amp{amp_max}_frame{frame_ref}.pdf',
               f_wf_Xext_np_no_gain, cmap='gray')

    # (d) wf g=g0
    plt.imsave(path_fig / f'fig12_d_wf_g=g0_amp{amp_max}_frame{frame_ref}.pdf', 
               f_wf_Xext_np, cmap='gray')




# %%
