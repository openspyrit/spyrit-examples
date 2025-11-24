# %% Import bib
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from spyrit.misc.disp import torch2numpy
import spyrit.core.torch as spytorch
from spyrit.misc.statistics import Cov2Var
from spyrit.core.meas import HadamSplit2d, DynamicHadamSplit2d
from spyrit.core.prep import Unsplit
from spyrit.misc.color import plot_hs

from spyrit.core.dual_arm import ComputeHomography, recalibrate, MotionFieldProjector
from spyrit.misc.load_data import read_acquisition
from spyrit.misc.disp import blue_box, get_frame, save_motion_video, save_field_video




# %% DETERMINE HOMOGRAPHY
save_fig = False

homo_folder = Path('homography/')
data_root = Path('../data/data_online/extended_FOV2')

results_root = Path('../../Images/images_thèse/2024_article/exp_results/')

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n = 64

n_acq = 64
data_folder = Path('obj_no_motion_cat_DoF-811_source_white_LED_f80mm-P2_Walsh_im_64x64_ti_1ms_zoom_x1')
data_file_prefix = 'obj_no_motion_cat_DoF-811_source_white_LED_f80mm-P2_Walsh_im_64x64_ti_1ms_zoom_x1'

read_homography = True
save_homography = False

snapshot = True  # with new acquisitions

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

data_folder = Path('obj_motion_Diag-UL-BR_starSector_DoF-811_source_white_LED_f80mm-P2_Walsh_im_64x64_ti_2ms_zoom_x2')
data_file_prefix = 'obj_motion_Diag-UL-BR_starSector_DoF-811_source_white_LED_f80mm-P2_Walsh_im_64x64_ti_2ms_zoom_x2'

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

n_wav = 8  # number of spectral channels to reconstruct
strategy = 'slice'  # 'slice' or 'bin'

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

w_beg, w_end = 0, 4
n_wav = w_end - w_beg
y1_exp = y1_exp[w_beg:w_end]  # drop some wavelengths 

## EXP ORDER
stat_folder_acq = Path('./stats/')
cov_acq_file = stat_folder_acq / ('Cov_{}x{}'.format(n_acq, n_acq) + '.npy')

Cov_acq = np.load(cov_acq_file)
Ord_acq = Cov2Var(Cov_acq)

Ord = torch.from_numpy(Ord_acq)


# %% PREP OP
prep_op = Unsplit()
y2_exp = prep_op(y1_exp)


# %% STATIC RECO
meas_op_stat = HadamSplit2d(M=M, h=n, order=Ord, dtype=dtype, device=device) 

f_stat = meas_op_stat.fast_pinv(y2_exp)


# %% Plot STATIC RECO (spectral): Fig 10.a
if n_wav > 1:
    f_stat_np = torch2numpy(f_stat.moveaxis(0, -1))
    f_stat_np = np.rot90(f_stat_np, 2)

    plot_hs(strategy, f_stat_np, wav, suptitle="fig_10_a", save_fig=save_fig, 
            results_root=results_root, data_folder=data_folder, colorbar_format='%.1f')


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

deform_folder = Path('star_diag_x2')
deform_prefix = 'star'




# %% ESTIM MOTION FROM CMOS CAMERA
amp_max = 28  #20
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
eta_in_X = 1e-1


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


# %% ######### PATTERN WARPING DYNAMIC MEASUREMENT MATRIX #########
warping = 'pattern'
def_field = projector(warping=warping, amp_max=amp_max).to(device)
meas_op.build_dynamic_forward(def_field, warping=warping, mode=forme_interp)

H_dyn = meas_op.H_dyn
H_dyn_in_X = H_dyn.reshape((H_dyn.shape[0], l, l))[:, amp_max:-amp_max, amp_max:-amp_max].reshape((H_dyn.shape[0], n**2))

# %% send to cpu for linalg operations
H_dyn, H_dyn_in_X = H_dyn.to('cpu'), H_dyn_in_X.to('cpu') # send to cpu for linalg operations

s = torch.linalg.svdvals(H_dyn)
print(f'H_dyn spectrum is [{s[-1]:.2f}, {s[0]:.2f}] (condition number: {s[0]/s[-1]:.2e})')
norm = s[0]

s = torch.linalg.svdvals(H_dyn_in_X)
print(f'H_dyn_in_X spectrum is [{s[-1]:.2f}, {s[0]:.2f}] (condition number: {s[0]/s[-1]:.2e})')
norm_in_X = s[0]

# %% RECO WITH LAPACK (LU factorization)
f_wh_Xext = torch.linalg.solve(H_dyn.T @ H_dyn + eta * norm ** 2 * D2, H_dyn.T @ y2_exp.T)
f_wh_X = torch.linalg.solve(H_dyn_in_X.T @ H_dyn_in_X + eta_in_X * norm_in_X ** 2 * D2_in_X, H_dyn_in_X.T @ y2_exp.T)


# %% ######### IMAGE WARPING DYNAMIC MEASUREMENT MATRIX #########
warping = 'image'
def_field = projector(warping=warping, amp_max=amp_max).to(device)
meas_op.build_dynamic_forward(def_field, warping=warping, mode=forme_interp)

H_dyn = meas_op.H_dyn
H_dyn_in_X = H_dyn.reshape((H_dyn.shape[0], l, l))[:, amp_max:-amp_max, amp_max:-amp_max].reshape((H_dyn.shape[0], n**2))


# %% send to cpu for linalg operations
H_dyn, H_dyn_in_X = H_dyn.to('cpu'), H_dyn_in_X.to('cpu') # send to cpu for linalg operations

s = torch.linalg.svdvals(H_dyn)
print(f'H_dyn spectrum is [{s[-1]:.2f}, {s[0]:.2f}] (condition number: {s[0]/s[-1]:.2e})')
norm = s[0]

s = torch.linalg.svdvals(H_dyn_in_X)
print(f'H_dyn_in_X spectrum is [{s[-1]:.2f}, {s[0]:.2f}] (condition number: {s[0]/s[-1]:.2e})')
norm_in_X = s[0]


# %% RECO WITH LAPACK (LU factorization)
f_wf_Xext = torch.linalg.solve(H_dyn.T @ H_dyn + eta * norm ** 2 * D2, H_dyn.T @ y2_exp.T)
f_wf_X = torch.linalg.solve(H_dyn_in_X.T @ H_dyn_in_X + eta_in_X * norm_in_X ** 2 * D2_in_X, H_dyn_in_X.T @ y2_exp.T)


# %% Fig. 09
f_wide_stat = np.pad(np.rot90(f_stat.mean(axis=0).cpu().numpy(), 2), ((amp_max, amp_max), (amp_max, amp_max)))
f_wh_X_np = np.pad(np.rot90(torch2numpy(f_wh_X.mean(axis=1)).reshape((n, n)), 2), ((amp_max, amp_max), (amp_max, amp_max)))
f_wf_X_np = np.pad(np.rot90(torch2numpy(f_wf_X.mean(axis=1)).reshape((n, n)), 2), ((amp_max, amp_max), (amp_max, amp_max)))

f_cmos_ref = np.rot90(torch2numpy(img_cmos_calibrated.squeeze()), 2)
f_wh_Xext_np = np.rot90(torch2numpy(f_wh_Xext.mean(axis=1)).reshape((l, l)), 2)
f_wf_Xext_np = np.rot90(torch2numpy(f_wf_Xext.mean(axis=1)).reshape((l, l)), 2)


fs = 30

fig, ax = plt.subplots(2, 3,  figsize=(15, 10))

ax[0, 0].imshow(blue_box(f_wide_stat, amp_max=amp_max), cmap='gray')
ax[0, 0].set_title('(a) Static', fontsize=fs)
ax[0, 0].axis('off')

ax[0, 1].imshow(blue_box(f_wh_X_np, amp_max=amp_max))
ax[0, 1].set_title('(b) wh in X', fontsize=fs)
ax[0, 1].axis('off')

ax[0, 2].imshow(blue_box(f_wf_X_np, amp_max=amp_max))
ax[0, 2].set_title('(c) wf in X', fontsize=fs)
ax[0, 2].axis('off')

ax[1, 0].imshow(blue_box(f_cmos_ref, amp_max=amp_max), cmap='gray')
ax[1, 0].set_title('(d) CMOS', fontsize=fs)
ax[1, 0].axis('off')

ax[1, 1].imshow(blue_box(f_wh_Xext_np, amp_max=amp_max))
ax[1, 1].set_title('(e) wh in X ext', fontsize=fs)
ax[1, 1].axis('off')

im = ax[1, 2].imshow(blue_box(f_wf_Xext_np, amp_max=amp_max))
ax[1, 2].set_title('(f) wf in X ext', fontsize=fs)
ax[1, 2].axis('off')

plt.tight_layout()
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


# %% Save deformation fields
if save_fig:
    path_fig = results_root / data_folder
    Path(path_fig).mkdir(parents=True, exist_ok=True)
    video_path = path_fig / 'deformation_quiver.mp4'

    n_frames = 200
    step = 6  # subsampling for arrows
    fps = 30

    save_field_video(def_field, video_path, n_frames=n_frames, step=step, fps=fps, figsize=(6, 6), dpi=200, scale=1, fs=16,
                     amp_max=amp_max, box_color='blue', box_linewidth=2)


# %% Spectral plots (change f_dyn if needed) : Fig 10.b
f_dyn = f_wf_Xext  # change if needed

if n_wav > 1:
    f_dyn_np = np.rot90(torch2numpy(f_dyn.reshape((l, l, n_wav))), 2)

    plot_hs(strategy, f_dyn_np, wav, suptitle="fig_10_b", save_fig=save_fig, 
            results_root=results_root, data_folder=data_folder, colorbar_format='%.1f')



# %% save as pdfs.
if save_fig:
    path_fig = results_root / data_folder
    Path(path_fig).mkdir(parents=True, exist_ok=True)

    # (a) Static reconstruction (wide)
    plt.imsave(path_fig / f'fig09_a_static_amp{amp_max}_frame{frame_ref}.pdf', 
               blue_box(f_wide_stat, amp_max=amp_max))

    # (b) wh in X
    plt.imsave(path_fig / f'fig09_b_wh_in_X_amp{amp_max}_frame{frame_ref}.pdf',
               blue_box(f_wh_X_np, amp_max=amp_max))

    # (c) wf in X
    plt.imsave(path_fig / f'fig09_c_wf_in_X_amp{amp_max}_frame{frame_ref}.pdf',
               blue_box(f_wf_X_np, amp_max=amp_max))

    # (d) CMOS reference
    plt.imsave(path_fig / f'fig09_d_cmos_amp{amp_max}_frame{frame_ref}.pdf', 
               blue_box(f_cmos_ref, amp_max=amp_max), cmap='gray')

    # (e) wh in X ext
    plt.imsave(path_fig / f'fig09_e_wh_in_X_ext_amp{amp_max}_frame{frame_ref}.pdf',
               blue_box(f_wh_Xext_np, amp_max=amp_max))

    # (f) wf in X ext
    plt.imsave(path_fig / f'fig09_f_wf_in_X_ext_amp{amp_max}_frame{frame_ref}.pdf',
               blue_box(f_wf_Xext_np, amp_max=amp_max))



# %%
