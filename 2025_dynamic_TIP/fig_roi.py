# %% Import bib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pathlib import Path

from spyrit.misc.disp import torch2numpy
import spyrit.core.torch as spytorch
from spyrit.misc.statistics import Cov2Var
from spyrit.core.meas import HadamSplit2d, DynamicHadamSplit2d
from spyrit.core.prep import Unsplit

from spyrit.core.dual_arm import ComputeHomography, recalibrate, MotionFieldProjector
from spyrit.misc.load_data import read_acquisition, download_girder
import matplotlib as mpl



# %% DETERMINE HOMOGRAPHY
save_fig = False

homo_folder = Path('homography/')
data_root = Path('../data/data_online/extended_FOV2')

results_root = Path('../../Images/images_thèse/2024_article/exp_results/')

# dtype = torch.float64
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

data_folder = Path('obj_no_mtion_starSector_DoF-811_source_white_LED_f80mm-P2_Walsh_im_64x64_ti_1ms_zoom_x1')
data_file_prefix = 'obj_no_mtion_starSector_DoF-811_source_white_LED_f80mm-P2_Walsh_im_64x64_ti_1ms_zoom_x1'

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


# %%
pos_x, pos_y = 64 - 10, 64 - 10
# ROI half-size in pixels (will produce a square ROI of size (2*ROI_HALF+1)^2)
ROI_HALF = 3

fs = 18

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(x_stat.mean(axis=0).cpu().numpy(), cmap='gray')
# draw ROI rectangle
h_img = x_stat.shape[1]
w_img = x_stat.shape[2]
x0 = int(max(0, pos_x - ROI_HALF))
y0 = int(max(0, pos_y - ROI_HALF))
x1 = int(min(w_img, pos_x + ROI_HALF + 1))
y1 = int(min(h_img, pos_y + ROI_HALF + 1))
width = x1 - x0
height = y1 - y0
rect = Rectangle((x0, y0), width, height, linewidth=2, edgecolor='cyan', facecolor='none')
ax[0].add_patch(rect)
ax[0].set_title("Pan rec", fontsize=20)

# mean spectrum over ROI
mean_spec = x_stat[:, y0 : y0 + height, x0 : x0 + width].mean(dim=(1, 2)).cpu().numpy()
ax[1].plot(wav, mean_spec)
ax[1].set_title("ROI spectrum", fontsize=20)
ax[1].set_xlabel("Wavelength (nm)", fontsize=fs)
ax[1].set_ylabel("Intensity", fontsize=fs)
plt.show()

# %% DYNAMIC ACQUISITION
n = 64
n_acq = 64

N = n ** 2
M = N

data_folder = Path('obj_motion_Diag-UL-BR_starSector_DoF-811_source_white_LED_f80mm-P2_Walsh_im_64x64_ti_1.5ms_zoom_x1')
data_file_prefix = 'obj_motion_Diag-UL-BR_starSector_DoF-811_source_white_LED_f80mm-P2_Walsh_im_64x64_ti_1.5ms_zoom_x1'

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


# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# draw ROI rectangle for this plot
h_img = f_stat.shape[1]
w_img = f_stat.shape[2]
x0 = int(max(0, pos_x - ROI_HALF))
y0 = int(max(0, pos_y - ROI_HALF))
x1 = int(min(w_img, pos_x + ROI_HALF + 1))
y1 = int(min(h_img, pos_y + ROI_HALF + 1))
width = x1 - x0
height = y1 - y0
rect = Rectangle((x0, y0), width, height, linewidth=2, edgecolor='cyan', facecolor='none')

ax[0].imshow(f_stat.mean(axis=0).cpu().numpy(), cmap='gray')
ax[0].add_patch(rect)
ax[0].set_title("Pan rec", fontsize=20)

# mean spectrum over ROI
mean_spec = f_stat[:, y0 : y0 + height, x0 : x0 + width].mean(dim=(1, 2)).cpu().numpy()
ax[1].plot(wav, mean_spec)
ax[1].set_title("ROI spectrum", fontsize=20)
ax[1].set_xlabel("Wavelength (nm)", fontsize=fs)
ax[1].set_ylabel("Intensity", fontsize=fs)
plt.show()


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


# %% stat acq flat field
x_stat_ff = x_stat / w.squeeze(0)

x_stat_ff_np = torch2numpy(torch.rot90(x_stat_ff, k=2, dims=(-2, -1)))

pos_x, pos_y = 64 - 10, 64 - 10

fs = 18

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# draw ROI rectangle for this plot
h_img = x_stat_ff.shape[1]
w_img = x_stat_ff.shape[2]
x0 = int(max(0, pos_x - ROI_HALF))
y0 = int(max(0, pos_y - ROI_HALF))
x1 = int(min(w_img, pos_x + ROI_HALF + 1))
y1 = int(min(h_img, pos_y + ROI_HALF + 1))
width = x1 - x0
height = y1 - y0

ax[0].imshow(x_stat_ff_np.mean(axis=0), cmap='gray', vmin=0, vmax=1)
rect = Rectangle((x0, y0), width, height, linewidth=2, edgecolor='cyan', facecolor='none')
ax[0].add_patch(rect)
ax[0].set_title("Pan rec", fontsize=20)

# mean spectrum over ROI (flat-field corrected)
mean_spec = x_stat_ff[:, y0 : y0 + height, x0 : x0 + width].mean(dim=(1, 2)).cpu().numpy()
ax[1].plot(wav, mean_spec)
ax[1].set_title("Spectrums", fontsize=20)
ax[1].set_xlabel("Wavelength (nm)", fontsize=fs)
ax[1].set_ylabel("Intensity", fontsize=fs)
plt.show()


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
                        dtype=dtype, device=torch.device('cpu'))  # cpu to avoid cuda OOM



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
# old : torch.rot90(w, k=2, dims=(-2, -1))

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
f_wf_Xext = torch.linalg.solve(H_dyn.T @ H_dyn + eta * norm ** 2 * D2, H_dyn.T @ y2_exp.T)


# %%
f_wf_Xext_np = np.rot90(torch2numpy(f_wf_Xext.view((l, l, n_wav)).moveaxis(-1, 0)), 2, axes=(1, 2))
f_wf_Xext_np /= t_dyn


# dynamic case: ROI centered at original coordinates shifted by amp_max
pos_x, pos_y = 10, 10
pos_x, pos_y = pos_x + amp_max, pos_y + amp_max
# recompute ROI coords (clamped to image)
h_img = f_wf_Xext_np.shape[1]
w_img = f_wf_Xext_np.shape[2]
x0 = int(max(0, pos_x - ROI_HALF))
y0 = int(max(0, pos_y - ROI_HALF))
x1 = int(min(w_img, pos_x + ROI_HALF + 1))
y1 = int(min(h_img, pos_y + ROI_HALF + 1))
width = x1 - x0
height = y1 - y0

fs = 18

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(f_wf_Xext_np.mean(axis=0), cmap='gray')
# draw ROI rectangle for this plot
rect = Rectangle((x0, y0), width, height, linewidth=2, edgecolor='cyan', facecolor='none')
ax[0].add_patch(rect)
ax[0].set_title("Pan rec", fontsize=20)

mean_spec = f_wf_Xext_np[:, y0 : y0 + height, x0 : x0 + width].mean(axis=(1, 2))
ax[1].plot(wav, mean_spec)
ax[1].set_title("ROI spectrum", fontsize=20)
ax[1].set_xlabel("Wavelength (nm)", fontsize=fs)
ax[1].set_ylabel("Intensity", fontsize=fs)
plt.show()




# %% compare static and dynamic reconstructions for the ROI
x_stat_ff_np_wide = np.zeros((n_wav, l, l))
x_stat_ff_np_wide[:, amp_max : amp_max + n, amp_max : amp_max + n] = x_stat_ff_np

# draw 1st ROI rectangle
pos_x, pos_y = 49, 52
ROI_HALF = 24

h_img = x_stat_ff_np_wide.shape[1]
w_img = x_stat_ff_np_wide.shape[2]
    
x0 = int(max(0, pos_x - ROI_HALF))
y0 = int(max(0, pos_y - ROI_HALF))
x1 = int(min(w_img, pos_x + ROI_HALF + 1))
y1 = int(min(h_img, pos_y + ROI_HALF + 1))
width = x1 - x0
height = y1 - y0

# draw 2nd ROI
pos_x, pos_y = 29, 47
ROI_HALF = 24

h_img = f_wf_Xext_np.shape[1]
w_img = f_wf_Xext_np.shape[2]
a0 = int(max(0, pos_x - ROI_HALF))
b0 = int(max(0, pos_y - ROI_HALF))
a1 = int(min(w_img, pos_x + ROI_HALF + 1))
b1 = int(min(h_img, pos_y + ROI_HALF + 1))
w0 = a1 - a0
h0 = b1 - b0


fig, ax = plt.subplots(1, 3, figsize=(15, 5))


ax[0].imshow(x_stat_ff_np_wide.mean(axis=0), cmap='gray', vmin=0, vmax=1)
rect_stat = Rectangle((x0, y0), width, height, linewidth=3, edgecolor='green', facecolor='none')
ax[0].add_patch(rect_stat)
rect_fov = Rectangle((amp_max, amp_max), n, n, linewidth=3, edgecolor='blue', facecolor='none')
ax[0].add_patch(rect_fov)
ax[0].set_title(f"Stat rec ff / {t_stat} ms", fontsize=20)


ax[1].imshow(f_wf_Xext_np.mean(axis=0), cmap='gray', vmin=0, vmax=1)
rect_dyn = Rectangle((a0, b0), w0, h0, linewidth=2, edgecolor='green', facecolor='none')
ax[1].add_patch(rect_dyn)
rect_fov = Rectangle((amp_max, amp_max), n, n, linewidth=2, edgecolor='blue', facecolor='none')
ax[1].add_patch(rect_fov)
ax[1].set_title(f"Dyn rec ff / {t_dyn} ms", fontsize=20)

mean_spec_stat = x_stat_ff_np_wide[:, y0 : y0 + height, x0 : x0 + width].mean(axis=(1, 2))
mean_spec_dyn = f_wf_Xext_np[:, b0 : b0 + h0, a0 : a0 + w0].mean(axis=(1, 2))

ax[2].plot(wav, mean_spec_stat, label='Static', color='orange', linestyle=(0, (1, 3)), lw=3)

ax[2].plot(wav, mean_spec_dyn, label='Dynamic', color='red', linestyle=(0, (1, 2)), lw=3)

ax[2].set_title("ROI spectrums", fontsize=20)
ax[2].set_xlabel("Wavelength (nm)", fontsize=fs)
plt.legend()
plt.show()


# %%
if save_fig:
    path_fig = results_root / 'fig_roi'
    path_fig.mkdir(parents=True, exist_ok=True)

    # derive per-axis figure size (split original fig width by number of axes)
    n_axes = len(fig.axes)
    total_w, total_h = fig.get_size_inches()
    single_size = (total_w / n_axes, total_h)

    for i, ax_src in enumerate(fig.axes, start=1):
        fig_i, ax_i = plt.subplots(figsize=single_size)

        # if the axis contains an image, copy it with no interpolation and re-add patches
        if ax_src.get_images():
            im = ax_src.get_images()[0]
            arr = im.get_array()
            cmap = im.get_cmap()
            vmin, vmax = im.get_clim()
            ax_i.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')

            # Copy Rectangle patches (ROI boxes) so overlays are preserved in the saved image
            for p in ax_src.patches:
                # Only handle Rectangle-like patches
                try:
                    xy = p.get_xy()
                    w_p = p.get_width()
                    h_p = p.get_height()
                except Exception:
                    # skip non-rectangular patches
                    continue

                # Extract common visual properties if available
                lw = getattr(p, 'get_linewidth', lambda: 1)()
                edge = getattr(p, 'get_edgecolor', lambda: 'cyan')()
                face = getattr(p, 'get_facecolor', lambda: 'none')()
                ls = getattr(p, 'get_linestyle', lambda: 'solid')()
                alpha = getattr(p, 'get_alpha', lambda: None)()

                try:
                    new_rect = Rectangle(xy, w_p, h_p, linewidth=lw,
                                         edgecolor=edge, facecolor=face,
                                         linestyle=ls, alpha=alpha)
                    ax_i.add_patch(new_rect)
                except Exception:
                    # if construction fails, skip this patch
                    continue

            ax_i.set_axis_off()  # match typical image display
        else:
            # copy line artists (e.g. spectrum plot)
            for line in ax_src.get_lines():
                ax_i.plot(line.get_xdata(), line.get_ydata(),
                          color=line.get_color(), linestyle=line.get_linestyle(),
                          linewidth=line.get_linewidth(), label=line.get_label())
            ax_i.set_xlabel(ax_src.get_xlabel())
            ax_i.set_ylabel(ax_src.get_ylabel())
            if ax_src.get_legend() is not None:
                ax_i.legend()

        out_path = path_fig / f'fig_roi_subplot_{i}.pdf'
        fig_i.savefig(out_path, bbox_inches='tight')
        plt.close(fig_i)

# %% give errors
mse_roi = np.mean((mean_spec_stat - mean_spec_dyn) ** 2)
print(f'RMSE between static and dynamic ROI spectrums: {mse_roi:.2e}')

mse_ref = np.mean(mean_spec_stat ** 2)
rel_mse_roi = mse_roi / mse_ref
print(f'Relative RMSE between static and dynamic ROI spectrums: {rel_mse_roi:.4%}')

# %% check on single pixels

x_stat_ff_np_wide = np.zeros((n_wav, l, l))
x_stat_ff_np_wide[:, amp_max : amp_max + n, amp_max : amp_max + n] = x_stat_ff_np

# points on the static
p1_x, p1_y = 73, 40

# points on the dynamic
p2_x, p2_y = 55, 40


fig, ax = plt.subplots(1, 3, figsize=(15, 5))


ax[0].imshow(x_stat_ff_np_wide.mean(axis=0), cmap='gray', vmin=0, vmax=1)
ax[0].plot(p1_x, p1_y, marker='o', markersize=3, color='orange')
rect_fov = Rectangle((amp_max, amp_max), n, n, linewidth=3, edgecolor='blue', facecolor='none')
ax[0].add_patch(rect_fov)
ax[0].set_title(f"Stat rec ff / {t_stat} ms", fontsize=20)


ax[1].imshow(f_wf_Xext_np.mean(axis=0), cmap='gray', vmin=0, vmax=1)
ax[1].plot(p2_x, p2_y, marker='o', markersize=3, color='red')
rect_fov = Rectangle((amp_max, amp_max), n, n, linewidth=2, edgecolor='blue', facecolor='none')
ax[1].add_patch(rect_fov)
ax[1].set_title(f"Dyn rec ff / {t_dyn} ms", fontsize=20)

spec_stat = x_stat_ff_np_wide[:, p1_y, p1_x]
spec_dyn = f_wf_Xext_np[:, p2_y, p2_x]

ax[2].plot(wav, spec_stat, label='Static', color='orange', linestyle=(0, (1, 3)), lw=3)

ax[2].plot(wav, spec_dyn, label='Dynamic', color='red', linestyle=(0, (1, 2)), lw=3)

ax[2].set_title("Spectrums", fontsize=20)
ax[2].set_xlabel("Wavelength (nm)", fontsize=fs)
plt.legend()
plt.show()