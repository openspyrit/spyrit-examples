#%% import lib
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)
from skimage import data, transform

from recordclass import recordclass

from spyrit.misc.statistics import Cov2Var
from spyrit.core.meas import HadamSplit2d
from spyrit.core.noise import Poisson
from spyrit.core.prep import Unsplit

from spyrit.misc import metrics
from spyrit.misc.disp import tensor2img, hs_to_rgb, plot_spectra_compare
from utils_pan_simu import simulate_hs_data, simulate_spectrum_RGB

from spyrit.core.panshrp import PGD, Pan_PGD_LR
from spyrit.misc.load_data import download_girder


# toolbox doesnt have a toml, I can't install in editable mode... Replace the path below with your local path to the hyperspectral_pansharpening_toolbox repo
import sys
sys.path.insert(
    0,
    "/home/maitre/Documents/pansharpening/third_party/spyrit-examples/2026_tdsi/hyperspectral_pansharpening_toolbox"
)
from hyperspectral_pansharpening_toolbox.CS.Brovey import BT_H 
from hyperspectral_pansharpening_toolbox.MRA.AWLP import AWLP
from hyperspectral_pansharpening_toolbox.ModelBasedOptimization.HySURE import HySURE


# %% config
save_fig = False

n = 64
N = n ** 2
und = 1
# und = 16
M = N // und
n_wav = 64

alpha = 15
# alpha = 50

device = 'cpu'   # /!\ Some methods on the toolbox only run on CPU. I modified it in my local repo, see with Nicolas if I should contribute to the other library?
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

height, width = 256, 256
N_hat = height * width
dtype = torch.float32

homography = torch.tensor([[n / width, 0, 0],
                           [0, n / height, 0],
                           [0, 0, 1]],
                          dtype=dtype,
                          device=device)

results_root = Path.cwd() / Path('resultats')
path_fig = results_root
if save_fig:
    path_fig.mkdir(parents=True, exist_ok=True)

## spectral params
wav_min, wav_max = 420, 650  ## JIONC RESULTS
wavelengths = torch.tensor([wav_min + w * (wav_max - wav_min) / (n_wav - 1) for w in range(n_wav)], dtype=dtype)


# %% reference image and spectral response

img_ref = transform.resize(data.astronaut(), (height, width, 3))  # RGB image in [0, 1], shipped with scikit-image
img_ref = torch.from_numpy(img_ref).to(dtype=dtype, device=device).moveaxis(-1, 0).unsqueeze(0)  # (1, 3, height, width)

spectrum = simulate_spectrum_RGB(wavelengths, CIE=False).to(device)

R = spectrum.sum(axis=0).unsqueeze(1)   # panchromatic response

plt.figure(figsize=(8, 6))
plt.plot(wavelengths, spectrum[0].cpu().numpy(), "o--", color='red', label='Red')
plt.plot(wavelengths, spectrum[1].cpu().numpy(), "o--", color='green', label='Green')
plt.plot(wavelengths, spectrum[2].cpu().numpy(), "o--", color='blue', label='Blue')
plt.legend()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

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

Cov_acq = np.load(local_folder / ('Cov_{}x{}'.format(n, n) + '.npy'))
Ord_acq = Cov2Var(Cov_acq)
Ord = torch.from_numpy(Ord_acq)



# %% simulate the acquisition
HS_ref, HS, P = simulate_hs_data(img_ref, n, homography, R, spectrum)
HS, P = HS.to(device), P.to(device)




torch.manual_seed(100)
noise_op = Poisson(alpha=alpha * n_wav)
meas_op = HadamSplit2d(n, M=M, order=Ord, noise_model=noise_op, dtype=dtype, device=device)
M_HS_split = meas_op(HS)

prep_op = Unsplit()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(tensor2img(P), cmap='gray')
ax[0].set_title('Panchromatic %d x %d' % (height, width))
ax[1].imshow(tensor2img(hs_to_rgb(HS, spectrum)))
ax[1].set_title('RGB %d x %d' % (n, n))
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()


# %% shared pre-reconstruction: coarse SPC cube
# The toolbox methods consume this low-resolution cube, while ours does not. Its
# cost is therefore reported apart rather than folded into any method.
eta_1 = 1 / N
if alpha == 5:
    eta_tv = 4.711e-2 / (und + 2.496e1)
elif und == 1:
    eta_tv = -4.951e-3 * np.exp(-2.353 / alpha) + 5.091e-3
else:
    raise ValueError("eta_tv is only known for alpha = 5 or und = 1")

t0 = time.time()
# spc_op = SPC_Solver(meas_op, spectrum)
# f_HS, _ = spc_op.spc_pgd(M_HS_split, type_reg='tv', eta_1=eta_1, eta_3=eta_tv, eta_4=0, n_iter=50, tau=None, step_show=None)
spc_op = PGD(meas_op, spectrum)
f_HS, _ = spc_op(M_HS_split, type_reg='tv', eta_1=eta_1, eta_3=eta_tv, eta_4=0, n_iter=50, tau=None, step_show=None)
t_spc = time.time() - t0

# Timed apart from the SPC solve: the upsampling alone is what the bicubic
# baseline below costs on top of the shared pre-reconstruction.
t0 = time.time()
ms = torch.nn.functional.interpolate(f_HS, size=P.shape[-2:], mode='bicubic',
                                     align_corners=False)
t_interp = time.time() - t0
print('SPC pre-reconstruction done in %.2f s, bicubic upsampling in %.3f s'
      % (t_spc, t_interp))

# PSF of the decimation, in the layout the toolbox expects
ks_h, ks_w = int(height / n), int(width / n)
psf = torch.ones((ks_h, ks_w), dtype=dtype, device=device) / (ks_h * ks_w)
ks_h_adj, ks_w_adj = ks_h + (ks_h % 2 == 0), ks_w + (ks_w % 2 == 0)
kernel = torch.zeros((ks_h_adj, ks_w_adj), dtype=dtype, device=device)
kernel[ks_h_adj - ks_h:, ks_w_adj - ks_w:] = psf

overlap = torch.arange(n_wav, dtype=dtype, device=device)

exp_info = {
    'ratio': P.shape[-2] // f_HS.shape[-2],
    'ms_lr': f_HS, 'ms': ms, 'pan': P, 'overlap': overlap,
    'name': 'SingleImage', 'root': '', 'img_number': 0, 'sensor': 'CustomSensor',
    # 'spec_response': R.T, 'spat_response': kernel,
    'p': 3
}
exp_input = recordclass('exp_info', exp_info.keys())(*exp_info.values())
# Normalize the metadata passed to third-party methods to the same device and
# floating-point precision. This avoids the double/float mismatch seen inside
# AWLP/HySURE on CUDA.
exp_input.ms_lr = exp_input.ms_lr.to(device=device, dtype=dtype)
exp_input.ms = exp_input.ms.to(device=device, dtype=dtype)
exp_input.pan = exp_input.pan.to(device=device, dtype=dtype)
exp_input.overlap = exp_input.overlap.to(device=device, dtype=dtype)

X_sota = {}    # reconstructions, keyed by method name
t_sota = {}    # wall-clock time of the method alone, in seconds

# Methods that need the SPC pre-reconstruction above before they can run
needs_spc = {'interp', 'cs_bt', 'mra_awlp', 'vo_hysure'}


# %% Baseline: bicubic upsampling of the low-resolution SPC cube
# No fusion at all, the panchromatic image is never used. It is the floor every
# method has to beat, and it shows what the SPC reconstruction alone provides.
X_sota['interp'] = ms.clone().detach()
t_sota['interp'] = t_interp
print('Bicubic baseline done in %.3f s' % t_sota['interp'])


# %% Ours
# compute_loss and X_ref left at their defaults: no loss, no per-iteration
# metrics, so the timing is comparable with the other methods.
rank = 10  # true rank is at most 3, but HYSURE runs on 10 so let's try and match it
fact = 'vca'
type_reg = 'h1'
eta_3 = 5e-2

t0 = time.time()
pan_op = Pan_PGD_LR(meas_op, rank, fact, homography, R=None, spectrum=spectrum, verbose=False)
# X_sota['ours'], _ = pan_op(M_HS_split, P, type_reg=type_reg, eta_3=eta_3, weighted=False,
                        #    step_show=None, n_iter=500)
X_sota['ours'], _ = pan_op(M_HS_split, P, type_reg=type_reg, eta_3=eta_3, weighted=False,
                           step_show=None, n_iter=50)
t_sota['ours'] = time.time() - t0
print('Ours done in %.2f s' % t_sota['ours'])


# %% CS: BT
t0 = time.time()
X_sota['cs_bt'] = BT_H(exp_input)
t_sota['cs_bt'] = time.time() - t0
print('BT done in %.2f s' % t_sota['cs_bt'])


# %% MRA: AWLP
# The toolbox's wavelet filters (Utils/spectral_tools.ndwt2_working) are built
# on the CPU and never moved to the input's device, so AWLP only runs on CPU.
# It also casts to float64 internally, hence the cast back.
awlp_input = recordclass('awlp_info', ['ms', 'pan', 'ratio'])(
    exp_input.ms.cpu(), exp_input.pan.cpu(), exp_input.ratio)
t0 = time.time()
X_sota['mra_awlp'] = AWLP(awlp_input).to(device=device, dtype=dtype)
t_sota['mra_awlp'] = time.time() - t0
print('AWLP done in %.2f s' % t_sota['mra_awlp'])


# %% Variational: HySURE
# The toolbox forces float64 in its solver (Utils/pansharpening_aux_tools.mldivide)
# and builds torch.eye() tensors without a device, so HySURE only runs on CPU
# with float64 inputs.
hysure_input = recordclass('hysure_info', ['ms_lr', 'pan', 'ratio', 'overlap'])(
    exp_input.ms_lr.cpu().double(), exp_input.pan.cpu().double(),
    exp_input.ratio, exp_input.overlap.cpu())
t0 = time.time()
X_sota['vo_hysure'] = HySURE(hysure_input).to(device=device, dtype=dtype)
t_sota['vo_hysure'] = time.time() - t0
print('HySURE done in %.2f s' % t_sota['vo_hysure'])


# %% display order and styling, shared by the table and the figures
plot_order = ['interp', 'cs_bt', 'mra_awlp', 'vo_hysure', 'ours']
methods_plot = ([m for m in plot_order if m in X_sota]
                + [m for m in X_sota if m not in plot_order])

label_color = {
    'interp': ('Bicubic', 'darkorange'),
    'cs_bt': ('BT', 'darkgreen'),
    'mra_awlp': ('AWLP', 'royalblue'),
    'vo_hysure': ('HySURE', 'slateblue'),
    'ours': ('Ours', 'red'),
}
style = lambda m: label_color.get(m, (m.replace('_', ' ').title(), 'gray'))

method_dash = {
    'interp':    ((0, (2, 1.4, 2, 1.4)),   '*'),   # baseline
    'cs_bt':     ((0, (1, 1.4)),           'o'),   # dotted
    'mra_awlp':  ((0, (4, 1.6)),           's'),   # dashed
    'vo_hysure': ((0, (6, 1.4, 1, 1.4)),   '^'),   # dash-dot
    'ours':      ('-',                     None),  # solid, no marker
}
dash = lambda m: method_dash.get(m, ('-', None))


# %% scores of every method
d = n / (height * width) ** 0.5
X2 = HS_ref.squeeze().moveaxis(0, -1)

scores_sota = {}
for method in methods_plot:
    X1 = X_sota[method].squeeze().moveaxis(0, -1).to(dtype=dtype, device=device)
    scores_sota[method] = {
        'cc': float(metrics.CC(X1, X2)),
        'sam (deg)': float(metrics.SAM(X1, X2)) * 180 / np.pi,
        'rmse': float(metrics.RMSE(X1, X2)),
        'psnr': float(metrics.PSNR(X1, X2)),
        'ergas': float(metrics.ERGAS(X1, X2, d)),
        'time (s)': t_sota.get(method, float('nan')),
    }

cols = ['cc', 'sam (deg)', 'rmse', 'psnr', 'ergas', 'time (s)']
print(f"\n{'method':12s}" + ''.join(f'{c:>12s}' for c in cols))
for method in methods_plot:
    s = scores_sota[method]
    print(f"{style(method)[0]:12s}" + ''.join(f'{s[c]:12.4g}' for c in cols))

print(f"\nTimes are the method alone. Add the shared SPC pre-reconstruction "
      f"({t_spc:.2f} s) for: {', '.join(sorted(needs_spc & set(methods_plot)))}.")

if save_fig:
    with open(path_fig / Path(f'scores_sota_alpha_{alpha}.json'), 'w') as f:
        json.dump({'scores': scores_sota, 'spc_pre_reconstruction (s)': t_spc}, f, indent=4)


# %% RGB rendering of every reconstruction
# Explicit layout: 'reference' is the ground truth, the other keys are X_sota
# entries. A key that is not available leaves its cell empty.
grid_layout = [
    ['reference', 'cs_bt', 'mra_awlp'],
    ['interp', 'vo_hysure', 'ours'],
]

rgb_sources = {'reference': HS_ref, **X_sota}

n_rows, n_cols = len(grid_layout), max(len(row) for row in grid_layout)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
axes = np.atleast_2d(axes)

for ax in axes.ravel():
    ax.axis('off')

for i_row, row in enumerate(grid_layout):
    for i_col, key in enumerate(row):
        ax = axes[i_row, i_col]
        if key not in rgb_sources:
            print(f"Warning: '{key}' not available, cell left empty")
            continue

        rgb_plot = np.clip(
            tensor2img(hs_to_rgb(rgb_sources[key].to(dtype=dtype, device=device), spectrum)),
            0, 1)
        ax.imshow(rgb_plot)
        if key == 'reference':
            ax.set_title('Reference')
        else:
            ax.set_title('%s (SAM %.2f deg)' % (style(key)[0], scores_sota[key]['sam (deg)']))

        if save_fig:
            plt.imsave(path_fig / Path(f'x_rec_{key}_alpha_{alpha}.png'), rgb_plot)

plt.tight_layout()
if save_fig:
    plt.savefig(path_fig / Path(f'sota_comparison_alpha_{alpha}.pdf'))
plt.show()


# %% spectra at chosen pixels, all methods compared

# positions in the high-resolution grid, in (x, y) order.
# spec_positions = [(240, 260), (130, 290), (50, 100)]
spec_positions = [(120, 130), (50, 200)]

# Methods left out for the visualization of spectra
spec_exclude = [
    'interp',
]
spec_methods = [m for m in methods_plot if m not in spec_exclude]

for pos_x, pos_y in spec_positions:
    if not (0 <= pos_x < width and 0 <= pos_y < height):
        raise ValueError(f"Position ({pos_x}, {pos_y}) outside the {width}x{height} grid")


spec_curves = {}
for method in spec_methods:
    label, color = style(method)
    ls, marker = dash(method)
    spec_curves[method] = dict(cube=X_sota[method], label=label, color=color,
                               ls=ls, marker=marker)

figs_spec = plot_spectra_compare(
    spec_curves, spec_positions, wavelengths,
    reference=HS_ref, spectrum=spectrum,
    ref_label='Reference', title_fmt='Position (%s)',
    legend_outside=False, legend_loc='best',
    show_residuals=False, same_y_scale=True, win_size=80,
    save_fig=save_fig, path_fig=path_fig)
plt.show()


# %%
