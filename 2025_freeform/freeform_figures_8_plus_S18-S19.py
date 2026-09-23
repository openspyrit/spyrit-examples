# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 2026

The script generates the figure 8 of the paper and Supplementary Figures 
S18, S17.

N is the mask size (number of ROI pixels), from the freeform N-scan
dataset acquired on 2026-09-11: 256, 512, 1024, 2048, 4096, 8192 and the
full 128x128 frame (N=16384). ROI labels ('ROI_N_<N>') and their N are
read back from the moy_ROI_N_*.npy files written by
freefom_figures_S11-S17.py, so this script does not need to hardcode
the list of N values and keeps working if more of them are added later.
moy/MSNR/sigma_dark are read per ROI from that same folder.

@author: mahieu
"""
#%%
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def tex_escape(text):
    """Escape LaTeX-special characters so `text` (e.g. an ROI label such as
    'ROI_N_4096') can be dropped into a plot title even when text.usetex
    is enabled below -- raw '_'/'&'/etc. otherwise make LaTeX fail (e.g.
    '&' is a misplaced alignment tab outside a tabular)."""
    conv = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
            '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'}
    return ''.join(conv.get(c, c) for c in text)


def discover_masks_cfg(cat_result_folder):
    """Every ROI this campaign produced results for, read back from the
    moy_ROI_N_*.npy files saved by freefom_figures_S11-S17.py --
    avoids hardcoding the list of N values / labels here."""
    moy_files = sorted(cat_result_folder.glob('moy_ROI_N_*.npy'))
    cfg = []
    for f in moy_files:
        label = f.stem[len('moy_'):]
        N = int(re.search(r'ROI_N_(\d+)$', label).group(1))
        cfg.append(dict(label=label, N=N))
    return sorted(cfg, key=lambda c: c['N'], reverse=True)


# Values computed and saved by freefom_figures_S11-S17.py
cat_result_folder = Path('figures/figures_S11-S17')

fig_folder = cat_result_folder
save_tag = True
ext = 'pdf'

masks_cfg = discover_masks_cfg(cat_result_folder)

# True  -- use each scan mode's own measured sigma_dark[ti, lambda] (the
#          2026-09-07 fix): one figure per ROI, 2 subplots (581nm/726nm),
#          each showing both channel counts.
# False -- legacy behaviour: a single, ti/lambda-independent sigma_const
#          for every curve; one figure per ROI overlaying all 4 bands as
#          4 marker symbols on a single set of curves (pre-fix behaviour).
USE_TI_DEPENDENT_SIGMA_DARK = False
sigma_const = 17.0  # used only when USE_TI_DEPENDENT_SIGMA_DARK is False

# True  -- use each scan mode's own measured moy[method, band] to place
#          that method's marker (2026-09-07 fix): moy differs substantially
#          by scan mode -- confirmed against the 2025-09-25 StarSector
#          dataset, RS's raw signal is only ~1.3% above the black/dark
#          level in BOTH campaigns (vs ~40-50x for S1), so RS's
#          reconstructed mean is far more sensitive to stray-light-fit
#          residuals than FH2/MH2/H1/S1's -- it is not interchangeable
#          with S1's moy.
# False -- legacy behaviour: reuse S1's own moy for every method's marker,
#          assuming (incorrectly, for RS) that all 5 scan modes see the
#          same mean flux.
USE_PER_METHOD_MEAN = False

#%% mean squared error
def mse_raster(N, f_mean=100.0, sigma=17.0, tot=1.0, gamma=1.0):

    fref  = sigma**2 / gamma**2 / tot
    alpha = N
    beta  = N**2

    return (alpha*f_mean + beta*fref)/tot

def mse_hadam_diff(N, M=None, f_mean=100.0, sigma=17.0, tot=1.0, gamma=1.0):

    # default
    if M is None: M=N

    fref  = sigma**2 / gamma**2 / tot
    alpha = 2*N
    beta  = 8*M

    return (alpha*f_mean + beta*fref)/tot

def mse_smatrix(N, M=None, f_mean=100.0, sigma=17.0, tot=1.0, gamma=1.0):

    # default
    if M is None: M=N

    fref  = sigma**2 / gamma**2 / tot
    alpha = 2*N
    beta  = 4*M

    return (alpha*f_mean + beta*fref)/tot

def find_closest_abscissa(y, x, y_target):

    # all arguments are 1d numpy arrays
    return x[np.argmin((y[:,np.newaxis]-y_target[np.newaxis,:])**2, axis=0)]

#%% Theoretical model parameters (ROI-independent)
ti = 2           # base integration time (ms), FH2/MH2's reference (see
                  # freefom_figures_S10-S16.py)
alpha = 1e7      # intensity in photons/pixel/s or None
gamma = 1.00     #
P = 128**2

# `tot`, the acquisition-time normalisation used inside the MSE formulas
# above, is set to the real, shared acquisition-time budget of the
# experiment (32768 patterns * ti, in seconds -- see `norm` in
# freefom_figures_S10-S16.py) rather than an arbitrary 1s: this
# makes fref = sigma_dark**2/gamma**2/tot below exactly match the
# convention already used for `fbar_ref` (the experimental markers), so
# curves and markers share one consistent x-axis and the dimensionless
# threshold lines (x=2, x=N-4) stay valid regardless of which sigma_dark
# is plugged in.
tot = 32.768 * ti

# flux
mm = -2
MM = 10
step = 0.01

f_mean = 10**np.arange(mm,MM + step, step)

lw = 2 # line width
fs = 18

# Columns of sigma_dark_{label}.npy / moy_{label}.npy / MSNR_{label}.npy
# are [581nm/31ch, 581nm/5ch, 726nm/31ch, 726nm/5ch] -- pixel indices 515
# and 1800 in lambda_central_list_dark/nc_list_dark (freefom_figures_
# S10-S16.py) convert to 581nm/726nm via the wavelength metadata
# (wavelength[0][515]=580.97, wavelength[0][1800]=726.16). sigma_dark
# changes markedly with lambda_central but barely with the channel count
# -- so group the 2 subplots by wavelength, and show both channel counts
# (as 2 nearly-superimposed curves/points) within each.
lambda_groups = [(0, 1), (2, 3)]   # band indices per subplot
lambda_labels = ['581 nm', '726 nm']
nc_labels = ['31 ch', '5 ch']
nc_styles = ['-', '--']
nc_symbols = ['o', 's']

# method_list order used by moy/MSNR/sigma_dark rows (see
# freefom_figures_S10-S16.py): FH2, RS, MH2, H1, S1 (S1 = reference)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
method_labels = ['FH2', 'RS', 'MH2', 'H1']  # colors[0..3], S1 is the reference (gain=0)

#%% MAIN LOOP over every ROI (ROI_N_256 ... ROI_N_16384)
gain_all = {}  # label -> dict(N=N, gain=gain matrix), filled below, used by
                # the compilation cell at the very end of this script.
msnr_all = {}  # label -> dict(N=N, msnr=msnr matrix, 5 methods incl. S1),
                # filled below, used by the MSNR-vs-N compilation cell.
mse_all = {}   # label -> dict(N=N, mse=std**2 matrix, 5 methods incl. S1),
                # filled below, used by the MSE(from std)-vs-N compilation cell.
mse_theory_all = {}  # label -> dict(N=N, mse=theoretical MSE matrix, 5
                # methods incl. S1), filled below from the paper's closed-
                # form MSE (Eq. 25/27/29/30/36) evaluated at this ROI's own
                # measured f_mean/sigma_dark -- used by the theory-vs-
                # experiment compilation cell at the very end of this script.

for cfg in masks_cfg:

    label = cfg['label']
    label_tex = tex_escape(label)  # display-safe version of label, for plot
                                    # titles only (text.usetex is turned on
                                    # further below) -- keep using `label`
                                    # (raw) for filenames/prints.
    N = cfg['N']
    print(f'=== ROI = {label} (N = {N}) ===')

    # ---- Experimental data for this ROI ----
    # moy/MSNR: [5 methods, 4 bands], rows = FH2, RS, MH2, H1, S1 (same
    # order as method_list in freefom_figures_S10-S16.py); columns:
    # 581nm (31ch), 581nm (5ch), 726nm (31ch), 726nm (5ch)
    moy        = np.load(cat_result_folder / f'moy_{label}.npy')
    msnr       = np.load(cat_result_folder / f'MSNR_{label}.npy')
    std        = np.load(cat_result_folder / f'std_{label}.npy')
    sigma_dark = np.load(cat_result_folder / f'sigma_dark_{label}.npy')
    msnr_all[label] = dict(N=N, msnr=msnr)
    mse_all[label]  = dict(N=N, mse=std**2)

    # gain (dB) of FH2/RS/MH2/H1 relative to S1, per band
    gain = msnr[:4, :] - msnr[4:5, :]
    gain_all[label] = dict(N=N, gain=gain)

    # Mean count in the wide (31-channel) bands, reused for the
    # corresponding narrow band -- same convention as the original,
    # single-ROI script.
    if USE_PER_METHOD_MEAN:
        # Each method's own moy (shape (5 methods, 4 bands)) -- see
        # USE_PER_METHOD_MEAN above.
        fbar = moy[:, [0, 0, 2, 2]] * 1e3 / 31
    else:
        # S1's moy only (shape (4 bands,)), reused for every method --
        # broadcasts against sigma_dark below.
        fbar = np.array([moy[-1, 0], moy[-1, 0], moy[-1, 2], moy[-1, 2]]) * 1e3 / 31
    # Normalised abscissa of each experimental point, using that method's
    # OWN measured sigma_dark[method, band] (real ti-dependent noise
    # floor). Result shape (5 methods, 4 bands) either way.
    fbar_ref = fbar / (sigma_dark**2 / tot)

    # ---- theoretical MSE for this ROI, at its own real measured f_mean and
    # sigma_dark (instead of the f_mean sweep used for the theory curves
    # below) -- lets the paper's closed-form MSE be compared directly
    # against the measured MSE (mse_all) on the same N axis. Same
    # (N, M) choice per scan mode as the sweep below: FH2 and MH2's pattern
    # count M is always P (they always use the full-FOV, 32768-pattern
    # measurement matrix, regardless of this ROI's own N), H1/S1's M is
    # this ROI's own N, and RS's formula only takes N.
    def _fbar(m, ll):
        return fbar[m, ll] if fbar.ndim == 2 else fbar[ll]

    mse_theory = np.empty((5, 4))
    for ll in range(4):
        mse_theory[0, ll] = mse_hadam_diff(P, P, _fbar(0, ll), sigma_dark[0, ll], tot, gamma)  # FH2
        mse_theory[1, ll] = mse_raster(N, _fbar(1, ll), sigma_dark[1, ll], tot, gamma)         # RS
        mse_theory[2, ll] = mse_hadam_diff(N, P, _fbar(2, ll), sigma_dark[2, ll], tot, gamma)  # MH2
        mse_theory[3, ll] = mse_hadam_diff(N, N, _fbar(3, ll), sigma_dark[3, ll], tot, gamma)  # H1
        mse_theory[4, ll] = mse_smatrix(N, N, _fbar(4, ll), sigma_dark[4, ll], tot, gamma)     # S1
    mse_theory_all[label] = dict(N=N, mse=mse_theory)

    # if USE_TI_DEPENDENT_SIGMA_DARK:

    #     # #%% Plot: 2 subplots, one per lambda_central (581nm, 726nm).
    #     # # Within each, both channel counts (31ch, 5ch) are shown -- as 2
    #     # # curves/points per method that land almost on top of each other,
    #     # # since sigma_dark barely depends on nc (only on lambda_central,
    #     # # see module docstring). Colour = method, line style / marker
    #     # # symbol = channel count.
    #     # fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharey=True)

    #     # for sp, (ll_wide, ll_narrow) in enumerate(lambda_groups):
    #     #     ax = axs[sp]
    #     #     ax.axhline(y=0, color='k', linestyle='-')

    #     #     for nc_i, ll in enumerate((ll_wide, ll_narrow)):

    #     #         sigma_fh2 = sigma_dark[0, ll]
    #     #         sigma_rs  = sigma_dark[1, ll]
    #     #         sigma_mh2 = sigma_dark[2, ll]
    #     #         sigma_h1  = sigma_dark[3, ll]
    #     #         sigma_s1  = sigma_dark[4, ll]

    #     #         mse_fh2 = mse_hadam_diff(P, P, f_mean, sigma_fh2, tot, gamma)
    #     #         mse_rs  = mse_raster(N, f_mean, sigma_rs, tot, gamma)
    #     #         mse_mh2 = mse_hadam_diff(N, P, f_mean, sigma_mh2, tot, gamma)
    #     #         mse_h1  = mse_hadam_diff(N, N, f_mean, sigma_h1, tot, gamma)
    #     #         mse_s1  = mse_smatrix(N, N, f_mean, sigma_s1, tot, gamma)

    #     #         boost_fh2 = 10*np.log10(mse_s1 / mse_fh2)
    #     #         boost_rs  = 10*np.log10(mse_s1 / mse_rs)
    #     #         boost_mh2 = 10*np.log10(mse_s1 / mse_mh2)
    #     #         boost_h1  = 10*np.log10(mse_s1 / mse_h1)

    #     #         sigmas = [sigma_fh2, sigma_rs, sigma_mh2, sigma_h1]
    #     #         boosts = [boost_fh2, boost_rs, boost_mh2, boost_h1]

    #     #         for m in range(4):
    #     #             x_curve = f_mean / (sigmas[m]**2 / gamma**2 / tot)
    #     #             ax.semilogx(x_curve, boosts[m], color=colors[m],
    #     #                         linestyle=nc_styles[nc_i], linewidth=lw)
    #     #             ax.semilogx(fbar_ref[m, ll], gain[m, ll], nc_symbols[nc_i],
    #     #                         color=colors[m], markersize=8, markeredgecolor='k')

    #     #     # Dimensionless thresholds (2, N-4): valid for every curve, since
    #     #     # x is defined the same way (f_mean/fref) for all of them
    #     #     # regardless of which sigma_dark went into that particular curve.
    #     #     ax.axvline(x=2, color='k', linestyle='--', linewidth=lw)
    #     #     ax.axvline(x=N-4, color=colors[1], linestyle='--', linewidth=lw)
    #     #     ax.axvspan(1e-2, 2, alpha=.15, color='grey')
    #     #     ax.axvspan(N-4, 1e5, alpha=.15, color=colors[1])

    #     #     ax.set_title(lambda_labels[sp], fontsize=fs)
    #     #     ax.set_xlabel(r'Normalized mean count $\bar{f} / f_{\rm ref}$', fontsize=fs-4)
    #     #     ax.grid(True)
    #     #     ax.set_xlim([1e-2, 1e5])

    #     # axs[0].set_ylabel('MSE gain w.r.t. S1 (in dB)', fontsize=fs)

    #     # # Two-part legend: method (color) and channel count (line style / symbol)
    #     # method_handles = [plt.Line2D([0], [0], color=colors[m], linewidth=lw,
    #     #                               label=method_labels[m]) for m in range(4)]
    #     # nc_handles = [plt.Line2D([0], [0], color='k', linestyle=nc_styles[i],
    #     #                           marker=nc_symbols[i], markerfacecolor='w',
    #     #                           label=nc_labels[i]) for i in range(2)]
    #     # legend1 = axs[0].legend(handles=method_handles, loc='lower right', fontsize=fs-6,
    #     #                          title='Scan mode')
    #     # axs[0].add_artist(legend1)
    #     # axs[0].legend(handles=nc_handles, loc='upper left', fontsize=fs-6, title='Channels')

    #     # # No tex_escape needed: 'Freeform region: N = <N>' has no
    #     # # LaTeX-special characters, so it renders fine even with
    #     # # text.usetex (turned on below).
    #     # fig.suptitle(f'Freeform region: N = {N}', fontsize=fs)
    #     # plt.tight_layout()

    #     # if save_tag:
    #     #     plt.rcParams['text.usetex'] = True
    #     #     fig.savefig(fig_folder/(f'figure_S18_{label}.'+ext), transparent=True, dpi=300)
    # else:
    #     #%% Plot: legacy behaviour -- a single, ti/lambda-independent
    #     # sigma_const for every curve (one set of curves, computed once),
    #     # single figure overlaying all 4 bands as 4 marker symbols per
    #     # method. Markers still use each method's real measured
    #     # sigma_dark (fbar_ref, computed above), so with sigma_const
    #     # picked far from a given band's real sigma_dark, markers and
    #     # curves visibly disagree -- this is the pre-fix behaviour.

tot_legacy = 1.0
fref_legacy = sigma_const**2 / gamma**2 / tot_legacy
f_mean_ref_legacy = f_mean / fref_legacy

mse_fh2 = mse_hadam_diff(P, P, f_mean, sigma_const, tot_legacy, gamma)
mse_rs  = mse_raster(N, f_mean, sigma_const, tot_legacy, gamma)
mse_mh2 = mse_hadam_diff(N, P, f_mean, sigma_const, tot_legacy, gamma)
mse_h1  = mse_hadam_diff(N, N, f_mean, sigma_const, tot_legacy, gamma)
mse_s1  = mse_smatrix(N, N, f_mean, sigma_const, tot_legacy, gamma)

boost_fh2 = 10*np.log10(mse_s1 / mse_fh2)
boost_rs  = 10*np.log10(mse_s1 / mse_rs)
boost_mh2 = 10*np.log10(mse_s1 / mse_mh2)
boost_h1  = 10*np.log10(mse_s1 / mse_h1)

symbol = 'ovd^'

plt.figure()
plt.axhline(y=0, color='k', linestyle='-')
plt.semilogx(f_mean_ref_legacy, boost_fh2, label='FH2', linewidth=lw)
plt.semilogx(f_mean_ref_legacy, boost_rs,  label='RS',  linewidth=lw)
plt.semilogx(f_mean_ref_legacy, boost_mh2, label='MH2', linewidth=lw)
plt.semilogx(f_mean_ref_legacy, boost_h1,  label='H1',  linewidth=lw)

for i in range(gain.shape[1]):
    plt.semilogx(fbar_ref[0, i], gain[0, i], symbol[i], color=colors[0])
    plt.semilogx(fbar_ref[1, i], gain[1, i], symbol[i], color=colors[1])
    plt.semilogx(fbar_ref[2, i], gain[2, i], symbol[i], color=colors[2])
    plt.semilogx(fbar_ref[3, i], gain[3, i], symbol[i], color=colors[3])

# No tex_escape needed: 'Freeform region: N = <N>' has no
# LaTeX-special characters, so it renders fine even with
# text.usetex (turned on below).
plt.title(f'Freeform region: N = {N}', fontsize=fs)
plt.xlabel(r'Normalized mean count $\bar{f} / f_{\rm ref}$', fontsize=fs)
plt.ylabel('MSE gain w.r.t. S1 (in dB)', fontsize=fs)
plt.legend(loc='lower right', fontsize=fs-2)
plt.grid(True)

plt.axvline(x=2, color='k', linestyle='--', linewidth=lw)
plt.axvline(x=N-4, color=colors[1], linestyle='--', linewidth=lw)
plt.xlim([1e-2, 1e5])
plt.axvspan(1e-2, 2, alpha=.15, color='grey')
plt.axvspan(N-4, 1e5, alpha=.15, color=colors[1])
plt.tight_layout()

if save_tag:
    plt.rcParams['text.usetex'] = True
    plt.savefig(fig_folder/(f'figure_S18_{label}.'+ext), transparent=True, dpi=300)

#%% Display the ROI figures
plt.show()

#%% Compile the ROIs -- gain vs N, one figure per wavelength
# Uses gain_all (filled in the main loop above): for each ROI label,
# gain_all[label] = dict(N=N, gain=gain) with gain shape (4 methods, 4 bands),
# columns = [581nm/31ch, 581nm/5ch, 726nm/31ch, 726nm/5ch] (see main loop).
# Per wavelength, average the 31-channel and 5-channel gain (columns 0-1 for
# 581nm, 2-3 for 726nm) and plot that average vs N (number of ROI pixels --
# each N corresponds to one ROI_N_<N> ROI produced by
# freefom_figures_S10-S16.py).
labels_by_N = sorted(gain_all, key=lambda label: gain_all[label]['N'])
N_list = [gain_all[label]['N'] for label in labels_by_N]

#%% Empirical calibration of the theoretical MSE onto the experimental
# MSE's absolute scale. mse_theory_all (paper Eq. 25/27/29/30/36, raw-count
# units) and mse_all (std**2 of the reconstructed image, in whatever
# internal units freefom_figures_S10-S16.py's reconstruction pipeline
# uses) are NOT on the same absolute scale -- checked numerically, raw
# theory MSE came out ~4-5 orders of magnitude above the experimental MSE
# (most likely a normalisation-convention mismatch between the paper's
# idealised pseudo-inverse and spyrit's actual reconstruction operators,
# e.g. HadamSplit2d.fast_pinv/ifwht/ifwalsh_S_torch -- not re-derived
# here). Rather than re-deriving that exactly, mse_calib[method] is fit
# empirically as the geometric mean of mse_exp/mse_theory across every ROI
# (each ROI's ratio is itself first averaged over the 4 bands, to keep the
# very noisy 5-channel bands from dominating), so
# mse_calib[method] * mse_theory_all[label]['mse'][method, band] lands on
# the same scale as mse_all. This is an empirical fit, not a first-
# principles result: k stays close to constant for FH2/H1/S1 (within
# ~15-40% across N) but drifts by ~2x across N for RS/MH2 -- so even after
# calibration, expect the theory/experiment overlay to be good but
# imperfect for those two scan modes.
mse_calib = np.empty(5)
for m in range(5):
    per_N_ratio = [np.exp(np.mean(np.log(mse_all[label]['mse'][m, :] / mse_theory_all[label]['mse'][m, :])))
                   for label in labels_by_N]
    mse_calib[m] = np.exp(np.mean(np.log(per_N_ratio)))

print('MSE theory calibration constants (geometric mean of mse_exp / mse_theory):')
for m, name in enumerate(method_labels + ['S1']):  # row order: FH2, RS, MH2, H1, S1
    print(f'  {name:5s} k = {mse_calib[m]:.4e}')

# fig, axs = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

# for sp, (c0, c1) in enumerate(lambda_groups):
#     ax = axs[sp]
#     ax.axhline(y=0, color='k', linestyle='-')

#     for m in range(4):
#         gain_vs_N = [gain_all[label]['gain'][m, [c0, c1]].mean()
#                      for label in labels_by_N]
#         ax.plot(N_list, gain_vs_N, marker='o', color=colors[m], linewidth=lw,
#                  markersize=8, markeredgecolor='k', label=method_labels[m])

#     ax.set_title(lambda_labels[sp], fontsize=fs)
#     ax.set_xlabel('N (pixel number in the freeform region)', fontsize=fs-4)
#     ax.set_xscale('log', base=2)
#     ax.set_xticks(N_list)
#     ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
#     ax.grid(True)

# axs[0].set_ylabel('MSE gain w.r.t. S1 (in dB)', fontsize=fs)
# axs[0].legend(loc='best', fontsize=fs-6, title='Scan mode')
# fig.suptitle('Gain vs ROI size N (31ch/5ch average per wavelength)', fontsize=fs)
# plt.tight_layout()

# if save_tag:
#     plt.rcParams['text.usetex'] = True
#     fig.savefig(fig_folder/('figure_9c_gain_vs_N.'+ext), transparent=True, dpi=300)

# plt.show()

#%% Compile the ROIs -- raw MSNR vs N, one figure per wavelength (all 5
# scan modes, S1 included, no comparison/subtraction between methods)
# Uses msnr_all (filled in the main loop above): for each ROI label,
# msnr_all[label] = dict(N=N, msnr=msnr) with msnr shape (5 methods, 4 bands),
# rows = FH2, RS, MH2, H1, S1; columns = [581nm/31ch, 581nm/5ch, 726nm/31ch,
# 726nm/5ch] (see main loop). Per wavelength, average the 31-channel and
# 5-channel MSNR (columns 0-1 for 581nm, 2-3 for 726nm) and plot that
# average vs N (number of ROI pixels of each ROI_N_<N>).
method_labels_all = method_labels + ['S1']
colors_all = colors[:4] + [colors[4]]

# fig, axs = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

# for sp, (c0, c1) in enumerate(lambda_groups):
#     ax = axs[sp]

#     for m in range(5):
#         msnr_vs_N = [msnr_all[label]['msnr'][m, [c0, c1]].mean()
#                      for label in labels_by_N]
#         ax.plot(N_list, msnr_vs_N, marker='o', color=colors_all[m], linewidth=lw,
#                  markersize=8, markeredgecolor='k', label=method_labels_all[m])

#     ax.set_title(lambda_labels[sp], fontsize=fs)
#     ax.set_xlabel('N (pixel number in the freeform region)', fontsize=fs-4)
#     ax.set_xscale('log', base=2)
#     ax.set_xticks(N_list)
#     ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
#     ax.grid(True)

# axs[0].set_ylabel('MSNR (dB)', fontsize=fs)
# axs[0].legend(loc='best', fontsize=fs-6, title='Scan mode')
# fig.suptitle('MSNR vs ROI size N (31ch/5ch average per wavelength)', fontsize=fs)
# plt.tight_layout()

# if save_tag:
#     plt.rcParams['text.usetex'] = True
#     fig.savefig(fig_folder/('figure_9d_msnr_vs_N.'+ext), transparent=True, dpi=300)

# plt.show()

# #%% Compile the ROIs -- MSE (from the measured std) vs N, averaged over
# # all 4 channels (both wavelengths, both channel counts), all 5 scan modes
# # incl. S1, no comparison/subtraction between methods. Uses mse_all (filled
# # in the main loop above): for each ROI label, mse_all[label] = dict(N=N,
# # mse=std**2), std loaded from std_{label}.npy (same [5 methods, 4 bands]
# # layout as msnr -- see freefom_figures_S10-S16.py). MSE = std**2 is
# # the empirical noise variance behind MSNR = 20*log10(moy/std), plotted
# # here directly (log scale) instead of the dB ratio to the mean.
# fig = plt.figure(figsize=(7, 6))
# ax = plt.gca()

# for m in range(5):
#     mse_vs_N = [mse_all[label]['mse'][m, :].mean() for label in labels_by_N]
#     ax.plot(N_list, mse_vs_N, marker='o', color=colors_all[m], linewidth=lw,
#              markersize=8, markeredgecolor='k', label=method_labels_all[m])

# ax.set_xlabel('N (ROI pixels)', fontsize=fs-4)
# ax.set_ylabel(r'MSE ($= \sigma^2$)', fontsize=fs)
# ax.set_xscale('log', base=2)
# ax.set_yscale('log')
# ax.set_xticks(N_list)
# ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
# ax.grid(True)
# ax.legend(loc='best', fontsize=fs-6, title='Scan mode')
# ax.set_title('MSE (average over all 4 channels)', fontsize=fs)
# plt.tight_layout()

# if save_tag:
#     plt.rcParams['text.usetex'] = True
#     fig.savefig(fig_folder/('figure_10_mse_vs_N_allchannels.'+ext), transparent=True, dpi=300)

# plt.show()

#%% Compile the ROIs -- MSE (from the measured std) vs N, same data as the
# cell above but WITHOUT averaging over the 4 channels, and restricted to
# only the first band (581nm, 31ch) and the last band (726nm, 5ch): one
# curve per scan mode (colour, same colors_all as above) AND per band
# (line style, band_styles below), so all 5*2 = 10 curves are shown and
# told apart. Uses mse_all (filled in the main loop above): for each ROI
# label, mse_all[label] = dict(N=N, mse=std**2) with mse shape
# (5 methods, 4 bands), columns = [581nm/31ch, 581nm/5ch, 726nm/31ch,
# 726nm/5ch] (see main loop / freefom_figures_S10-S16.py).
band_channel_labels = [f'{lambda_labels[0]} ({nc_labels[0]})',
                        f'{lambda_labels[0]} ({nc_labels[1]})',
                        f'{lambda_labels[1]} ({nc_labels[0]})',
                        f'{lambda_labels[1]} ({nc_labels[1]})']
# One marker per band (wavelength x channel count), per the paper's
# convention: 'o' (581nm/31ch), 'v' (581nm/5ch), 'D' (726nm/31ch),
# '^' (726nm/5ch) -- used by the figure_8 / figure_S19 gain-vs-N plots.
band_markers = ['o', 'v', 'D', '^']
band_styles = ['-', '--', ':', '-.']
band_idx = [0, 3]  # first band (581nm, 31ch) and last band (726nm, 5ch)

# fig = plt.figure(figsize=(8, 6))
# ax = plt.gca()

# for m in range(5):
#     for c in band_idx:
#         mse_vs_N = [mse_all[label]['mse'][m, c] for label in labels_by_N]
#         ax.plot(N_list, mse_vs_N, color=colors_all[m], linestyle=band_styles[c],
#                  marker='o', linewidth=lw, markersize=6, markeredgecolor='k')

# ax.set_xlabel('N (ROI pixels)', fontsize=fs-4)
# ax.set_ylabel(r'MSE ($= \sigma^2$)', fontsize=fs)
# ax.set_xscale('log', base=2)
# ax.set_yscale('log')
# ax.set_xticks(N_list)
# ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
# ax.grid(True)
# ax.set_title('MSE (per channel)', fontsize=fs)

# # Two-part legend: scan mode (colour) and channel (line style) -- same
# # convention as the sigma_dark-dependent figure earlier in this script.
# method_handles = [plt.Line2D([0], [0], color=colors_all[m], linewidth=lw,
#                               label=method_labels_all[m]) for m in range(5)]
# band_handles = [plt.Line2D([0], [0], color='k', linestyle=band_styles[c],
#                             linewidth=lw, label=band_channel_labels[c])
#                  for c in band_idx]
# legend1 = ax.legend(handles=method_handles, loc='upper left', fontsize=fs-6,
#                      title='Scan mode')
# ax.add_artist(legend1)
# # Placed next to (to the right of) the "Scan mode" legend rather than
# # stacked below it -- bbox_to_anchor x is a rough offset in axes-fraction
# # coordinates; nudge it if the two legends overlap or leave a gap.
# ax.legend(handles=band_handles, loc='upper left', bbox_to_anchor=(0.14, 1.0),
#           fontsize=fs-6, title='Channel')

# plt.tight_layout()

# if save_tag:
#     plt.rcParams['text.usetex'] = True
#     fig.savefig(fig_folder/('figure_10_mse_vs_N_perchannel.'+ext), transparent=True, dpi=300)

# plt.show()

#%% Compile the ROIs -- MSE (from the measured std) vs N, same data as the
# two cells above but for a single band only: 726nm, 31ch (band index 2 --
# see the [581nm/31ch, 581nm/5ch, 726nm/31ch, 726nm/5ch] column order in
# the main loop / freefom_figures_S10-S16.py). One curve per scan mode
# (colour, same colors_all as above); no channel/line-style legend is
# needed since there is only one band here.
band_single = 2  # 726nm, 31ch

# fig = plt.figure(figsize=(8, 6))
# ax = plt.gca()

# for m in range(5):
#     mse_vs_N = [mse_all[label]['mse'][m, band_single] for label in labels_by_N]
#     ax.plot(N_list, mse_vs_N, marker='o', color=colors_all[m], linewidth=lw,
#              markersize=8, markeredgecolor='k', label=method_labels_all[m])

# ax.set_xlabel('N (ROI pixels)', fontsize=fs-4)
# ax.set_ylabel(r'MSE ($= \sigma^2$)', fontsize=fs)
# ax.set_xscale('log', base=2)
# ax.set_yscale('log')
# ax.set_xticks(N_list)
# ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
# ax.grid(True)
# ax.legend(loc='upper left', fontsize=fs-6, title='Scan mode')
# ax.set_title(f'MSE ({band_channel_labels[band_single]})', fontsize=fs)
# plt.tight_layout()

# if save_tag:
#     plt.rcParams['text.usetex'] = True
#     fig.savefig(fig_folder/('figure_10_mse_vs_N_726nm_31ch.'+ext), transparent=True, dpi=300)

# plt.show()

#%% Compare theory vs experiment, for every band, as the MSE GAIN (dB)
# relative to FH2 rather than the raw MSE: mse_theory_all and mse_all are
# each in their own internal units (the reconstructed-image scaling baked
# into freefom_figures_S10-S16.py for the experimental side, vs. the
# paper's raw-count Eq. 25/27/29/30/36 for the theoretical side), so their
# absolute values are NOT directly comparable -- checked numerically, raw
# theory MSE came out ~4-5 orders of magnitude above the experimental
# MSE. The GAIN relative to FH2 (10*log10(mse_FH2/mse_method), same
# convention as the paper's Fig. 9 and the f_mean-swept theory curves
# earlier in this script, but taking FH2 -- instead of S1 -- as the
# reference) is a ratio, so that common scale factor cancels out and
# theory/experiment become directly comparable. One figure per band.
# Extend the theory curves below the smallest measured ROI (N=256) down to
# N=1: same synthetic N grid (powers of 2) reused for every band/method
# below.
N_ext = 2 ** np.arange(0, int(np.log2(min(N_list))))  # 1, 2, ..., N_min/2
N_ticks = sorted(set(N_ext.tolist()) | set(N_list))  # synthetic + measured N,
                # for the x-ticks/vertical gridlines below -- same convention
                # as the measured-only N_list ticks used elsewhere

for band_i, file_tag in [(0, '581nm_31ch'), (1, '581nm_5ch'),
                          (2, '726nm_31ch'), (3, '726nm_5ch')]:

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # FH2 (the reference) always uses the full-FOV P,P measurement
    # regardless of ROI size, so its theory MSE is N-independent -- reused
    # as a constant denominator for the extended (N<256) points below.
    mse_fh2_th = mse_hadam_diff(P, P, _fbar(0, band_i), sigma_dark[0, band_i], tot, gamma)

    for m in [1, 2, 3, 4]:  # RS, MH2, H1, S1 -- FH2 (index 0) is the reference
        gain_exp = [msnr_all[label]['msnr'][m, band_i] - msnr_all[label]['msnr'][0, band_i]
                    for label in labels_by_N]
        gain_th = [10 * np.log10(mse_theory_all[label]['mse'][0, band_i]
                                  / mse_theory_all[label]['mse'][m, band_i])
                   for label in labels_by_N]

        # N=1..N_min/2 points: same closed-form MSE, evaluated at this
        # synthetic N grid, reusing the smallest ROI's own measured
        # fbar/sigma_dark (`_fbar`/`sigma_dark`, still bound to the last
        # main-loop iteration -- ROI_N_256, since masks_cfg iterates from
        # the largest to the smallest N). Mean flux and dark noise are
        # physical properties of the acquisition, not of the ROI size, so
        # reusing them below N=256 is a reasonable extrapolation of the
        # curve, not a re-measurement -- experimental markers are NOT
        # extrapolated this way, only the theory (solid) curves are.
        if m == 1:
            mse_m_ext = mse_raster(N_ext, _fbar(1, band_i), sigma_dark[1, band_i], tot, gamma)
        elif m == 2:
            mse_m_ext = mse_hadam_diff(N_ext, P, _fbar(2, band_i), sigma_dark[2, band_i], tot, gamma)
        elif m == 3:
            mse_m_ext = mse_hadam_diff(N_ext, N_ext, _fbar(3, band_i), sigma_dark[3, band_i], tot, gamma)
        else:  # m == 4, S1
            mse_m_ext = mse_smatrix(N_ext, N_ext, _fbar(4, band_i), sigma_dark[4, band_i], tot, gamma)
        gain_th_ext = 10 * np.log10(mse_fh2_th / mse_m_ext)

        N_th_full = np.concatenate([N_ext, N_list])
        gain_th_full = np.concatenate([gain_th_ext, gain_th])

        # Marker shape follows the band (wavelength x channel count) --
        # see band_markers above.
        ax.plot(N_list, gain_exp, marker=band_markers[band_i], linestyle='none',
                 color=colors[m-1], markersize=8, markeredgecolor='k')
        ax.plot(N_th_full, gain_th_full, linestyle='-', color=colors[m-1], linewidth=lw)

    ax.axhline(y=0, color='k', linestyle='-')
    ax.set_xlabel('N (pixel number in the freeform region)', fontsize=fs-4)
    ax.set_ylabel('MSE gain w.r.t. FH2 (in dB)', fontsize=fs)
    ax.set_xscale('log', base=2)
    ax.set_xticks(N_ticks)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True)

    method_handles = [plt.Line2D([0], [0], color=colors[m-1], linewidth=lw,
                                  label=method_labels_all[m]) for m in [1, 2, 3, 4]]
    ax.legend(handles=method_handles, loc='lower left', fontsize=fs-6)

    plt.tight_layout()

    if save_tag:
        plt.rcParams['text.usetex'] = True
        fig.savefig(fig_folder/(f'figure_8_gain_vs_N_{file_tag}_theory_vs_exp_vs_FH2.'+ext), transparent=True, dpi=300)

    plt.show()

#%% Same "Gain vs FH2, theory vs experiment" data as the cell above, but as
# ONE figure with 4 subplots (one per band) instead of 4 separate figures --
# reuses N_ext/N_ticks/mse_fh2_th/gain_exp/gain_th computed the same way.
fig, axs = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)
axs = axs.ravel()

for sp, (band_i, file_tag) in enumerate([(0, '581nm_31ch'), (1, '581nm_5ch'),
                                          (2, '726nm_31ch'), (3, '726nm_5ch')]):
    ax = axs[sp]

    mse_fh2_th = mse_hadam_diff(P, P, _fbar(0, band_i), sigma_dark[0, band_i], tot, gamma)

    for m in [1, 2, 3, 4]:  # RS, MH2, H1, S1 -- FH2 (index 0) is the reference
        gain_exp = [msnr_all[label]['msnr'][m, band_i] - msnr_all[label]['msnr'][0, band_i]
                    for label in labels_by_N]
        gain_th = [10 * np.log10(mse_theory_all[label]['mse'][0, band_i]
                                  / mse_theory_all[label]['mse'][m, band_i])
                   for label in labels_by_N]

        if m == 1:
            mse_m_ext = mse_raster(N_ext, _fbar(1, band_i), sigma_dark[1, band_i], tot, gamma)
        elif m == 2:
            mse_m_ext = mse_hadam_diff(N_ext, P, _fbar(2, band_i), sigma_dark[2, band_i], tot, gamma)
        elif m == 3:
            mse_m_ext = mse_hadam_diff(N_ext, N_ext, _fbar(3, band_i), sigma_dark[3, band_i], tot, gamma)
        else:  # m == 4, S1
            mse_m_ext = mse_smatrix(N_ext, N_ext, _fbar(4, band_i), sigma_dark[4, band_i], tot, gamma)
        gain_th_ext = 10 * np.log10(mse_fh2_th / mse_m_ext)

        N_th_full = np.concatenate([N_ext, N_list])
        gain_th_full = np.concatenate([gain_th_ext, gain_th])

        # Marker shape follows the band (wavelength x channel count) --
        # see band_markers above.
        ax.plot(N_list, gain_exp, marker=band_markers[band_i], linestyle='none',
                 color=colors[m-1], markersize=8, markeredgecolor='k')
        ax.plot(N_th_full, gain_th_full, linestyle='-', color=colors[m-1], linewidth=lw)

    ax.axhline(y=0, color='k', linestyle='-')
    ax.set_xscale('log', base=2)
    ax.set_xticks(N_ticks)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True)
    ax.set_title(band_channel_labels[band_i], fontsize=fs)

    if sp % 2 == 0:
        ax.set_ylabel('MSE gain w.r.t. FH2 (in dB)', fontsize=fs-2)
    if sp // 2 == 1:
        ax.set_xlabel('N (pixel number in the freeform region)', fontsize=fs-4)

# One shared legend for the whole figure instead of one per subplot.
method_handles = [plt.Line2D([0], [0], color=colors[m-1], linewidth=lw,
                              label=method_labels_all[m]) for m in [1, 2, 3, 4]]
axs[0].legend(handles=method_handles, loc='lower left', fontsize=fs-6)

plt.tight_layout()

if save_tag:
    plt.rcParams['text.usetex'] = True
    fig.savefig(fig_folder/('figure_S19_gain_vs_N_theory_vs_exp_vs_FH2_4subplots.'+ext), transparent=True, dpi=300)

plt.show()


