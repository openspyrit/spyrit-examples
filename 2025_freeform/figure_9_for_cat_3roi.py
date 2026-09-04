#%% Entries -- per-ROI configuration
# N is the mask size (number of ROI pixels): 4096 for 'cat' (head+legs),
# 2048 for 'cat-head', 1024 for 'cat-legs'. moy/MSNR/sigma_dark are read
# per ROI from the folder written by freefom_add_figure_of_cat_3roi.py.
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

masks_cfg = [
    dict(label='cat',      N=4096),
    dict(label='cat-head', N=2048),
    dict(label='cat-legs', N=1024),
]

# Values computed and saved by freefom_add_figure_of_cat_3roi.py
cat_result_folder = Path('../result/freeform/figures_cat_3roi')

fig_folder = cat_result_folder
save_tag = True
ext = 'pdf'

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
ti = 2
tot   = 1        # total acquisition time in s
alpha = 1e7      # intensity in photons/pixel/s or None
sigma = 17.0     # gaussian std
gamma = 1.00     #
P = 128**2

# flux
mm = -2
MM = 10
step = 0.01

f_mean = 10**np.arange(mm,MM + step, step)
fref   = sigma**2 / gamma**2 / tot
f_mean_ref = f_mean/fref

lw = 2 # line width
fs = 18

#%% MAIN LOOP over the 3 ROIs (cat, cat-head, cat-legs)
for cfg in masks_cfg:

    label = cfg['label']
    N = cfg['N']
    print(f'=== ROI = {label} (N = {N}) ===')

    # ---- Simulated/theoretical MSEs, for this ROI's N ----
    mse_fh2 = mse_hadam_diff(P, P, f_mean, sigma, tot, gamma)
    mse_rs  = mse_raster(N, f_mean, sigma, tot, gamma)
    mse_mh2 = mse_hadam_diff(N, P, f_mean, sigma, tot, gamma)
    mse_h1  = mse_hadam_diff(N, N, f_mean, sigma, tot, gamma)
    mse_s1  = mse_smatrix(N, N, f_mean, sigma, tot, gamma)

    # boosts
    boost_fh2 = 10*np.log10(mse_s1 / mse_fh2)
    boost_rs  = 10*np.log10(mse_s1 / mse_rs)
    boost_mh2 = 10*np.log10(mse_s1 / mse_mh2)
    boost_h1  = 10*np.log10(mse_s1 / mse_h1)

    # ---- Experimental data for this ROI ----
    # moy/MSNR: [5 methods, 4 bands], rows = FH2, RS, MH2, H1, S1 (same
    # order as method_list in freefom_add_figure_of_cat_3roi.py); columns:
    # 581nm (31ch), 581nm (5ch), 726nm (31ch), 726nm (5ch)
    moy        = np.load(cat_result_folder / f'moy_{label}.npy')
    msnr       = np.load(cat_result_folder / f'MSNR_{label}.npy')
    sigma_dark = np.load(cat_result_folder / f'sigma_dark_{label}.npy')

    # gain (dB) of FH2/RS/MH2/H1 relative to S1, per band
    gain = msnr[:4, :] - msnr[4:5, :]

    # Equivalent normalised fbarref (curve-matching, kept for reference --
    # not used in the active plot below, which instead positions points
    # from the experimental mean count / dark noise, see fbar_ref)
    fbar_fh2 = find_closest_abscissa(boost_fh2, f_mean_ref, gain[0, :])
    fbar_rs  = find_closest_abscissa(boost_rs,  f_mean_ref, gain[1, :])
    fbar_mh2 = find_closest_abscissa(boost_mh2, f_mean_ref, gain[2, :])
    fbar_h1  = find_closest_abscissa(boost_h1,  f_mean_ref, gain[3, :])

    #%% Plot from theoretical fbar/fref
    # S1's mean count in the wide (31-channel) bands, reused for the
    # corresponding narrow band -- same convention as the original,
    # single-ROI script.
    fbar = np.array([moy[-1, 0], moy[-1, 0], moy[-1, 2], moy[-1, 2]]) * 1e3 / 31
    fbar_ref = fbar / (sigma_dark**2 / (32.768 * ti))

    plt.figure()

    plt.axhline(y=0, color='k', linestyle='-')

    plt.semilogx(f_mean_ref, boost_fh2, label='FH2', linewidth=lw)
    plt.semilogx(f_mean_ref, boost_rs,  label='RS',  linewidth=lw)
    plt.semilogx(f_mean_ref, boost_mh2, label='MH2', linewidth=lw)
    plt.semilogx(f_mean_ref, boost_h1,  label='H1',  linewidth=lw)

    symbol = 'ovd^'
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i in range(gain.shape[1]):
        plt.semilogx(fbar_ref[0, i], gain[0, i], symbol[i], color=colors[0])
        plt.semilogx(fbar_ref[1, i], gain[1, i], symbol[i], color=colors[1])
        plt.semilogx(fbar_ref[2, i], gain[2, i], symbol[i], color=colors[2])
        plt.semilogx(fbar_ref[3, i], gain[3, i], symbol[i], color=colors[3])

    plt.title(f'ROI: {label} (N = {N})')
    plt.xlabel(r'Normalized mean count $\bar{f} / f_{\rm ref}$', fontsize=fs)
    plt.ylabel('MSE gain w.r.t. S1 (in dB)', fontsize=fs)
    plt.legend(loc='lower right', fontsize=fs-2)
    plt.grid(True)

    # vertical lines
    plt.axvline(x=2, color='k', linestyle='--', linewidth=lw)
    plt.axvline(x=N-4, color=colors[1], linestyle='--', linewidth=lw)

    # limits
    plt.xlim([1e-2,1e5])

    # fill
    ax = plt.gca()
    plt.fill_between((f_mean/fref), 0, 1, where=f_mean/fref <= 2,
                     alpha=.15, color = 'grey', transform=ax.get_xaxis_transform())

    plt.fill_between((f_mean/fref), 0, 1, where=f_mean/fref >= N-4,
                     alpha=.15, color = colors[1], transform=ax.get_xaxis_transform())
    plt.tight_layout()

    # save
    if save_tag:
        plt.rcParams['text.usetex'] = True
        #plt.rcParams['lines.linewidth'] = 0.5
        plt.savefig(fig_folder/(f'figure_9b_{label}.'+ext), transparent=True, dpi=300)

#%% Display the 3 ROI figures
plt.show()
