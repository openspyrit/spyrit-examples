# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 19:26:29 2025

@author: ducros
"""
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

#%% Simulated/theoretical MSEs
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

tot   = 1        # total acquisition time in s
alpha = 1e7      # intensity in photons/pixel/s or None 
sigma = 17.0     # gaussian std
gamma = 1.00     # 
N = 4096
P = 128**2

# flux
mm = -2
MM = 10
step = 0.01

f_mean = 10**np.arange(mm,MM + step, step)
fref   = sigma**2 / gamma**2 / tot
f_mean_ref = f_mean/fref

# MSE
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

#%% Experimental MSEs

# experimental SNR
ind_msnr = [2,5,8,11]
ind_std  = [1,4,7,10]
ind_fbar = [0,3,6,9]

# These values were computed in figure_7.py
data = np.array([[ 50.57,   5.96,  18.58,   8.2 ,   2.33,  10.93,   1.99,   2.02,
         -0.14,   0.33,   0.65,  -5.89],
       [ 50.02,  26.98,   5.36,   8.68,   8.57,   0.11,   0.  ,  20.93,
        -20.28,   0.  ,   5.76, -24.97],
       [ 50.64,   3.43,  23.38,   8.16,   1.27,  16.16,   2.02,   1.75,
          1.26,   0.32,   0.51,  -3.93],
       [ 50.67,   3.08,  24.34,   8.17,   1.21,  16.59,   2.03,   1.03,
          5.89,   0.33,   0.33,  -0.06],
       [ 50.7 ,   3.01,  24.52,   8.18,   1.16,  16.95,   2.03,   0.88,
          7.25,   0.33,   0.29,   0.86]])

# MSNR (already in data, just is a check)
msnr = 20*np.log10(data[:,ind_fbar]/data[:,ind_std])

# boost from MSNR with respect to S1
gain = data[:4,ind_msnr] - data[4,ind_msnr]


# These values were computed in figure_7.py
sigma_dark = np.array([[19.06419074, 18.59970943, 16.18167311, 16.95223562],
       [29.58125051, 29.37583337, 17.28607546, 17.99641753],
       [19.06419074, 18.59970943, 16.18167311, 16.95223562],
       [24.02996294, 23.81448521, 16.70019888, 17.40761311],
       [29.58125051, 29.37583337, 17.28607546, 17.99641753]])

fbar = np.array([data[-1,0], data[-1,0], data[-1,6], data[-1,6]])*1e3/31
fbar_ref = fbar / (sigma_dark**2 /32.768)

#%% Plot options
plt.rcParams['text.usetex'] = True # faster than True
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['font.size'] = 14

fig_folder = Path('figures/presentation/')

save_tag = True
ext = 'png'
lw = 2 # line width
fs = 18

    
    
#%%

# Experimental MSNR with respect to H1
gain = data[:3,ind_msnr] - data[3,ind_msnr]

# Theoretical MSNR
boost_fh2 = 10*np.log10(mse_h1/mse_fh2) 
boost_rs  = 10*np.log10(mse_h1/mse_rs) 
boost_mh2 = 10*np.log10(mse_h1/mse_mh2) 


plt.figure()

plt.semilogx(f_mean_ref, boost_fh2, label='Full 2D', linewidth=lw)
plt.semilogx(f_mean_ref, boost_rs,  label='Raster scan',  linewidth=lw)
plt.semilogx(f_mean_ref, boost_mh2, label='Masked 2D', linewidth=lw)
plt.axhline(y=0, color='k', linestyle='-', label='1D', linewidth=lw)
    
symbol = 'ovd^'
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
 
for i in range(sigma_dark.shape[1]):
    for j in range(3):
        plt.semilogx(fbar_ref[j,i], gain[j,i], symbol[i], color=colors[j])

# plt.title("S matrix MSE gain")
plt.xlabel(r'Normalized mean count $\bar{f} / f_{\rm ref}$', fontsize=fs)
plt.ylabel('MSE gain w.r.t. 1D Hadamard (in dB)', fontsize=fs)
plt.legend(loc='lower right', fontsize=fs-2)
plt.grid(True)

# vertical lines
plt.axvline(x=N-4, color=colors[1], linestyle='--', linewidth=lw)
    
# limits
plt.xlim([1e-1,1e5])

# fill
ax = plt.gca()
plt.fill_between((f_mean/fref), 0, 1, where=f_mean/fref >= N-4,
                 alpha=.15, color = colors[1], transform=ax.get_xaxis_transform())
plt.tight_layout()

# save
if save_tag:
    plt.rcParams['text.usetex'] = True
    #plt.rcParams['lines.linewidth'] = 0.5
    plt.savefig(fig_folder/('figure_9_wrt_1D.'+ext), transparent=True, dpi=300)