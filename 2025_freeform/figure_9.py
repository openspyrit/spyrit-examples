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
# fh2 = np.array([31.93, 3.57, 19.0, 5.14, 1.36, 11.5, 1.38, 1.17, 1.4, 0.23, 0.37, -4])
# rs  = np.array([32.08, 14.22, 7.1, 5.54, 4.91, 1.0, None, 9.8, -16.8, None, 3.13, -22.7])
# mh2 = np.array([32.57, 1.98, 24.3, 5.28, 0.71, 17.5, 1.41, 1.01, 2.9, 0.23, 0.3, -2.4 ])
# h1  = np.array([31.37, 1.8, 24.8, 5.06, 0.7, 17.2, 1.39, 0.62, 7.1, 0.22, 0.2, 0.8 ])
# s1  = np.array([32.58, 1.74, 25.5, 5.27, 0.66, 18.0, 1.41, 0.5, 9.0, 0.23, 0.17, 2.6])

ind_msnr = [2,5,8,11]
ind_std  = [1,4,7,10]
ind_fbar = [0,3,6,9]

# data = np.stack((fh2, rs, mh2, h1, s1))
# data = data.astype(np.float64)  # dtype = object -> dtype = np.float64

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

# boost from MSNR
gain = data[:4,ind_msnr] - data[4,ind_msnr]

# boost from STD
#gain = 20*np.log10(data[4,ind_std]/data[:4,ind_std])

# Equivalent normalised fbarref
fbar_fh2 = find_closest_abscissa(boost_fh2, f_mean_ref, gain[0,:])
fbar_rs  = find_closest_abscissa(boost_rs,  f_mean_ref, gain[1,:])
fbar_mh2 = find_closest_abscissa(boost_mh2, f_mean_ref, gain[2,:])
fbar_h1  = find_closest_abscissa(boost_h1,  f_mean_ref, gain[3,:])

#%% Plot
plt.rcParams['text.usetex'] = True # faster than True


fig_folder = Path('figures')

save_tag = True
ext = 'pdf'
lw = 2 # line width
fs = 18  


plt.figure()

plt.axhline(y=0, color='k', linestyle='-')

plt.semilogx(f_mean_ref, boost_fh2, label='FH2', linewidth=lw)
plt.semilogx(f_mean_ref, boost_rs,  label='RS',  linewidth=lw)
plt.semilogx(f_mean_ref, boost_mh2, label='MH2', linewidth=lw)
plt.semilogx(f_mean_ref, boost_h1,  label='H1',  linewidth=lw)
    
symbol = 'ovd^'
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
 
for i in range(len(fbar_fh2)):
    plt.semilogx(fbar_fh2[i], gain[0,i], symbol[i], color=colors[0])
    plt.semilogx(fbar_rs[i],  gain[1,i], symbol[i], color=colors[1])
    plt.semilogx(fbar_mh2[i], gain[2,i], symbol[i], color=colors[2])
    plt.semilogx(fbar_h1[i],  gain[3,i], symbol[i], color=colors[3])

# plt.title("S matrix MSE gain")
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
    plt.savefig(fig_folder/('figure_9a.'+ext), transparent=True, dpi=300)
    
#%% Plot from theoretical fbar /fref
# fbar_ref_581 = data[-1,0]*1e3/31/(17**2/32.768)
# fbar_ref_726 = data[-1,6]*1e3/31/(17**2/32.768)
# fbar_ref = [fbar_ref_581,fbar_ref_581,fbar_ref_726,fbar_ref_726]

# These values were computed in figure_7.py
sigma_dark = np.array([[19.06419074, 18.59970943, 16.18167311, 16.95223562],
       [29.58125051, 29.37583337, 17.28607546, 17.99641753],
       [19.06419074, 18.59970943, 16.18167311, 16.95223562],
       [24.02996294, 23.81448521, 16.70019888, 17.40761311],
       [29.58125051, 29.37583337, 17.28607546, 17.99641753]])

fbar = np.array([data[-1,0], data[-1,0], data[-1,6], data[-1,6]])*1e3/31
fbar_ref = fbar / (sigma_dark**2 /32.768)

plt.figure()

plt.axhline(y=0, color='k', linestyle='-')

plt.semilogx(f_mean_ref, boost_fh2, label='FH2', linewidth=lw)
plt.semilogx(f_mean_ref, boost_rs,  label='RS',  linewidth=lw)
plt.semilogx(f_mean_ref, boost_mh2, label='MH2', linewidth=lw)
plt.semilogx(f_mean_ref, boost_h1,  label='H1',  linewidth=lw)
    
symbol = 'ovd^'
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
 
for i in range(len(fbar_fh2)):
    plt.semilogx(fbar_ref[0,i], gain[0,i], symbol[i], color=colors[0])
    plt.semilogx(fbar_ref[1,i], gain[1,i], symbol[i], color=colors[1])
    plt.semilogx(fbar_ref[2,i], gain[2,i], symbol[i], color=colors[2])
    plt.semilogx(fbar_ref[3,i], gain[3,i], symbol[i], color=colors[3])
    # plt.semilogx(fbar_ref[i], gain[0,i], symbol[i], color=colors[0])
    # plt.semilogx(fbar_ref[i], gain[1,i], symbol[i], color=colors[1])
    # plt.semilogx(fbar_ref[i], gain[2,i], symbol[i], color=colors[2])
    # plt.semilogx(fbar_ref[i], gain[3,i], symbol[i], color=colors[3])

# plt.title("S matrix MSE gain")
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
    plt.savefig(fig_folder/('figure_9b.'+ext), transparent=True, dpi=300)