#%% mean squared error
def mse_raster(f_n, sigma, dt, gamma):
    return f_n / dt + sigma**2 / (gamma*dt)**2
    #return 1 / (sigma**2/(gamma*dt)**2) * ( f_n * gamma**2 * dt / (sigma**2) + 1)
    
    
def mse_hadam_split(N, f_mean, sigma, dt, gamma=1.0):
    return f_mean / dt + (2/N) * sigma**2 / (gamma*dt)**2


def mse_smatrix(N, f_mean, sigma, dt, gamma=1.0):
    return 2 * f_mean/dt + 4 / N * sigma**2 / (gamma*dt)**2
    # fref =  sigma**2 / gamma**2 / t
    # 2/t*(N*f_mean + 2*M *fref]


#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# save_tag = False
lw = 1 # line width
fig_folder = Path('figures')


# # raster scan
mm = -2
MM = 10
step = 0.01
f_plot = 10**np.arange(mm,MM + step, step)


ii=0
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
    

#%% MSEs
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True

save_tag = False

tot   = 1        # total acquisition time in s
alpha = 1e7      # intensity in photons/pixel/s or None 
sigma = 50.0     # gaussian std
gamma = 1.41         # 
N_list = [128]


plt.figure()

# raster scan
for N in N_list:
    
    dt_rs = tot / N
    fref = sigma**2 / gamma**2 / tot
    
    # hadam split
    mse_rs = mse_raster(f_plot, sigma, dt_rs, gamma)
    mse_dB = 10*np.log10(mse_rs)
    plt.semilogx(f_plot/fref, mse_dB, 
                 label='RS', color=colors[1], linestyle='--', linewidth=lw)


# split hadamard
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
ii=0

for N in N_list:
    
    dt_hs = tot / N / 2
    fref = sigma**2 / gamma**2 / tot
    
    # hadam split    
    mse_hs = mse_hadam_split(N, f_plot, sigma, dt_hs, gamma)
    mse_dB = 10*np.log10(mse_hs)
    plt.semilogx(f_plot/fref, mse_dB, 
                 label='SH', color=colors[1], linestyle='-.', linewidth=lw)
    ii = ii +1
    
# S-matrix
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
ii=0

for N in N_list:
    
    dt_sm = tot / (N)
    fref = sigma**2 / gamma**2 / tot
    
    # hadam split    
    mse_hs = mse_smatrix(N, f_plot, sigma, dt_sm, gamma)
    mse_dB = 10*np.log10(mse_hs)
    plt.semilogx(f_plot/fref, mse_dB, label='SM', color=colors[1], linewidth=lw)
    
    ii = ii +1

    
plt.xlabel(r'Normalized mean count $\bar{f} / f_{\rm ref}$')
plt.ylabel('MSE (in dB)')
plt.legend(loc='upper left')
plt.grid(True)

# vertical lines
#plt.axvline(x=4, color='k', linestyle='--')
ax = plt.gca()
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

ii=0
for N in N_list:
    plt.axvline(x=N-4, color='k', linewidth=lw)
    ii = ii +1


# limits
plt.xlim([1e-1,1e3])
plt.ylim([55,90])

# fill
plt.fill_between((f_plot/fref), 0, 1, where=f_plot/fref >= N-4, 
                 alpha=.15, color = colors[1], transform=ax.get_xaxis_transform())

# save
if save_tag:
    #plt.rcParams['text.usetex'] = True
    #plt.rcParams['lines.linewidth'] = 0.5
    plt.savefig(fig_folder/'mse.pdf', transparent=True, dpi=300)

#%% Normalized MSE
import numpy as np
import matplotlib.pyplot as plt

save_tag = False

tot   = 1        # total acquisition time in s
alpha = 1e7      # intensity in photons/pixel/s or None 
sigma = 50.0     # gaussian std
gamma = 1.41         # 
N_list = [128]


plt.figure()

# raster scan
for N in N_list:
    
    dt_rs = tot / N
    fref = sigma**2 / gamma**2 / tot
    
    # hadam split
    mse_rs = mse_raster(f_plot, sigma, dt_rs, gamma)
    mse_dB = 10*np.log10(mse_rs) - 20*np.log10(f_plot)
    plt.semilogx(f_plot/fref, mse_dB, 
                 label='raster scan', color=colors[1], linestyle='--', linewidth=lw)


# split hadamard
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
ii=0

for N in N_list:
    
    dt_hs = tot / N / 2
    fref = sigma**2 / gamma**2 / tot
    
    # hadam split    
    mse_hs = mse_hadam_split(N, f_plot, sigma, dt_hs, gamma)
    mse_dB = 10*np.log10(mse_hs) - 20*np.log10(f_plot)
    plt.semilogx(f_plot/fref, mse_dB, 
                 label='split Hadamard', color=colors[1], linestyle='-.', linewidth=lw)
    ii = ii +1
    
# S-matrix
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
ii=0

for N in N_list:
    
    dt_sm = tot / (N)
    fref = sigma**2 / gamma**2 / tot
    
    # hadam split    
    mse_hs = mse_smatrix(N, f_plot, sigma, dt_sm, gamma)
    mse_dB = 10*np.log10(mse_hs) - 20*np.log10(f_plot)
    plt.semilogx(f_plot/fref, mse_dB, label='S-matrix', color=colors[1], linewidth=lw)
    
    ii = ii +1

    
plt.xlabel(r'Normalized mean count $\bar{f} / f_{\rm ref}$')
plt.ylabel('normalized MSE (in dB)')
plt.legend(loc='upper right')
plt.grid(True)

# vertical lines
#plt.axvline(x=4, color='k', linestyle='--')
ax = plt.gca()
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

ii=0
for N in N_list:
    plt.axvline(x=N-4, color='k', linewidth=lw)
    ii = ii +1


# limits
# plt.xlim([1e-1,1e5])
# plt.ylim([-55,15])
plt.xlim([1e-1,1e5])
plt.ylim([-60,30])

# fill
plt.fill_between((f_plot/fref), 0, 1, where=f_plot/fref >= N-4, 
                 alpha=.15, color = colors[1], transform=ax.get_xaxis_transform())

# save
if save_tag:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['font.size'] = 14
    plt.savefig(fig_folder/'figure_2.pdf', transparent=True, dpi=300)