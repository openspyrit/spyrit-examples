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

tot   = 1        # total acquisition time in s
alpha = 1e7      # intensity in photons/pixel/s or None 
sigma = 17.0     # gaussian std
gamma = 1.41         # 
N_list = [2048,4096, 8192]


save_tag = True
lw = 1 # line width
fig_folder = Path('figures')


plt.figure()

# raster scan
mm = -2
MM = 10
step = 0.01
f_plot = 10**np.arange(mm,MM + step, step)

for N in N_list:
    
    dt_rs = tot / N
    dt_sm = tot / N
    fref = sigma**2 / gamma**2 / tot
    
    # hadam split
    mse_rs = mse_raster(f_plot, sigma, dt_rs, gamma)
    mse_hs = mse_smatrix(N, f_plot, sigma, dt_sm, gamma)

    boost = - 10*np.log10(mse_hs / mse_rs)
    plt.semilogx(f_plot/fref, boost, label=f'$N$ = {N}', linewidth=lw)
    
# plt.title("S matrix MSE gain")
plt.xlabel(r'Normalized mean count $\bar{f} / f_{\rm ref}$')
plt.ylabel('MSE gain (in dB)')
plt.legend(loc='upper right')
plt.grid(True)

# vertical lines
plt.axvline(x=2, color='k', linewidth=lw)

ii=0
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
for N in N_list:
    plt.axvline(x=N-4, color=colors[ii], linestyle='--', linewidth=lw)
    ii = ii +1
    
# limits
plt.xlim([1e-2,1e6])

# fill
ax = plt.gca()
plt.fill_between((f_plot/fref), 0, 1, where=f_plot/fref <= 2,
                 alpha=.15, color = 'grey', transform=ax.get_xaxis_transform())
plt.fill_between((f_plot/fref), 0, 1, where=(N_list[1]-4 >= f_plot/fref)& (f_plot/fref>= N_list[0]-4),
                 alpha=.15, color = colors[0], transform=ax.get_xaxis_transform())
plt.fill_between((f_plot/fref), 0, 1, where=(N_list[2]-4 >= f_plot/fref)& (f_plot/fref>= N_list[1]-4),
                 alpha=.15, color = colors[1], transform=ax.get_xaxis_transform())
plt.fill_between((f_plot/fref), 0, 1, where=f_plot/fref >= N-4,
                 alpha=.15, color = colors[2], transform=ax.get_xaxis_transform())


plt.axhline(y=0, color='k', linestyle='-')

# save
if save_tag:
    plt.rcParams['text.usetex'] = True
    #plt.rcParams['lines.linewidth'] = 0.5
    plt.savefig(fig_folder/'figure_4.pdf', transparent=True, dpi=300)