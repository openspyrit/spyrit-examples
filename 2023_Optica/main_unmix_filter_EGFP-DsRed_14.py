# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 11:19:23 2023

@author: ducros
"""
# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
from fonction.matrix_tools import smooth

#%% Load reference spectrum
notch_path = "./data/Reference_spectra/ZET532NF_transmission.csv"
EGFP_path = "./data/Reference_spectra/EGFP_fpbase_spectra.csv"
DsRed2_path = "./data/Reference_spectra/DsRed2_fpbase_spectra.csv"


data_DsRed = np.genfromtxt(DsRed2_path, delimiter=',',skip_header=True)
data_EGFP  = np.genfromtxt(EGFP_path, delimiter=',',skip_header=True)
data_notch = np.genfromtxt(notch_path, delimiter=';',skip_header=True)

# on utilise les spectres expérimentaux. 
# La DsRed est en réalité de la DsRed2

Nc = 128 # number of channels

# Interpolate spectra to have Nc channels
from scipy import interpolate

#L_lambda = np.linspace(500, 608, Nc)
L_lambda = np.linspace(488, 610, Nc)
f = interpolate.interp1d(data_EGFP[:,0], data_EGFP[:,1], kind='cubic')
L_EGFP = f(L_lambda)   
f = interpolate.interp1d(data_DsRed[:,0], data_DsRed[:,1], kind='cubic')
L_DsRed_exp = f(L_lambda)
f = interpolate.interp1d(data_notch[:,0], data_notch[:,1], kind='cubic')
L_notch = f(L_lambda)

# plot
fig, axs = plt.subplots()
axs.plot(L_lambda, L_DsRed_exp, label='DsRed') 
axs.plot(L_lambda, L_EGFP, label='EGFP')
axs.plot(L_lambda, L_notch, label='Notch filter')
axs.set_xlabel('wavelength (nm)')
axs.set_title('Spectra used for unmixing')
axs.legend()

#%% Load hypercube
from pathlib import Path

# Load data
Nl = 512 # number of pixcels along the y dimensions 
Nh = 512 # number of measured Walsh_Hadmard coefficients (correpond to the h dimensions)
Nc = 128 # number of channels

T_list = range(1,27)    # slice indices
load_path = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D'
suffix = '_shift'               # with leading underscore e.g. '', '_shift' '_registered'
recon = 'tikhonet50_div1.5' + suffix         # 'pinv'  'tikhonet50_div1.5'

# all slices are unmixed jointly!
Nz = T_list.stop - T_list.start + 1
xyzl_cube = np.zeros((Nl,Nh,Nz,Nc))

# loop over z-slices
for z, t in enumerate(T_list):
    if t<6:
        date = '2023_03_13'
        Run = f'RUN{t+1:04}'
    else:
        date = '2023_03_14'
        Run = f'RUN{t-5:04}'
    
    # Load
    filename = f'T{t}_{Run}_{date}_rec_{recon}_exp_{Nl}x{Nh}x{Nc}.npy'
    print(filename)
    xyl_cube = np.load(Path(load_path) / 'Reconstruction/hypercube'/ recon / filename)
    
    xyzl_cube[:,:,z,:] = xyl_cube
    
# Replace negative values by zeros
# xyzl_cube = np.where(xyzl_cube > 0, xyzl_cube, 0)

#%% Double check pinv recon from pilot
# from pathlib import Path

# # Load data
# Nl = 512 # number of pixcels along the y dimensions 
# Nh = 512 # number of measured Walsh_Hadmard coefficients (correpond to the h dimensions)
# Nc = 128 # number of channels

# T_list = range(1,27)    # slice indices
# load_path = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D'

# # all slices are unmixed jointly!
# Nz = T_list.stop - T_list.start + 1
# xyzl_cube = np.zeros((Nl,Nh,Nz,Nc))

# # loop over z-slices
# for z, t in enumerate(T_list):
#     if t<6:
#         date = '2023_03_13'
#         Run = f'RUN{t+1:04}'
#     else:
#         date = '2023_03_14'
#         Run = f'RUN{t-5:04}'
    
#     filename = f'T{t}_{Run}_{date}_Had_rc_pinv_{Nl}x{Nh}x{Nc}.npy'
#     print(filename)
#     xyl_cube = np.load(Path(load_path) / 'Reconstruction/hyper_cube_pilot' / filename)
    
#     xyzl_cube[:,:,z,:] = xyl_cube
    
# Replace negative values by zeros
# xyzl_cube = np.where(xyzl_cube > 0, xyzl_cube, 0)

#%% Estimate auto-fluorescence spectrum
from matplotlib.patches import Rectangle

# Area where the autofluorescence is estimated, which corresponds to the "sac vitellin"
x1,x2 = 220,260
y1,y2 = 335,355
z1,z2 = 1,5

spec_fluo = np.mean(xyzl_cube[x1:x2, y1:y2, z1:z2,:], axis=(0,1,2))
spec_fluo /= np.max(spec_fluo)

# plot autofluorescence area
fig, axs = plt.subplots(1, z2-z1, figsize=(2*(z2-z1),z2-z1))

for i, z in enumerate(range(z1,z2)):
    axs[i].imshow(xyzl_cube[:,:,z,64])
    axs[i].set_xlabel('y-axis')
    axs[i].set_ylabel('x-axis')
    # Create a Rectangle patch
    axs[i].add_patch(Rectangle((y1, x1), y2-y1+1, x2-x1+1, linewidth=1, edgecolor='r', facecolor='none'))
    
fig.suptitle(f'Autofluorescence estimation area; x in [{x1}-{x2}]; y in [{y1}-{y2}]')



fig, axs = plt.subplots()
axs.plot(L_lambda, L_DsRed_exp, '--r',  label='DsRed (theo)') 
axs.plot(L_lambda, L_EGFP*L_notch, '--g',  label='EGFP x Notch (theo)') 
axs.plot(L_lambda, L_notch, '--k',  label='Notch (theo)') 
axs.plot(L_lambda, spec_fluo, color=(1,0.5, 0),  label='autofluo (estimated)')
axs.legend()

#%% Estimate fluorescence spectra
from matplotlib.patches import Rectangle

z = 4
# Area where the dsre spectrum is estimated
x1_r,x2_r = 221,226
y1_r,y2_r = 232,237

# Area where the egfp spectrum is estimated
x1_g,x2_g = 321+16,330+16
y1_g,y2_g = 367,372

L_DsRed_pure = np.mean(xyzl_cube[x1_r:x2_r, y1_r:y2_r, z,:], axis=(0,1))
L_DsRed_pure /= L_DsRed_pure.max() 
L_EGFP_pure = np.mean(xyzl_cube[x1_g:x2_g, y1_g:y2_g, z,:], axis=(0,1))
L_EGFP_pure /= L_EGFP_pure.max() 

# plot area
fig, axs = plt.subplots(1, 2)
axs[0].imshow(xyzl_cube[:,:,z,64])
axs[0].set_xlabel('y-axis')
axs[0].set_ylabel('x-axis')
# Create a Rectangle patch
axs[0].add_patch(Rectangle((y1_r, x1_r), y2_r-y1_r+1, x2_r-x1_r+1, linewidth=2, edgecolor='r', facecolor='none'))
axs[0].add_patch(Rectangle((y1_g, x1_g), y2_g-y1_g+1, x2_g-x1_g+1, linewidth=2, edgecolor='g', facecolor='none'))
    

axs[1].plot(L_lambda, L_DsRed_pure, 'r', label='DsRed (expe)')
axs[1].plot(L_lambda, L_DsRed_exp, '--r',  label='DsRed (theo)') 
axs[1].plot(L_lambda, L_EGFP_pure, 'g', label='EGFP (expe)')
axs[1].plot(L_lambda, L_EGFP*L_notch, '--g',  label='EGFP x Notch (theo)') 
axs[1].plot(L_lambda, L_notch, '--k',  label='Notch (theo)') 
axs[1].legend()

fig.suptitle(f'DsRed estimation area; x in [{x1_r}-{x2_r}]; y in [{y1_r}-{y2_r}]')

#%% Unmixing
from pysptools import abundance_maps

member_list = ['DsRed','EGFP','Autofluo'] #['DsRed','EGFP','Autofluo','Noise']
unmix = '' + suffix   # '_calib_blind_' '_calib_' '' 
unmixing_folder = '/Unmixing' +  unmix + '/'

method_unmix = 'NNLS' # 'NNLS''_UCLS'
Nm = 4
Nl,Nh = xyzl_cube.shape[:2] # new shape after registration

# blind region (removed from unmixing)
blind_start = 518   # in nm
blind_end = 547     # in nm
ind = np.where((L_lambda < blind_start ) | (L_lambda > blind_end))

# End member (spectra) matrix
U = np.zeros((Nm,Nc))
U[0] = L_DsRed_exp*L_notch#
U[1] = L_EGFP*L_notch #L_mRFP
U[2] = spec_fluo # spectre d'autofluo normalisé

folder_unmix = recon + '_' + method_unmix

print('-- unmixing')

# measurment matrix with all pixels across rows
M = np.reshape(xyzl_cube, (-1,Nc)) 

# Remove blind region
U_blind = U[:,ind[0]]
M_blind = M[:,ind[0]]

# https://pysptools.sourceforge.io/abundance_maps.html
if method_unmix == 'NNLS':
    # Non-negative Constrained Least Squares
    Abondance = abundance_maps.amaps.NNLS(M_blind, U_blind)
    
elif method_unmix == 'UCLS':
    # Unconstrained Least Squares
    Abondance = abundance_maps.amaps.UCLS(M_blind, U_blind)

abond_unmix_4D = np.reshape(Abondance,(Nl,Nh,Nz,Nm))

#%% Manual registration (compensate for sample shift during acquisition)
x_shift = 36
y_shift = 12

print('-- registration')
abond_unmix_4D_reg = np.zeros_like(abond_unmix_4D)
nl, nh = Nl-x_shift, Nh-y_shift
dl, dh = x_shift//2, y_shift//2

abond_unmix_4D_reg[dl:dl+nl, dh:dh+nh, :5,  :] = abond_unmix_4D[:(Nl-x_shift), :(Nh-y_shift), :5,   :]
abond_unmix_4D_reg[dl:dl+nl, dh:dh+nh, 5:13,:] = abond_unmix_4D[:(Nl-x_shift), y_shift:,      5:13, :]
abond_unmix_4D_reg[dl:dl+nl, dh:dh+nh, 13:, :] = abond_unmix_4D[x_shift:,      y_shift:,      13:,  :]

abond_unmix_4D = abond_unmix_4D_reg

#%% Separate fluorophores simulating two band-pass filters
Nf = 2 # number of filter

filtering_folder = '/Filtering' + unmix + '/'
filter_list = ['green','red']

c_egfp = 510  # in nm
c_dsred = 600    # in nm
bandwidth = 10 # in nm


folder_filter = recon # 

print('-- filtering')

Filt_sep = np.zeros((Nf,Nc))
Filt_sep[0] = np.where((L_lambda < c_egfp + bandwidth/2)&(L_lambda > c_egfp - bandwidth/2), 
                    1, 0)
Filt_sep[1] = np.where((L_lambda < c_dsred + bandwidth/2)&(L_lambda > c_dsred - bandwidth/2), 
                    1, 0)

abond_filter_4D = M @ Filt_sep.T
abond_filter_4D = abond_filter_4D.reshape((Nl,Nh,Nz,Nf))

#%% save data
# Save abondance maps
from spyrit.misc.disp import add_colorbar, noaxis

save_tag = True

# unmixing
if save_tag:
    # create folder
    save_path = Path(load_path + unmixing_folder + folder_unmix)
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # save unmixed data    
    filename = 'abondance.npy'
    np.save(save_path / filename, abond_unmix_4D)
    
    # save members/spectra  
    filename = 'spectra.npy'
    np.save(save_path / filename, U_blind)

# filtering
if save_tag:    
    # create folder
    save_path = Path(load_path + filtering_folder + folder_filter)
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # save unmixed data    
    filename = 'abondance.npy'
    np.save(save_path / filename, abond_filter_4D)
    
    # save members/spectra  
    filename = 'spectra.npy'
    np.save(save_path / filename, Filt_sep)

#%% Plot and save unmixing results
save_fig = True
save_ext = ['svg'] # ['pdf', 'svg', 'png']

if save_fig:
    # create folder
    save_path = Path(load_path + unmixing_folder + folder_unmix)
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
if save_fig:
    filename = 'spectra_pure_pixels'
    for ext in save_ext:
        filename_with_ext = filename + '.' + ext
        print(filename_with_ext)
        plt.savefig(save_path / filename_with_ext, bbox_inches='tight', dpi=600)
    plt.close(fig)  
        
# Abondance maps
# loop over all z-slices
for z, t in enumerate(T_list):
    # Plot
    fig, axs = plt.subplots(1, len(member_list), figsize=(2*len(member_list),len(member_list)))
    
    # loop over all members/fluorophores
    for m, member in  enumerate(member_list):
        
        im = axs[m].imshow(abond_unmix_4D[:,:,z,m])
        axs[m].set_title(f'{member}')
        add_colorbar(im, 'bottom')
    
    noaxis(axs)
    
    if save_fig:
        filename = f'T{t}_abondance'
        for ext in save_ext:
            filename_with_ext = filename + '.' + ext
            print(filename_with_ext)
            plt.savefig(save_path / filename_with_ext, bbox_inches='tight', dpi=600)
        plt.close(fig)

# unmixing: spectra
col = ['r', 'g', (1,0.5, 0)]  # color list

L_lambda_blind = L_lambda[ind[0]]

fig, axs = plt.subplots()
for m in range(len(member_list)):
    axs.plot(L_lambda_blind, U_blind[m], color=col[m], label = member_list[m])
    
axs.plot(L_lambda, L_notch, 'k', ls = 'dashed', label = 'notch filter')
axs.fill_between(L_lambda, L_notch, 1, color='k', alpha=.1)
axs.legend()
plt.xlabel('Wavelenght (nm)')
plt.ylabel('Intensity (normalized)')

if save_fig:       
    filename = 'spectra'
    for ext in save_ext:
        filename_with_ext = filename + '.' + ext
        print(filename_with_ext)
        plt.savefig(save_path / filename_with_ext, bbox_inches='tight', dpi=600)
    plt.close()
    
#%% plot autofluorescence area
fig, axs = plt.subplots(1, z2-z1, figsize=(2*(z2-z1),z2-z1))

for i, z in enumerate(range(z1,z2)):
    axs[i].imshow(xyzl_cube[:,:,z,64])
    axs[i].set_xlabel('y-axis')
    axs[i].set_ylabel('x-axis')
    # Create a Rectangle patch
    axs[i].add_patch(Rectangle((y1, x1), y2-y1+1, x2-x1+1, linewidth=1, edgecolor='r', facecolor='none'))
    
fig.suptitle(f'Autofluorescence estimation area; x in [{x1}-{x2}]; y in [{y1}-{y2}]')

save_ext = ['svg'] # ['pdf', 'svg', 'png']
filename = 'autofluorescence'
for ext in save_ext:
    filename_with_ext = filename + '.' + ext
    print(filename_with_ext)
    plt.savefig(save_path / filename_with_ext, bbox_inches='tight', dpi=600)
plt.close(fig)

#%% Plot Filtering
save_fig = True
save_ext = ['svg'] # ['pdf', 'svg', 'png']

if save_fig:
    # create folder
    save_path = Path(load_path + filtering_folder + folder_filter)
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
# Abondance maps
# loop over all z-slices
for z, t in enumerate(T_list):
    # Plot
    fig, axs = plt.subplots(1, len(filter_list))
    
    # loop over all members/fluorophores
    for m, filt in  enumerate(filter_list):
        
        im = axs[m].imshow(abond_filter_4D[:,:,z,m])
        axs[m].set_title(f'{filt}')
        add_colorbar(im, 'bottom')
    
    noaxis(axs)
    
    if save_fig:
        filename = f'T{t}_abondance'
        for ext in save_ext:
            filename_with_ext = filename + '.' + ext
            print(filename_with_ext)
            plt.savefig(save_path / filename_with_ext, bbox_inches='tight', dpi=600)
        plt.close(fig)
        
# Spectra
col = ['r', 'g', (1,0.5, 0)]  # color list
plt.rcParams['font.size'] = '16'

fig, axs = plt.subplots()
axs.plot(L_lambda, L_DsRed_exp, color=col[0], label = member_list[0])
axs.plot(L_lambda, L_EGFP,      color=col[1], label = member_list[1])
axs.plot(L_lambda, U[2],        color=col[2], label = member_list[2])
axs.plot(L_lambda, L_notch,     color='k', ls = 'dashed', label = 'notch filter')
axs.fill_between(L_lambda, L_notch, 1, color='k', alpha=.1)
axs.plot(L_lambda, Filt_sep[0], color=col[1], ls = 'dashdot', label = 'green filter')
axs.fill_between(L_lambda, Filt_sep[0], 0, color=col[1], alpha=.1)
axs.plot(L_lambda, Filt_sep[1], color=col[0], ls = 'dashdot', label = 'red filter')
axs.fill_between(L_lambda, Filt_sep[1], 0, color=col[0], alpha=.1)
axs.legend(loc='upper right', bbox_to_anchor=(1.5, 1.05), frameon=False)
plt.xlabel('Wavelenght (nm)')
plt.ylabel('Intensity (normalized)')

if save_fig:      
    filename = 'spectra'
    for ext in save_ext:
        filename_with_ext = filename + '.' + ext
        print(filename_with_ext)
        plt.savefig(save_path / filename_with_ext, bbox_inches='tight', dpi=600)
    plt.close()