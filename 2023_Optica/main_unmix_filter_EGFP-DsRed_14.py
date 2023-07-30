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
EGFP_path = "./data/Reference_spectra/EGFP_fpbase_spectra_500_608.csv"
DsRed2_path = "./data/Reference_spectra/DsRed2_fpbase_spectra_500_608.csv"
DsRed_path = "./data/Reference_spectra/DsRed_fpbase_spectra_500_608.csv"
DsRed_exp_path = "./data/Reference_spectra/DsRed_express_fpbase_spectra_500_6088.csv"#

data_DsRed_exp = np.genfromtxt(DsRed_exp_path, delimiter=',',skip_header=True)
# data_DsRed = np.genfromtxt(DsRed_path, delimiter=',',skip_header=True)
# data_DsRed2 = np.genfromtxt(DsRed2_path, delimiter=',',skip_header=True)
data_EGFP = np.genfromtxt(EGFP_path, delimiter=',',skip_header=True)

# on utilise les spectres expérimentaux. 
# La DsRed est en réalité de la DsRed2

Nc = 128 # number of channels

# Interpolate spectra to have Nc channels
from scipy import interpolate

L_lambda = np.linspace(500, 608, Nc)
f = interpolate.interp1d(data_EGFP[:,0], data_EGFP[:,1], kind='cubic')
L_EGFP = f(L_lambda)   
f = interpolate.interp1d(data_DsRed_exp[:,0], data_DsRed_exp[:,1], kind='cubic')
L_DsRed_exp = f(L_lambda)

# Notch filter @ 513 nm
Filt_notch = np.ones(Nc)
Filt_notch[40:57] = np.zeros(17)
Filt_notch = smooth(Filt_notch,6)
Filt_notch[:4] = np.ones(4)
Filt_notch[128-4:] = np.ones(4)

# plot
fig, axs = plt.subplots()
axs.plot(L_lambda, L_DsRed_exp, label='DsRed') 
axs.plot(L_lambda, L_EGFP, label='EGFP')
axs.plot(L_lambda, Filt_notch, label='Notch filter')
axs.set_xlabel('wavelength (nm)')
axs.set_title('Spectra used for unmixing')
axs.legend()

#%%
from pathlib import Path

# Load data
Nl = 512 # number of pixcels along the y dimensions 
Nh = 512 # number of measured Walsh_Hadmard coefficients (correpond to the h dimensions)
Nc = 128 # number of channels

T_list = range(1,27)    # slice indices
load_path = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D'
recon = 'pinv'  # 'pinv' 'tikhonet50_div1.5'

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

#%% Estimate auto-fluorescence spectrum
from matplotlib.patches import Rectangle

# Area where the autofluorescence is estimated, which corresponds to the "sac vitellin"
x1,x2 = 220,260
y1,y2 = 335,345
z1,z2 = 1,6

spec_fluo = np.mean(xyzl_cube[x1:x2, y1:y2, z1:z2,:], axis=(0,1,2))

# plot autofluorescence area
for z in range(z1,z2):
    fig, axs = plt.subplots()
    axs.imshow(xyzl_cube[:,:,z,64])
    axs.set_title(f'Autofluorescence estimation area; x in [{x1}-{x2}]; y in [{y1}-{y2}]')
    axs.set_xlabel('y-axis')
    axs.set_ylabel('x-axis')
    # Create a Rectangle patch
    axs.add_patch(Rectangle((y1, x1), y2-y1+1, x2-x1+1, linewidth=1, edgecolor='r', facecolor='none'))

#%% Manual registration (compensate for sample shift during acquisition)
x_shift = 36
y_shift = 12

print('-- registration')

xyzl_cube_reg = np.zeros((Nl-x_shift,Nh-y_shift,Nz,Nc))
xyzl_cube_reg[:, :, :5,  :] = xyzl_cube[:(Nl-x_shift), :(Nh-y_shift), :5,   :]
xyzl_cube_reg[:, :, 5:13,:] = xyzl_cube[:(Nl-x_shift), y_shift:,      5:13, :]
xyzl_cube_reg[:, :, 13:, :] = xyzl_cube[x_shift:,      y_shift:,      13:,  :]

# abond_unmix_4D_reg = abond_unmix_4D

#%% Unmixing
from pysptools import abundance_maps

member_list = ['DsRed','EGFP','Autofluo','Noise'] #['Dsred','mRFP','AF']

method_unmix = 'NNLS' # 'NNLS''_UCLS'
Nm = 4
Nl,Nh = xyzl_cube_reg.shape[:2] # new shape after registration

# End member (spectra) matrix
U = np.zeros((Nm,Nc))
U[0] = L_DsRed_exp*Filt_notch#
U[1] = L_EGFP*Filt_notch #L_mRFP
U[2] = spec_fluo/np.max(spec_fluo) # spectre d'autofluo normalisé
U[3] = np.ones(128)*1/10 

folder_unmix = recon + '_' + method_unmix

print('-- unmixing')

# measurment matrix with all pixels across rows
M = np.reshape(xyzl_cube_reg, (-1,Nc)) 

# https://pysptools.sourceforge.io/abundance_maps.html
if method_unmix == 'NNLS':
    # Non-negative Constrained Least Squares
    Abondance = abundance_maps.amaps.NNLS(M, U)
    
elif method_unmix == 'UCLS':
    # Unconstrained Least Squares
    Abondance = abundance_maps.amaps.UCLS(M, U)

abond_unmix_4D = np.reshape(Abondance,(Nl,Nh,Nz,Nm))

#%% Separate fluorophores simulating two band-pass filters
Nf = 2 # number of filter

filter_list = ['green','red']
l_green_1, l_green_2 =  6,  18 # green filter band
l_red_1,   l_red_2   = 99, 111 # red filter band

folder_filter = recon # 

print('-- filtering')

Filt_sep = np.zeros((Nf,Nc))
Filt_sep[0,l_green_1:l_green_2] = 1
Filt_sep[1,l_red_1:l_red_2] = 1

abond_filter_4D = M @ Filt_sep.T
abond_filter_4D = abond_filter_4D.reshape((Nl,Nh,Nz,Nf))

#%% save data
# Save abondance maps
from spyrit.misc.disp import add_colorbar, noaxis

save_tag = True

# unmixing
if save_tag:
    # create folder
    save_path = Path(load_path + '/Unmixing/' + folder_unmix)
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # save unmixed data    
    filename = 'abondance.npy'
    np.save(save_path / filename, abond_unmix_4D)
    
    # save members/spectra  
    filename = 'spectra.npy'
    np.save(save_path / filename, U)

# filtering
if save_tag:    
    # create folder
    save_path = Path(load_path + '/Filtering/' + folder_filter)
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # save unmixed data    
    filename = 'abondance.npy'
    np.save(save_path / filename, abond_filter_4D)
    
    # save members/spectra  
    filename = 'spectra.npy'
    np.save(save_path / filename, Filt_sep)

#%% Plot Unmixing
save_fig = True

if save_fig:
    # create folder
    save_path = Path(load_path + '/Unmixing/' + folder_unmix)
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
# Abondance maps
# loop over all z-slices
for z, t in enumerate(T_list):
    # Plot
    fig, axs = plt.subplots(1, 4, figsize=(10,5))
    
    # loop over all members/fluorophores
    for m, member in  enumerate(member_list):
        
        im = axs[m].imshow(abond_unmix_4D[:,:,z,m])
        axs[m].set_title(f'{member}')
        add_colorbar(im, 'bottom')
    
    noaxis(axs)
    
    if save_fig:
        filename = f'T{t}_abondance.png'
        print(filename)
        plt.savefig(save_path / filename, bbox_inches='tight', dpi=600)
        plt.close(fig)

# unmixing: spectra
fig, axs = plt.subplots()
axs.plot(L_lambda, U[0], 'r', label = member_list[0])
axs.plot(L_lambda, U[1], 'g', label = member_list[1])
axs.plot(L_lambda, U[2], 'b', label = member_list[2])
axs.plot(L_lambda, U[3], 'c', label = member_list[3])
axs.plot(L_lambda, Filt_notch, 'k', ls = 'dashed', label = 'notch filter')
axs.fill_between(L_lambda, Filt_notch, 1, color='k', alpha=.1)
axs.legend()
plt.xlabel('Wavelenght (nm)')
plt.ylabel('Intensity (normalized)')

if save_fig:       
    filename = 'spectra.png'
    print(filename)
    plt.savefig(save_path / filename, bbox_inches='tight', dpi=600)
    plt.close()

#%% Plot Filtering
save_fig = True

if save_fig:
    # create folder
    save_path = Path(load_path + '/Filtering/' + folder_filter)
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
        filename = f'T{t}_abondance.png'
        print(filename)
        plt.savefig(save_path / filename, bbox_inches='tight', dpi=600)
        plt.close(fig)
        
# Spectra
fig, axs = plt.subplots()
axs.plot(L_lambda, L_DsRed_exp, 'r', label = member_list[0])
axs.plot(L_lambda, L_EGFP, 'g', label = member_list[1])
axs.plot(L_lambda, U[2], 'b', label = member_list[2])
axs.plot(L_lambda, Filt_notch, 'k', ls = 'dashed', label = 'notch filter')
axs.fill_between(L_lambda, Filt_notch, 1, color='k', alpha=.1)
axs.plot(L_lambda, Filt_sep[0], 'g', ls = 'dashdot', label = 'green filter')
axs.fill_between(L_lambda, Filt_sep[0], 0, color='g', alpha=.1)
axs.plot(L_lambda, Filt_sep[1], 'r', ls = 'dashdot', label = 'red filter')
axs.fill_between(L_lambda, Filt_sep[1], 0, color='r', alpha=.1)
axs.legend()
plt.xlabel('Wavelenght (nm)')
plt.ylabel('Intensity (normalized)')

if save_fig:      
    filename = 'spectra.png'
    print(filename)
    plt.savefig(save_path / filename, bbox_inches='tight', dpi=600)
    plt.close()