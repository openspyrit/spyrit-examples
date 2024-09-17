# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 11:19:23 2023

@author: ducros
"""
# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt

#%% Load reference spectrum
DsRed_path = "./data/Reference_spectra//DsRed2_fpbase_spectra.csv" 
mRFP_path =  "./data/Reference_spectra//mRFP1_fpbase_spectra.csv"

data_DsRed = np.genfromtxt(DsRed_path, delimiter=',',skip_header=True)#,
data_mRFP = np.genfromtxt(mRFP_path, delimiter=';',skip_header=True)

mRFP_path =  "./data/Reference_spectra//mRFP12_fpbase_spectra.csv"
data_mRFP = np.genfromtxt(mRFP_path, delimiter=',',skip_header=True)

suffix = 'calib_mrfp12'

Nc = 128 # number of channels

# Interpolate spectra to have Nc channels
from scipy import interpolate

#L_lambda = np.linspace(550, 658, Nc)
L_lambda = np.linspace(538, 660, Nc)
f = interpolate.interp1d(data_mRFP[:,0], data_mRFP[:,1], kind='cubic')
L_mRFP = f(L_lambda)   
f = interpolate.interp1d(data_DsRed[:,0], data_DsRed[:,1], kind='cubic')
L_DsRed = f(L_lambda)


#%%
from pathlib import Path

# Load data
Nl = 512 # number of pixcels along the y dimensions 
Nh = 512 # number of measured Walsh_Hadmard coefficients (correpond to the h dimensions)
Nc = 128 # number of channels

T_list = [*range(4, 8), *range(9, 25)] # slice indices, Run0008 corrupted
load_path = './data/2023_02_28_mRFP_DsRed_3D'
recon = 'pinv'  # 'pinv' 'tikhonet50_div1.5' 'pinv_shift' 'tikhonet50_div1.5_shift'

# all slices are unmixed jointly!
Nz = len(T_list)
xyzl_cube = np.zeros((Nl,Nh,Nz,Nc))

# loop over z-slices
for z, t in enumerate(T_list):
    Run = f'RUN{t:04}'

    # Load
    filename = f'{Run}_rec_{recon}_exp_{Nl}x{Nh}x{Nc}.npy'
    print(filename)
    xyl_cube = np.load(Path(load_path) / 'Reconstruction/hypercube'/ recon / filename)
    
    xyzl_cube[:,:,z,:] = xyl_cube
    
# Replace negative values by zeros
# xyzl_cube = np.where(xyzl_cube > 0, xyzl_cube, 0)

#%% Estimate auto-fluorescence spectrum
from matplotlib.patches import Rectangle

# Area where the autofluorescence is estimated, which corresponds to the "sac vitellin"
x1,x2 = 235-30,285-30
y1,y2 = 320,345
z1,z2 = 0,4  # ['RUN0004','RUN0005','RUN0006']

L_fluo = np.mean(xyzl_cube[x1:x2, y1:y2, z1:z2,:], axis=(0,1,2))
L_fluo /= np.max(L_fluo)

# plot autofluorescence area
for z in range(z1,z2):
    fig, axs = plt.subplots()
    axs.imshow(xyzl_cube[:,:,z,64])
    axs.set_title(f'Autofluorescence estimation area; x in [{x1}-{x2}]; y in [{y1}-{y2}]')
    axs.set_xlabel('y-axis')
    axs.set_ylabel('x-axis')
    # Create a Rectangle patch
    axs.add_patch(Rectangle((y1, x1), y2-y1+1, x2-x1+1, linewidth=1, edgecolor='r', facecolor='none'))
    
    
#%% Estimate fluorescence spectra
from matplotlib.patches import Rectangle

z = 14  #window
dx = 3  #window

# Area where the mrfp spectrum is estimated
x1_r,x2_r = 269-dx,269+dx
y1_r,y2_r = 380-dx,380+dx

# Area where the dsred spectrum is estimated
x1_g,x2_g = 414-dx,414+dx
y1_g,y2_g = 401-dx,401+dx

# Integrate
L_mRFP_pure = np.mean(xyzl_cube[x1_r:x2_r, y1_r:y2_r, z,:], axis=(0,1))
L_DsRed_pure = np.mean(xyzl_cube[x1_g:x2_g, y1_g:y2_g, z,:], axis=(0,1))
L_mRFP_pure /= L_mRFP_pure.max() 
L_DsRed_pure /= L_DsRed_pure.max() 

# plot area
fig, axs = plt.subplots(1, 2)
axs[0].imshow(xyzl_cube[:,:,z,64])
axs[0].set_xlabel('y-axis')
axs[0].set_ylabel('x-axis')
# Create a Rectangle patch
axs[0].add_patch(Rectangle((y1_r, x1_r), y2_r-y1_r+1, x2_r-x1_r+1, linewidth=2, edgecolor='r', facecolor='none'))
axs[0].add_patch(Rectangle((y1_g, x1_g), y2_g-y1_g+1, x2_g-x1_g+1, linewidth=2, edgecolor='g', facecolor='none'))
    
# Plot spectra
axs[1].plot(L_lambda, L_DsRed_pure, 'm', label='DsRed (expe)')
axs[1].plot(L_lambda, L_DsRed, '--m',    label='DsRed (theo)') 
axs[1].plot(L_lambda, L_mRFP_pure, 'r',  label='mRFP (expe)')
axs[1].plot(L_lambda, L_mRFP, '--r',     label='mRFP (theo)') 
axs[1].legend()

fig.suptitle(f'DsRed estimation area; x in [{x1_r}-{x2_r}]; y in [{y1_r}-{y2_r}]')
    

#%% Unmixing
from pysptools import abundance_maps

unmixing_folder =  '/Unmixing_' + suffix + '/'
member_list = ['DsRed','mRFP','Autofluo']  # member_list = ['DsRed','mRFP','Autofluo','Noise'] 

method_unmix = 'NNLS' # 'NNLS''_UCLS'
Nm = 4
Nl,Nh = xyzl_cube.shape[:2] # new shape after registration

# End member (spectra) matrix
U = np.zeros((Nm,Nc))
U[0] = L_DsRed
U[1] = L_mRFP 
U[2] = L_fluo # spectre d'autofluo normalisé
U[3] = np.ones(128)*1/10 

unmixing_subfolder = recon + '_' + method_unmix

print('-- unmixing')

# measurment matrix with all pixels across rows
M = np.reshape(xyzl_cube, (-1,Nc)) 

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

filtering_folder = '/Filtering' + '_' + suffix + '/'
filter_list = ['orange','red']
l_green_1, l_green_2 =  40, 52 # green filter band
l_red_1,   l_red_2   = 62, 74 # red filter band

c_dsred = 570  # in nm
c_mrfp = 620    # in nm
bandwidth = 10 # in nm

filtering_subfolder = recon # 

print('-- filtering')


Filt_sep = np.zeros((Nf,Nc))
Filt_sep[0] = np.where((L_lambda < c_dsred + bandwidth/2)&(L_lambda > c_dsred - bandwidth/2), 
                    1, 0)
Filt_sep[1] = np.where((L_lambda < c_mrfp + bandwidth/2)&(L_lambda > c_mrfp - bandwidth/2), 
                    1, 0)

# Filt_sep = np.zeros((Nf,Nc))
# Filt_sep[0,l_green_1:l_green_2] = 1
# Filt_sep[1,l_red_1:l_red_2] = 1

abond_filter_4D = M @ Filt_sep.T
abond_filter_4D = abond_filter_4D.reshape((Nl,Nh,Nz,Nf))

#%% save data
# Save abondance maps
from spyrit.misc.disp import add_colorbar, noaxis

save_tag = True

# unmixing
if save_tag:
    # create folder
    save_path = Path(load_path + unmixing_folder + unmixing_subfolder)
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
    save_path = Path(load_path + filtering_folder + filtering_subfolder)
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
save_ext = ['svg'] # ['pdf', 'png']

if save_fig:
    # create folder
    save_path = Path(load_path + unmixing_folder + unmixing_subfolder)
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
col = ['m', 'r', (1,0.5, 0)]  # color list

fig, axs = plt.subplots()
for m in range(len(member_list)):
    axs.plot(L_lambda, U[m], color=col[m], label = member_list[m])
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

save_ext = ['svg'] # ['pdf', 'png']
filename = 'autofluorescence'
for ext in save_ext:
    filename_with_ext = filename + '.' + ext
    print(filename_with_ext)
    plt.savefig(save_path / filename_with_ext, bbox_inches='tight', dpi=600)
plt.close(fig)


#%% Plot Filtering
save_fig = True
save_ext = ['svg'] # ['pdf', 'png']

if save_fig:
    # create folder
    save_path = Path(load_path + filtering_folder + filtering_subfolder)
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
fig, axs = plt.subplots()
axs.plot(L_lambda, L_DsRed, 'm', label = member_list[0])
axs.plot(L_lambda, L_mRFP, 'r', label = member_list[1])
axs.plot(L_lambda, U[2], color=(1,0.5, 0), label = member_list[2])
axs.plot(L_lambda, Filt_sep[0], 'm', ls = 'dashdot', label = 'DsRed filter')
axs.fill_between(L_lambda, Filt_sep[0], 0, color='g', alpha=.1)
axs.plot(L_lambda, Filt_sep[1], 'r', ls = 'dashdot', label = 'mRFP filter')
axs.fill_between(L_lambda, Filt_sep[1], 0, color='r', alpha=.1)
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