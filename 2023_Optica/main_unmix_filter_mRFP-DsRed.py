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
DsRed_exp_path = "./data/Reference_spectra//DsRed_express_fpbase_spectra_550_658.csv" 
mCherry_path = "./data/Reference_spectra//mCherry_fpbase_spectra_550_658.csv"
mRFP_path =  "./data/Reference_spectra//mRFP1_fpbase_spectra_550_658.csv"

data_DsRed_exp = np.genfromtxt(DsRed_exp_path, delimiter=';',skip_header=True)#,
data_mCherry = np.genfromtxt(mCherry_path, delimiter=';',skip_header=True)
data_mRFP = np.genfromtxt(mRFP_path, delimiter=';',skip_header=True)

Nc = 128 # number of channels

# Interpolate spectra to have Nc channels
from scipy import interpolate

L_lambda = np.linspace(550, 658, Nc)
f = interpolate.interp1d(data_mRFP[:,0], data_mRFP[:,1], kind='cubic')
L_mRFP = f(L_lambda)   
f = interpolate.interp1d(data_DsRed_exp[:,0], data_DsRed_exp[:,1], kind='cubic')
L_DsRed_exp = f(L_lambda)

# plot
fig, axs = plt.subplots()
axs.plot(L_lambda, L_DsRed_exp, label='DsRed') 
axs.plot(L_lambda, L_mRFP, label='mRFP')
axs.set_xlabel('wavelength (nm)')
axs.set_title('Spectra used for unmixing')
axs.legend()

#%%
from pathlib import Path

# Load data
Nl = 512 # number of pixcels along the y dimensions 
Nh = 512 # number of measured Walsh_Hadmard coefficients (correpond to the h dimensions)
Nc = 128 # number of channels

T_list = [*range(4, 8), *range(9, 25)] # slice indices, Run0008 corrupted
load_path = './data/2023_02_28_mRFP_DsRed_3D'
recon = 'tikhonet50_div1.5'  # 'pinv' 'tikhonet50_div1.5'

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
x1,x2 = 235-50,285-50
y1,y2 = 320,345
z1,z2 = 0,4  # ['RUN0004','RUN0005','RUN0006']

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
    

#%% Unmixing
from pysptools import abundance_maps

unmixing_folder =  '/Unmixing_3/'
member_list = ['DsRed','mRFP','Autofluo']  # member_list = ['DsRed','mRFP','Autofluo','Noise'] 

method_unmix = 'NNLS' # 'NNLS''_UCLS'
Nm = 4
Nl,Nh = xyzl_cube.shape[:2] # new shape after registration

# End member (spectra) matrix
U = np.zeros((Nm,Nc))
U[0] = L_DsRed_exp
U[1] = L_mRFP 
U[2] = spec_fluo/np.max(spec_fluo) # spectre d'autofluo normalisé
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

filtering_folder = '/Filtering_3/'
filter_list = ['orange','red']
l_green_1, l_green_2 =  40, 52 # green filter band
l_red_1,   l_red_2   = 62, 74 # red filter band

filtering_subfolder = recon # 

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
            plt.savefig(save_path / filename, bbox_inches='tight', dpi=600)
        plt.close(fig)

# unmixing: spectra
col = ['r', 'g', 'b', 'c']  # color list

fig, axs = plt.subplots()
for m in range(len(member_list)):
    axs.plot(L_lambda, U[m], col[m], label = member_list[m])
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
axs.plot(L_lambda, L_DsRed_exp, 'r', label = member_list[0])
axs.plot(L_lambda, L_mRFP, 'g', label = member_list[1])
axs.plot(L_lambda, U[2], 'b', label = member_list[2])
axs.plot(L_lambda, Filt_sep[0], 'g', ls = 'dashdot', label = 'green filter')
axs.fill_between(L_lambda, Filt_sep[0], 0, color='g', alpha=.1)
axs.plot(L_lambda, Filt_sep[1], 'r', ls = 'dashdot', label = 'red filter')
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