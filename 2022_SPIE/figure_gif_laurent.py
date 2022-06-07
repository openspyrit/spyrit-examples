# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'


#%%
from spas import *
from matplotlib import pyplot as plt
import os
import numpy as np

import spyrit.misc.walsh_hadamard as wh
import spas.transfer_data_to_girder as transf
from spas import plot_spec_to_rgb_image as plt_rgb
from spas.acquisition import *
from spas.acquisition import setup_cam
from spas.visualization import snapshotVisu
from pyueye import ueye
import time    
from pathlib import Path
from PIL import Image
import imageio


#%% Reconstruction without NN
########################## to be change ############################
#spc/data/setup_v1.3/2022-01-14_SeimensStar_LinearColorFilter/SeimensStar_whiteLamp_linear_color_filter
setup_version = 'setup_v1.3'
data_folder_name = '2022-03-11_Cat'#'2021-07-26-spectral-analysis'
data_name = 'Cat_LinearColoredFilter'#'Cat_whiteLamp'#'colored-siemens'
########################### path ###################################
data_path = '../data/' + data_folder_name + '/' + data_name + '/' + data_name
had_reco_path = data_path + '_had_reco.npz'
overview_path = '../data/' + data_folder_name + '/overview'
gif_path = '../data/' + data_folder_name + '/' + data_name + '/gif/'
gif_temp_path = gif_path+'temp/'
gif_temp2_path = gif_path+'temp2/'

if not os.path.exists(overview_path):
    os.makedirs(overview_path)
    
fig_had_reco_path = overview_path + '/' + 'HAD_RECO_' + data_name   
nn_reco_path = overview_path + '/' + 'NN_RECO_' + data_name
########################## read raw data ###########################
file = np.load(data_path+'_spectraldata.npz')
M = file['spectral_data']#['arr_0']#['spectral_data']
Q = wh.walsh2_matrix(64)

metadata_path = data_path + '_metadata.json'
metadata, acquisition_parameters, spectrometer_parameters, DMD_parameters = read_metadata(metadata_path)

# subsampling
nsub = 1
M_sub = M[:8192//nsub,:]
acquisition_parameters.patterns_sub = acquisition_parameters.patterns[:8192//nsub]

GT = reconstruction_hadamard(acquisition_parameters.patterns, 'walsh', Q, M)
GT = np.rot90(GT, 1)
GT = np.rot90(GT, 1)
# np.savez_compressed(had_reco_path, GT)

plt.figure
plt.imshow(np.sum(GT, axis = 2))
plt.savefig(gif_path+'image_sum/'+'hadamard_sum.png')
plt.show()

metadata, acquisition_metadata, spectrometer_params, DMD_params = read_metadata(data_path+'_metadata.json')
wavelengths = acquisition_metadata.wavelengths

######### Plot the 8 slices in one figure / RAW(Hadamar reco)
F_bin8, wavelengths_bin8, bin_width = spectral_slicing(GT.T, acquisition_parameters.wavelengths, 515, 660, 8)
F_bin_rot8 = np.rot90(F_bin8, axes=(1,2))
F_bin_flip8 = F_bin_rot8[:,::-1,:]

plot_color(F_bin_flip8, wavelengths_bin8)
plt.savefig(gif_temp2_path+'em/'+'0_hadamard.png')
plt.savefig(gif_temp2_path+'mmse/'+'0_hadamard.png')
plt.show()

######### Plot slice per slice / RAW(Hadamar reco)
F_bin, wavelengths_bin, bin_width = spectral_slicing(GT.T, acquisition_parameters.wavelengths, 515, 660, 256)
F_bin_rot = np.rot90(F_bin, axes=(1,2))
F_bin_flip = F_bin_rot[:,::-1,:]

bin_ = F_bin_flip.shape[0]
img_size = GT.shape[2]
gamma = 0.8
ims = []

for bin_num in range(bin_):
    if 1==1:#mod(bin_num, 50) == 0:
        color = generate_colormap(wavelengths_bin[bin_num], img_size, gamma)
        colormap = ListedColormap(color)
        
        plt.figure
        plt.imshow(F_bin_flip[bin_num,:,:], cmap = colormap)
        plt.axis('off')
        plt.title('\u03BB' + '=' + str(round(wavelengths_bin[bin_num])) + 'nm')
        #plt.show()
        im = gif_temp_path+f'{bin_num}.png'
        ims.append(im)
        plt.savefig(im)
        plt.close()

fps = 50
with imageio.get_writer(gif_path+'Hadamard.gif', mode='I', fps = fps) as writer:
    for im in ims:
        image = imageio.imread(im)
        writer.append_data(image)
        
for im in set(ims):
    os.remove(im)
#%% NN Reco
folderpath = '../data/reco_Antonio/results/Cat/'
#filename = 'cat_em_net_2048.npy'
ff = os.listdir(folderpath)

for filename in ff:
#     if filename.find('mmse') >= 0:
#         if filename.find('512') >= 0:
#             titre = 'CR = 1/4'
#             save_title = 'mmse/3_'
#         elif filename.find('1024') >= 0:
#             titre = 'CR = 1/2'
#             save_title = 'mmse/2_'
#         elif filename.find('2048') >= 0:
#             titre = 'CR = 1/1'
#             save_title = 'mmse/1_'  
#     elif filename.find('em') >= 0:        
#         if filename.find('512') >= 0:
#             titre = 'CR = 1/4'
#             save_title = 'em/3_'
#         elif filename.find('1024') >= 0:
#             titre = 'CR = 1/2'
#             save_title = 'em/2_'
#         elif filename.find('2048') >= 0:
#             titre = 'CR = 1/1'
#             save_title = 'em/1_'
    
    cube = np.load(folderpath+filename)
    cube = np.rot90(cube, axes=(1,2))
    cube = np.rot90(cube, axes=(1,2))
    
    plt.figure
    plt.imshow(np.sum(cube, axis = 0))   
    plt.title(titre)
    plt.savefig(gif_path+'image_sum/'+filename[0:-4]+'_sum.png')
    plt.show()
    
    ######### Plot the 8 slices in one figure / NN reco
    F_bin8, wavelengths_bin8, bin_width = spectral_slicing(cube, acquisition_parameters.wavelengths, 515, 660, 8)
    F_bin_rot8 = np.rot90(F_bin8, axes=(1,2))
    F_bin_flip8 = F_bin_rot8[:,::-1,:]
    
    plot_color(F_bin_flip8, wavelengths_bin8)
    plt.savefig(gif_temp2_path+save_title+filename[0:-4]+'.png')      
            
    #plt.show()
    
    ######### Plot slice per slice / NN reco
    F_bin_cube, wavelengths_bin, bin_width = spectral_slicing(cube, acquisition_parameters.wavelengths, 515, 660, 256)
    
    bin_ = F_bin_cube.shape[0]
    img_size = cube.shape[2]
    gamma = 0.8
    
    for bin_num in range(bin_):
        1
        color = generate_colormap(wavelengths_bin[bin_num], img_size, gamma)
        colormap = ListedColormap(color)
        
        plt.figure
        plt.imshow(F_bin_cube[bin_num,:,:], cmap = colormap)
        plt.axis('off')
        plt.title('\u03BB' + '=' + str(round(wavelengths_bin[bin_num])) + 'nm')
        #plt.show()
        im = gif_temp_path+f'{bin_num}.png'
        ims.append(im)
        plt.savefig(im)
        plt.close()
        
    fps = 50
    with imageio.get_writer(gif_path+filename[0:-4]+'.gif', mode='I', fps = fps) as writer:
        for im in ims:
            image = imageio.imread(im)
            writer.append_data(image)

    for im in set(ims):
        os.remove(im)


# ff = os.listdir(gif_temp2_path+'em/')

# fps = 1
# with imageio.get_writer(gif_path+'em.gif', mode='I', fps = fps) as writer:
#     for im in ff:
#         image = imageio.imread(gif_temp2_path+'em/'+im)
#         writer.append_data(image)

# plt.figure
# plt.imshow(np.sum(GT, axis = 2))
# plt.show()

# subsampling
# =============================================================================
# N = 64
# nsub = 2
# M_sub = M[:8192//nsub,:]
# acquisition_parameters.patterns_sub = acquisition_parameters.patterns[:8192//nsub]
# GT_sub = reconstruction_hadamard(acquisition_parameters.patterns_sub, 'walsh', Q, M_sub)
# F_bin_sub, wavelengths_bin, bin_width = spectral_binning(GT_sub.T, acquisition_parameters.wavelengths, 530, 730, 8)
# 
# 
# 
# plot_color(F_bin_sub, wavelengths_bin)
# plt.savefig(fig_had_reco_path + '_wavelength_binning_subsamplig=' + str(nsub) + '.png')
# plt.show()
# =============================================================================

# =============================================================================
# plt.figure
# plt.imshow(GT[:,:,0])
# plt.title(f'lambda = {wavelengths[0]:.2f} nm')
# plt.savefig(fig_had_reco_path + '_' + f'lambda = {wavelengths[0]:.2f} nm.png')
# plt.show()
# 
# plt.figure
# plt.imshow(GT[:,:,410])
# plt.title(f'lambda = {wavelengths[410]:.2f} nm')
# plt.savefig(fig_had_reco_path + '_' + f'lambda = {wavelengths[410]:.2f} nm.png')
# plt.show()
# 
# plt.figure
# plt.imshow(GT[:,:,820])
# plt.title(f'lambda = {wavelengths[820]:.2f} nm')
# plt.savefig(fig_had_reco_path + '_' + f'lambda = {wavelengths[820]:.2f} nm.png')
# plt.show()
# 
# plt.figure
# plt.imshow(GT[:,:,1230])
# plt.title(f'lambda = {wavelengths[1230]:.2f} nm')
# plt.savefig(fig_had_reco_path + '_' + f'lambda = {wavelengths[1230]:.2f} nm.png')
# plt.show()
# 
# plt.figure
# plt.imshow(GT[:,:,2047])
# plt.title(f'lambda = {wavelengths[2047]:.2f} nm')
# plt.savefig(fig_had_reco_path + '_' + f'lambda = {wavelengths[2047]:.2f} nm.png')
# plt.show()
# 
# plt.figure
# plt.imshow(np.sum(GT,axis=2))
# plt.title('Sum of all wavelengths')
# plt.savefig(fig_had_reco_path + '_sum_of_wavelengths.png')
# plt.show()
# 
# plt.figure
# plt.scatter(wavelengths, np.mean(np.mean(GT,axis=1),axis=0))
# plt.grid()
# plt.xlabel('Lambda (nm)')
# plt.title('Spectral view in the spatial mean')
# plt.savefig(fig_had_reco_path + '_spectral_axe_of_the_hypercube.png')
# plt.show()
# 
# indx = np.where(GT == np.max(GT))
# sp =  GT[indx[0],indx[1],:]
# plt.figure
# plt.scatter(wavelengths, sp.T)
# plt.grid()
# plt.xlabel('Lambda (nm)')
# plt.title('Spectral view of the max intensity')
# plt.savefig(fig_had_reco_path + '_spectral_axe_of_max_intensity.png')
# plt.show()
# =============================================================================
#%% Reconstruct with NN
network_params = ReconstructionParameters(
    img_size=64,
    CR=1024,
    denoise=True,
    epochs=40,
    learning_rate=1e-3,
    step_size=20,
    gamma=0.2,
    batch_size=256,
    regularization=1e-7,
    N0=50.0,
    sig=0.0,
    arch_name='c0mp',)

cov_path = '../stats/new-nicolas/Cov_64x64.npy'
mean_path = '../stats/new-nicolas/Average_64x64.npy'
H_path = '../stats/new-nicolas/H.npy'
model_root = '../models/new-nicolas/'

model, device = setup_reconstruction(cov_path, mean_path, H_path, model_root, network_params)
noise = load_noise('../noise-calibration/fit_model2.npz')

reconstruction_params = {
    'model': model,
    'device': device,
    'batches': 1,
    'noise': noise,
}

F_bin, wavelengths_bin, bin_width, noise_bin = spectral_binning(M.T, acquisition_parameters.wavelengths, 530, 730, 8, 0, noise)
recon = reconstruct(model, device, F_bin[0:8192//4,:], 1, noise_bin)            
plot_color(recon, wavelengths_bin)
plt.savefig(nn_reco_path + '_reco_wavelength_binning.png')
plt.show()

#%% transfer data to girder
transf.transfer_data_to_girder(metadata, acquisition_parameters, spectrometer_params, DMD_params, setup_version, data_folder_name, data_name)
#%% delete plots
Question = input("Do you want to delete the figures yes [y] ?  ")
if Question == ("y") or Question == ("yes"):        
    shutil.rmtree(overview_path)
    print ("==> figures deleted")










