# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:04:25 2023

@author: ducros
"""
import torch
import numpy as np
from spyrit.misc.statistics import stat_stl10, stat_2, stat_imagenet
from statistics_dev import data_loaders_ImageNet
from pathlib import Path
from spyrit.misc.walsh_hadamard import walsh2_matrix, walsh2
import matplotlib.pyplot as plt
from spyrit.misc.disp import imagepanel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Covariance of stl10 images
data_folder = '../../data'
save_folder = './stat/stl10'

for img_size in [32, 64]:
    print(f'Image size: {img_size}')
    
    for get_size in ['resize','ccrop', 'rcrop']:
        print(f'Transform: {get_size}')
        
        stat_folder = Path(save_folder + '_' + get_size)
        stat_stl10(stat_root = stat_folder, data_root = data_folder,
                   img_size = img_size, batch_size = 1024, get_size = get_size)
        
#%% Covariance of imgaeNet images
data_folder = '../../data/ILSVRC2012_v10102019'
save_folder = './stat/ILSVRC2012_v10102019'

for img_size in [16]:
    print(f'Image size: {img_size}')
    
    for get_size in ['ccrop']: #['resize','ccrop', 'rcrop']
        print(f'Transform: {get_size}')
        
        stat_folder = Path(save_folder + '_' + get_size)
        stat_imagenet(stat_root = stat_folder, data_root = data_folder,
                   img_size = img_size, batch_size = 1024, get_size = get_size)
        

#%% 32 x 32 images, imageNet, resize
data_folder = '../../data/ILSVRC2012_v10102019'
save_folder = Path('./stat/ILSVRC2012_v10102019_resize/')

dataloaders = data_loaders_ImageNet(data_folder, img_size=32, batch_size=1024)

stat_2(dataloaders['train'], device, save_folder)

#%% visualize a few images, resize
inputs, _ = next(iter(dataloaders['train']))
imagepanel(inputs[0,0,:,:], inputs[1,0,:,:], inputs[2,0,:,:], inputs[3,0,:,:])
plt.savefig(save_folder/'images_32x32.png')

#%% 64 x 64 images, ImageNet, resize
data_folder = '../../data/ILSVRC2012_v10102019'
save_folder = Path('./stat/ILSVRC2012_v10102019_resize/')

dataloaders = data_loaders_ImageNet(data_folder, img_size=64, batch_size=1024)

stat_2(dataloaders['train'], device, save_folder)

#%% visualize a few images, center crop
inputs, _ = next(iter(dataloaders['train']))
imagepanel(inputs[0,0,:,:], inputs[1,0,:,:], inputs[2,0,:,:], inputs[3,0,:,:])
plt.savefig(save_folder/'images_64x64.png')

#%% 32 x 32 images, imageNet, center crop
data_folder = '../../data/ILSVRC2012_v10102019'
save_folder = Path('./stat/ILSVRC2012_v10102019_ccrop/')

dataloaders = data_loaders_ImageNet(data_folder, img_size=32, batch_size=1024)

stat_2(dataloaders['train'], device, save_folder)

#%% visualize a few images, center crop
inputs, _ = next(iter(dataloaders['train']))
imagepanel(inputs[0,0,:,:], inputs[1,0,:,:], inputs[2,0,:,:], inputs[3,0,:,:])
plt.savefig(save_folder/'images_32x32.png')

#%% 64 x 64 images, ImageNet, center crop
data_folder = '../../data/ILSVRC2012_v10102019'
save_folder = Path('./stat/ILSVRC2012_v10102019_ccrop/')

dataloaders = data_loaders_ImageNet(data_folder, img_size=64, batch_size=1024)

stat_2(dataloaders['train'], device, save_folder)

#%% visualize a few images, center crop
inputs, _ = next(iter(dataloaders['train']))
imagepanel(inputs[0,0,:,:], inputs[1,0,:,:], inputs[2,0,:,:], inputs[3,0,:,:])
plt.savefig(save_folder/'images_64x64.png')

#%% 32 x 32 images, imageNet, random crop
from spyrit.misc.statistics import data_loaders_ImageNet
data_folder = '../../data/ILSVRC2012_v10102019'
save_folder = Path('./stat/ILSVRC2012_v10102019_rcrop/')

dataloaders = data_loaders_ImageNet(data_folder, img_size=32, batch_size=1024)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
stat_2(dataloaders['train'], device, save_folder)

#%% visualize a few images, random crop
inputs, _ = next(iter(dataloaders['train']))
imagepanel(inputs[0,0,:,:], inputs[1,0,:,:], inputs[2,0,:,:], inputs[3,0,:,:])
plt.savefig(save_folder/'images_32x32.png')

#%% 64 x 64 images, ImageNet, random crop
data_folder = '../../data/ILSVRC2012_v10102019'
save_folder = Path('./stat/ILSVRC2012_v10102019_rcrop/')

dataloaders = data_loaders_ImageNet(data_folder, img_size=64, batch_size=1024)

stat_2(dataloaders['train'], device, save_folder)

#%% visualize a few images, random crop
inputs, _ = next(iter(dataloaders['train']))
imagepanel(inputs[0,0,:,:], inputs[1,0,:,:], inputs[2,0,:,:], inputs[3,0,:,:])
plt.savefig(save_folder/'images_64x64.png')

    
#%% image domain vs Hadamard domain covariance matrices, stl10
N = 64

stat_folder = './stat/stl10_resize'
cov_file   = f'Cov_2_{N}x{N}.npy'
mean_file   = f'Average_2_{N}x{N}.npy'
mean_dir  = np.load(Path(stat_folder) / mean_file)
sigma_dir = np.load(Path(stat_folder) / cov_file)

stat_folder = '../../stat/stl10'
cov_file   = f'Cov_{N}x{N}.npy'
mean_file   = f'Average_{N}x{N}.npy'
mean_walsh  = np.load(Path(stat_folder) / mean_file)
sigma_walsh = np.load(Path(stat_folder) / cov_file)

H = walsh2_matrix(N)

diff = np.linalg.norm(mean_walsh - walsh2(mean_dir)) / np.linalg.norm(mean_walsh)
print(f'Mean difference: {diff}')
print(f'Mean (walsh): {mean_walsh[3,18:24]}')
print(f'Mean (direct): {walsh2(mean_dir)[3,18:25]}')

diff = np.linalg.norm(sigma_walsh - H @ sigma_dir @ H.T) / np.linalg.norm(sigma_walsh)
print(f'Covariance difference: {diff}')
print(f'Mean (walsh): {sigma_walsh[648,139:144]}')
print(f'Mean (direct): {(H @ sigma_dir @ H.T)[648,139:144]}')

#%% image domain vs Hadamard domain covariance matrices, imageNet
N = 64

stat_folder = './stat/ILSVRC2012_v10102019_resize'
cov_file   = f'Cov_2_{N}x{N}.npy'
mean_file   = f'Average_2_{N}x{N}.npy'
mean_dir  = np.load(Path(stat_folder) / mean_file)
sigma_dir = np.load(Path(stat_folder) / cov_file)

stat_folder = '../../stat/ILSVRC2012_v10102019'
cov_file   = f'Cov_{N}x{N}.npy'
mean_file   = f'Average_{N}x{N}.npy'
mean_walsh  = np.load(Path(stat_folder) / mean_file)
sigma_walsh = np.load(Path(stat_folder) / cov_file)

H = walsh2_matrix(N)

diff = np.linalg.norm(mean_walsh - walsh2(mean_dir)) / np.linalg.norm(mean_walsh)
print(f'Mean difference: {diff}')
print(f'Mean (walsh): {mean_walsh[3,18:24]}')
print(f'Mean (direct): {walsh2(mean_dir)[3,18:25]}')

diff = np.linalg.norm(sigma_walsh - H @ sigma_dir @ H.T) / np.linalg.norm(sigma_walsh)
print(f'Covariance difference: {diff}')
print(f'Covariance (walsh): {sigma_walsh[648,139:144]}')
print(f'Covariance (direct): {(H @ sigma_dir @ H.T)[648,139:144]}')
print('--')
print(f'Variance difference: {diff}')
print(f'Variance (walsh): {sigma_walsh[139:143,139:143]}')
print(f'Variance (direct): {(H @ sigma_dir @ H.T)[139:143,139:143]}')

#%% center crop vs resize, ImageNet
import matplotlib.pyplot as plt
from spyrit.misc.disp import imagesc

N = 64

stat_folder = './stat/ILSVRC2012_v10102019_resize'
cov_file   = f'Cov_2_{N}x{N}.npy'
sigma_1 = np.load(Path(stat_folder) / cov_file)

stat_folder = './stat/ILSVRC2012_v10102019_ccrop'
cov_file   = f'Cov_2_{N}x{N}.npy'
sigma_2 = np.load(Path(stat_folder) / cov_file)

stat_folder = './stat/ILSVRC2012_v10102019_rcrop'
cov_file   = f'Cov_2_{N}x{N}.npy'
sigma_3 = np.load(Path(stat_folder) / cov_file)

imagesc(sigma_1[500,:].reshape(N,N), 'resize')
imagesc(sigma_2[500,:].reshape(N,N), 'center crop')
imagesc(sigma_3[500,:].reshape(N,N), 'random crop')
imagesc(sigma_1[300,:].reshape(N,N), 'resize')
imagesc(sigma_2[300,:].reshape(N,N), 'center crop')
imagesc(sigma_3[300,:].reshape(N,N), 'random crop')

#%% STL vs ImageNet
import matplotlib.pyplot as plt
from spyrit.misc.disp import imagesc

N = 32

stat_folder = './stat/stl10_ccrop'
cov_file   = f'Cov_2_{N}x{N}.npy'
sigma_1 = np.load(Path(stat_folder) / cov_file)

stat_folder = './stat/ILSVRC2012_v10102019_ccrop'
cov_file   = f'Cov_2_{N}x{N}.npy'
sigma_2 = np.load(Path(stat_folder) / cov_file)

imagesc(sigma_1[500,:].reshape(N,N), 'stl10, center crop')
imagesc(sigma_2[500,:].reshape(N,N), 'imageNet, center crop')
imagesc(sigma_1[300,:].reshape(N,N), 'stl10, center crop')
imagesc(sigma_2[300,:].reshape(N,N), 'imageNet, center crop')

#%% stl10 vs ImageNet covariance matrices, all resized
N = 64

stat_folder = './stat/ILSVRC2012_v10102019_resize'
cov_file   = f'Cov_2_{N}x{N}.npy'
mean_file   = f'Average_2_{N}x{N}.npy'
mean_imagenet  = np.load(Path(stat_folder) / mean_file)
sigma_imagenet = np.load(Path(stat_folder) / cov_file)

stat_folder = './stat/stl10_resize'
cov_file   = f'Cov_2_{N}x{N}.npy'
mean_file   = f'Average_2_{N}x{N}.npy'
mean_stl10  = np.load(Path(stat_folder) / mean_file)
sigma_stl10 = np.load(Path(stat_folder) / cov_file)

print(f'Mean (stl10): {mean_stl10[3,18:24]}')
print(f'Mean (imagenet): {walsh2(mean_imagenet)[3,18:25]}')

print(f'Covariance (stl10): {sigma_stl10[648,139:144]}')
print(f'Covariance (imagenet): {sigma_imagenet[648,139:144]}')

print(f'Variance (stl10):\n {sigma_stl10[139:143,139:143]}')
print(f'Variance (imagenet):\n {sigma_imagenet[139:143,139:143]}')

imagesc(sigma_stl10[500,:].reshape(N,N), 'stl10')
imagesc(sigma_imagenet[500,:].reshape(N,N), 'imageNet')
imagesc(sigma_stl10[300,:].reshape(N,N), 'stl10')
imagesc(sigma_imagenet[300,:].reshape(N,N), 'imageNet')