# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 08:48:35 2022

This scripts reconstructs the images in Figure 8

NB (15-Sep-22): to debug needs to run
import collections
collections.Callable = collections.abc.Callable

"""

#%%
import os
import torch
import numpy as np
import math
from matplotlib import pyplot as plt
from pathlib import Path
# get debug in spyder
import collections
collections.Callable = collections.abc.Callable

from spyrit.misc.statistics import Cov2Var
from spyrit.core.noise import Poisson 
from spyrit.core.meas import HadamSplit
from spyrit.core.prep import SplitPoisson
from spyrit.core.recon import DCNet, PinvNet
from spyrit.core.train import load_net
from spyrit.core.nnet import Unet
from spyrit.misc.sampling import reorder, Permutation_Matrix
from spyrit.misc.disp import add_colorbar, noaxis


from spas import read_metadata, spectral_slicing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Torch device: {device}')

#%% user-defined
# used for acquisition
N_acq = 64

# for reconstruction
N_rec = 128  # 128 or 64
M_list = [4096] #[4096, 1024, 512] # for N_rec = 128
#M_list = [4095, 1024, 512] # for N_rec = 64

N0 = 10     # Check if we used 10 in the paper
stat_folder_rec = Path('../../stat/oe_paper/') # Path('../../stat/ILSVRC2012_v10102019/')

net_arch    = 'dc-net'      # ['dc-net','pinv-net']
net_denoi   = 'unet'        # ['unet', 'cnn']
net_data    = 'imagenet'    # 'imagenet'
bs = 256
 
save_root = Path('../../recon/')

# Select methods for reconstruction
# Pinv
mode_pinv = False
# Pinv-UNet
mode_pinv_unet = False
model_pinvnet_path = "../../model"    
name_pinvnet = 'pinv-net_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07'
# DCNet
mode_dcnet = False
# DRUNet
mode_pinvnet_drunet = True
if mode_pinvnet_drunet:
    #from spyrit.external.drunet import DRUNet
    from drunet import DRUNet
    noise_level = 30
    model_drunet_path = "../../model"
    name_drunet = 'drunet_gray.pth'

#%% Parameters for simulated images 
mode_sim = False
if mode_sim:
    path_natural_images = '../../images'
    img_max = 1
    img_min = -1

    import torchvision
    def transform_gray_norm(img_size, crop_type = 'center'): 
        """ 
        Args:
            img_size=int, image size
        
        Create torchvision transform for natural images (stl10, imagenet):
        convert them to grayscale, then to tensor, and normalize between [-1, 1]
        """

        if crop_type=='center':
            transforms_resize = [torchvision.transforms.Resize(img_size), 
                                torchvision.transforms.CenterCrop(img_size)]                           

        transform = torchvision.transforms.Compose(
            [torchvision.transforms.functional.to_grayscale,
            *transforms_resize,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])])
        return transform

    # Create dataset and loader (expects class folder 'images/test/')
    #from spyrit.misc.statistics import transform_gray_norm
    transform = transform_gray_norm(N_rec)
    dataset = torchvision.datasets.ImageFolder(root=path_natural_images, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 7)

    x, _ = next(iter(dataloader))
    print(f'Shape of input images: {x.shape}')

    # Select image
    img_id = 4
    x = x[img_id:img_id+1,:,:,:]
    x = x.detach().clone()
    b,c,h,w = x.shape

    # plot
    x_plot = x.view(-1,h,h).cpu().numpy() 
    fig = plt.figure(figsize=(7,7))
    plt.imshow(x_plot[0,:,:], cmap='gray', vmin=img_min, vmax=img_max)
    plt.axis('off')

#%% covariance matrix and network filnames
if N_rec==64:
    cov_rec_file= stat_folder_rec/ ('Cov_{}x{}'.format(N_rec, N_rec)+'.npy')
elif N_rec==128:
    cov_rec_file= stat_folder_rec/ ('Cov_8_{}x{}'.format(N_rec, N_rec)+'.npy')
    
#%% Networks
for M in M_list:
    
    if (N_rec == 128) and (M == 4096):
        net_order   = 'rect'
    else:
        net_order   = 'var'

    net_suffix  = f'N0_{N0}_N_{N_rec}_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_{bs}_reg_1e-07_light'
    
    #%% Init and load trained network
    # Covariance in hadamard domain
    Cov_rec = np.load(cov_rec_file)
    
    # Sampling order
    if net_order == 'rect':
        Ord_rec = np.ones((N_rec, N_rec))
        n_sub = math.ceil(M**0.5)
        Ord_rec[:,n_sub:] = 0
        Ord_rec[n_sub:,:] = 0
        
    elif net_order == 'var':
        Ord_rec = Cov2Var(Cov_rec)
        
    # Init network  
    meas = HadamSplit(M, N_rec, Ord_rec)
    noise = Poisson(meas, N0) # could be replaced by anything here as we just need to recon
    prep  = SplitPoisson(N0, meas)    
    denoi = Unet()
    denoi_pinv = Unet()
    if mode_dcnet:
        model = DCNet(noise, prep, Cov_rec, denoi)
        
        # Load trained DC-Net
        net_title = f'{net_arch}_{net_denoi}_{net_data}_{net_order}_{net_suffix}'
        title = '../../model/oe_paper/' + net_title
        load_net(title, model, device, strict = False)
        model.eval()                    # Mandantory when batchNorm is used

        #model.prep.set_expe()
        model.to(device)
    
    if mode_pinv:
        model_pinv = PinvNet(noise, prep)
        model_pinv.to(device)        
        
    if mode_pinv_unet:
        # Load trained Pinv-Net
        model_pinv_unet = PinvNet(noise, prep, denoi_pinv)
        load_net(os.path.join(model_pinvnet_path, name_pinvnet), model_pinv_unet, device, strict = False)
        model_pinv_unet.eval()      
        model_pinv_unet.to(device)

    if mode_pinvnet_drunet:
        # DRUNet(noise_level=5, n_channels=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose')
        denoi_drunet = DRUNet()#noise_level=noise_level, n_channels=1)

        # Set the device for DRUNet
        denoi_drunet = denoi_drunet.to(device)
        # Load pretrained weights
        denoi_drunet.load_state_dict(torch.load(os.path.join(model_drunet_path, name_drunet)), strict=False)               
        denoi_drunet.eval()

        # Define DCNet with DRUNet denoising
        model_pinvnet_drunet = PinvNet(noise, prep, denoi_drunet)
        model_pinvnet_drunet.to(device)


    #%% Load expe data and unsplit
    data_root = Path('../../data/')

    data_file_prefix_list = ['zoom_x12_usaf_group5',
                             'zoom_x12_starsector',
                             'tomato_slice_2_zoomx2',
                             'tomato_slice_2_zoomx12',
                             ]
    
    #%% Load data
    for data_file_prefix in data_file_prefix_list:
        
        print(Path(data_file_prefix) / data_file_prefix)
        
        # meta data
        meta_path = data_root / Path(data_file_prefix) / (data_file_prefix + '_metadata.json')
        _, acquisition_parameters, _, _ = read_metadata(meta_path)
        wavelengths = acquisition_parameters.wavelengths 
        
        # data
        full_path = data_root / Path(data_file_prefix) / (data_file_prefix + '_spectraldata.npz')
        raw = np.load(full_path)
        meas= raw['spectral_data']
        
        # reorder measurements to match with the reconstruction order
        Ord_acq = -np.array(acquisition_parameters.patterns)[::2]//2   # pattern order
        Ord_acq = np.reshape(Ord_acq, (N_acq,N_acq))                   # sampling map
        
        Perm_rec = Permutation_Matrix(Ord_rec)    # from natural order to reconstrcution order 
        Perm_acq = Permutation_Matrix(Ord_acq).T  # from acquisition to natural order
        meas = reorder(meas, Perm_acq, Perm_rec)
        
        #%% Reconstruct a single spectral slice from full reconstruction
        wav_min = 579 
        wav_max = 579.1
        wav_num = 1
        meas_slice, wavelengths_slice, _ = spectral_slicing(meas.T, 
                                                        wavelengths, 
                                                        wav_min, 
                                                        wav_max, 
                                                        wav_num)
        with torch.no_grad():
            m = torch.Tensor(meas_slice[:2*M,:]).to(device)
            
            if mode_dcnet:
                rec_gpu = model.reconstruct_expe(m)
                rec = rec_gpu.cpu().detach().numpy().squeeze()
            
                #%% Plot or save 
                # rotate
                #rec = np.rot90(rec,2)
                
                fig , axs = plt.subplots(1,1)
                im = axs.imshow(rec, cmap='gray')
                noaxis(axs)
                add_colorbar(im, 'bottom')
                
                full_path = save_root / (data_file_prefix + '_' + f'{M}_{N_rec}' + '.pdf')
                fig.savefig(full_path, bbox_inches='tight')        
        
        #%% pseudo inverse
        if M==4096:
            if mode_pinv:                    
                rec_pinv_gpu = model_pinv.reconstruct_expe(m)
                rec_pinv = rec_pinv_gpu.cpu().detach().numpy().squeeze()
            
                fig , axs = plt.subplots(1,1)
                im = axs.imshow(rec_pinv, cmap='gray')
                noaxis(axs)
                add_colorbar(im, 'bottom')
                
                full_path = save_root / (data_file_prefix + '_' + f'pinv_{N_rec}' + '.pdf')
                fig.savefig(full_path, bbox_inches='tight', dpi=600)

            #% pseudo inverse + unet   
            if mode_pinv_unet:
                with torch.no_grad():            
                    rec_pinv_unet_gpu = model_pinv_unet.reconstruct_expe(m)
                    rec_pinv_unet = rec_pinv_unet_gpu.cpu().detach().numpy().squeeze()
                
                    fig , axs = plt.subplots(1,1)
                    im = axs.imshow(rec_pinv_unet, cmap='gray')
                    noaxis(axs)
                    add_colorbar(im, 'bottom')
                    
                    full_path = save_root / (data_file_prefix + '_' + f'pinv_unet_{N_rec}' + '.pdf')
                    fig.savefig(full_path, bbox_inches='tight', dpi=600) 

            #% pseudo inverse + drunet
            if mode_pinvnet_drunet:
                # Set noise level
                denoi_drunet.set_noise_level(noise_level)
                with torch.no_grad():            
                    rec_pinvnet_drunet_gpu = model_pinvnet_drunet.reconstruct_expe(m)
                    rec_pinvnet_drunet = rec_pinvnet_drunet_gpu.cpu().detach().numpy().squeeze()
                
                    fig , axs = plt.subplots(1,1)
                    im = axs.imshow(rec_pinvnet_drunet, cmap='gray')
                    noaxis(axs)
                    add_colorbar(im, 'bottom')
                    
                    full_path = save_root / (data_file_prefix + '_' + f'pinvnet_drunet_n{noise_level}_{N_rec}' + '.pdf')
                    fig.savefig(full_path, bbox_inches='tight', dpi=600)      

        
        #%% simulations
        if mode_sim:
            x = x.view(b * c, h * w)
            y = noise(x.to(device))
            with torch.no_grad():
                rec_sim_gpu = model.reconstruct(y.to(device))
                rec_sim = rec_sim_gpu.cpu().detach().numpy().squeeze()
            
            fig , axs = plt.subplots(1,1)
            im = axs.imshow(rec_sim, cmap='gray')
            noaxis(axs)
            add_colorbar(im, 'bottom')

            full_path = save_root / (data_file_prefix + '_' + f'sim{img_id}_{N_rec}' + '.pdf')
            fig.savefig(full_path, bbox_inches='tight', dpi=600)
