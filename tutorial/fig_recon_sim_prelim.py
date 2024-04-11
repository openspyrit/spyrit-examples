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
import pickle
# get debug in spyder
import collections
collections.Callable = collections.abc.Callable

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse

import torchvision

from spyrit.misc.statistics import Cov2Var, data_loaders_ImageNet
from spyrit.core.noise import Poisson 
from spyrit.core.meas import HadamSplit
from spyrit.core.prep import SplitPoisson
from spyrit.core.recon import DCNet, PinvNet, LearnedPGD
from spyrit.core.train import load_net
from spyrit.core.nnet import Unet, ConvNet, Identity
from spyrit.misc.sampling import reorder, Permutation_Matrix
from spyrit.misc.disp import add_colorbar, noaxis

torch.manual_seed(0)

#%% user-defined
# used for acquisition
N_acq = 32

# for reconstruction
N_rec = 128  # 128 or 64
M_list = [4096] #[4096, 1024, 512] # for N_rec = 128
#N_rec = 64
#M_list = [1024]

N0 = 50     # Check if we used 10 in the paper
stat_folder_rec = Path('../../stat/oe_paper/') # Path('../../stat/ILSVRC2012_v10102019/')

# Reconstruct simulated images from folder
mode_sim = True 
mode_sim_crop = False

# Evaluate metrics on ImageNet test set
mode_eval_metrics = False
metrics_eval = ['nrmse', 'ssim', 'psnr'] 
num_batchs_metrics = None # Number of batchs to evaluate: None: all

# Reconstruction of experimental data
mode_exp = False
if mode_exp:
    from spas import read_metadata, spectral_slicing

# Reconstruction parameters
metrics = False # Compute metrics: MSE
log_fidelity = False
step_estimation = False
#wls = False
step_grad = False

# limits for plotting images
vmin = -1 # None: no limit
vmax = 1 

colorbar = False

save_root = Path('../../recon/')
if not os.path.exists(save_root):
    os.makedirs(save_root)

# Simulated images
path_natural_images = '../../data/spy_pub_imgs/target/'

# Select image
img_id = None # None: all images; 0: first image

bs = 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Path for data evaluation
data_root = '../../data/ILSVRC2012_v10102019/'
net_data    = 'imagenet'    # 'imagenet'
# ---------------------------------------------------------
# Models
# (net_arch, net_denoi, other_specs). 
# lpgd: other_specs=(lpgd_iter, step_decay, wls)
# drunet: other_specs=(noise_level)
models_specs = [ 
                # DC-Net
                {   'net_arch':'dc-net',   
                    'net_denoi': 'unet', 
                    'other_specs': {}, 
                    'model_path': '../../model/oe_paper/',
                    'model_name': 'dc-net_unet_imagenet_rect_N0_10_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07_light', # Taken from the data specifications
                },
                # Pinv-Net: previously used, now lpgd 1 iter
                #{   'net_arch':'pinv-net', 
                #    'net_denoi': 'unet', 
                #    'other_specs': {}, 
                #    'model_path': '../../model/', 
                #    'model_name': 'pinv-net_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07',
                #},
                # LPGD 1 it: Pinv
                {   'net_arch':'lpgd',     
                    'net_denoi': 'I',    
                    'other_specs': {'lpgd_iter': 1, 'step_decay': 1,    'wls': False}, 
                    'model_path': None,
                    'model_name': None,
                },
                # LPGD 1 it: GD P0
                #{'net_arch':'lpgd',     'net_denoi': 'P0',   'other_specs': None},
                # LPGD 1 it Unet : Pinv-net
                {   'net_arch':'lpgd',     
                    'net_denoi': 'unet', 
                    'other_specs': {'lpgd_iter': 1, 'step_decay': 1,    'wls': False},
                    'model_path': '../../model/',
                    'model_name': 'lpgd_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_128_reg_1e-07_uit_1',
                },
                # LPGD 3 it Unet  decay
                {   'net_arch':'lpgd',     
                    'net_denoi': 'unet', 
                    'other_specs': {'lpgd_iter': 3, 'step_decay': 0.9,  'wls': False},
                    'model_path': '../../model/',
                    'model_name': 'lpgd_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_128_reg_1e-07_uit_3_sdec0-9',
                },
                # 6 it (run twice with gradient step estimation)
                #model_name = "lpgd_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_15_lr_0.001_sss_10_sdr_0.5_bs_128_reg_1e-07_uit_6_sgrad_sdec0-9_cont"
                # DRUNet
                {   'net_arch':'pinv-net', 
                    'net_denoi':'drunet', 
                    'other_specs': {'noise_level': 40},
                    'model_path': '../../model/',
                    'model_name': 'drunet_gray.pth',
                },                
                ]

#models_specs = models_specs[2:]

# Assess several noise values for DRUNet
noise_levels = [20, 25, 30, 35, 40, 45, 50]
for noise_level in noise_levels:
    models_specs.append(models_specs[-1].copy())
    models_specs[-1]['other_specs'] = {'noise_level': noise_level}

######################################################
# Reconstruction functions
def init_denoi(net_denoi, model_path = None, model_name = None, lpgd_iter = None,):
    if net_denoi == 'unet':
        denoi = Unet()
    elif net_denoi == 'cnn':
        denoi = ConvNet() 
    elif net_denoi == 'cnn-diff':
        import torch.nn as nn 
        denoi = nn.ModuleList([ConvNet() for _ in range(lpgd_iter)])
    elif net_denoi == 'drunet':
        from drunet import DRUNet
        denoi = DRUNet()          
        denoi = denoi.to(device)
        denoi.load_state_dict(torch.load(os.path.join(model_path, model_name)), strict=False) 
        denoi.eval()
        # load_net(os.path.join(model_path, model_name), denoi, device, strict = False)  
    elif net_denoi == 'P0':
        from spyrit.core.nnet import ProjectToZero
        denoi = ProjectToZero()
    elif net_denoi == 'I':
        denoi = Identity()
    return denoi    

def init_reconstruction_network(noise, prep, Cov_rec, net_arch, net_denoi = None, 
                                model_path = None, model_name = None,
                                lpgd_iter = None, step_decay = None, wls = False, step_grad = False):
    # Denoiser
    if net_denoi:
        denoi = init_denoi(net_denoi, model_path, model_name, lpgd_iter)
    
    # Reconstruction network
    if net_arch == 'dc-net':
        model = DCNet(noise, prep, sigma=torch.tensor(Cov_rec), denoi=denoi)        
    elif net_arch == 'pinv-net':
        model = PinvNet(noise, prep, denoi)
    elif net_arch == 'lpgd':
        model = LearnedPGD(noise, 
                              prep, 
                              denoi,
                              iter_stop = lpgd_iter, 
                              wls=wls,
                              step_estimation=step_estimation,
                              step_decay=step_decay,
                              gt=x_gt,
                              step_grad=step_grad)
    if net_denoi != 'I' and net_denoi != 'P0':
        load_net(os.path.join(model_path, model_name), model, device, strict = False)
    model.eval()    # Mandantory when batchNorm is used
    model.to(device)
    return model, denoi

# Reconstruction: set attributes and reconstruct
def reconstruct(model, y, device, log_fidelity= False, step_estimation = False, wls = False, metrics = False):
    with torch.no_grad():
        # Not all models have these attributes
        if step_estimation:
            model.step_estimation = step_estimation
        if log_fidelity:
            model.log_fidelity = log_fidelity
        if wls:
            model.wls = wls

        rec_sim_gpu = model.reconstruct(y.to(device))
        
        if log_fidelity:
            data_fidelity = model.cost
        else:
            data_fidelity = None
        if hasattr(model, 'mse'):
            mse = model.mse
        else:
            mse = None

        # Metrics 
        if metrics:
            rec_sim = rec_sim_gpu.cpu().detach().numpy()
            mse = compute_nrmse(rec_sim, x_gt)
        
        rec_sim = rec_sim_gpu.cpu().detach().numpy().squeeze()
        rec_sim = rec_sim.reshape(b, N_rec, N_rec)
    return rec_sim, data_fidelity, mse

def save_metrics(metric, path_save):
    #path_save = path_save + '_metrics.pkl'
    with open(path_save, 'wb') as f:
        pickle.dump(metric, f)
    print(f'Metrics saved as {os.path.abspath(path_save)}')

def plot_data(data, ylabel=None, xlabel=None, path_save = None, ylim=None):
    fig=plt.figure(); plt.plot(data)
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    if ylim:
        plt.ylim(ylim)
    if path_save:
        fig.savefig(full_path, bbox_inches='tight', dpi=600)                

def imshow_specs(x, vmin=None, vmax=None, colorbar=False, path_save=None):
    if vmin is not None:
        im = plt.imshow(x, cmap='gray', vmin=vmin, vmax=vmax)
    else: 
        im = plt.imshow(x, cmap='gray')
    plt.axis('off')
    if colorbar:
        add_colorbar(im, 'bottom')
    if path_save:
        plt.savefig(path_save, bbox_inches='tight', dpi=600)

# Sampling order
def sampling_order(N_rec, M, net_order):
    if net_order == 'rect':
        Ord_rec = np.ones((N_rec, N_rec))
        n_sub = math.ceil(M**0.5)
        Ord_rec[:,n_sub:] = 0
        Ord_rec[n_sub:,:] = 0
        
    elif net_order == 'var':
        Ord_rec = Cov2Var(Cov_rec)
    return Ord_rec

def data_loader(data_root_train, data_root_val, img_size, batch_size):
    # Data loaders
    dataloaders = data_loaders_ImageNet(data_root_train, 
                                data_root_val, 
                                img_size=img_size, 
                                batch_size=batch_size)
    return dataloaders['val']

def compute_nrmse(x, x_gt):
    if isinstance(x, np.ndarray):
        nrmse_val = nrmse(x, x_gt)
    else:
        nrmse_val = torch.linalg.norm(x - x_gt)/ torch.linalg.norm(x_gt)
    return nrmse_val

def compute_pnsr(x, x_gt):
    psnr_val = psnr(x, x_gt, data_range=x_gt.max() - x_gt.min())
    return psnr_val

def compute_ssim(x, x_gt):
    ssim_val = ssim(x, x_gt, data_range=x_gt.max() - x_gt.min())
    return ssim_val

def evaluate_model(model, dataloader, device, metrics = ['nrmse', 'ssim', 'psnr'], num_batchs = None):
    # Evaluate
    model.eval()
    results = {}
    for i, (inputs, _) in enumerate(dataloader):
        if num_batchs is not None and i >= num_batchs:
            break
        inputs = inputs.to(device)
        outputs = model(inputs)
        inputs = inputs.cpu().detach().numpy().squeeze()
        outputs = outputs.cpu().detach().numpy().squeeze()
        nrmse_val = []
        ssim_val = []
        psnr_val = []
        if 'nrmse' in metrics:
            mse_batch = compute_nrmse(outputs, inputs)
            nrmse_val.append(mse_batch)
        if 'ssim' in metrics:
            ssim_batch = compute_ssim(outputs, inputs)
            ssim_val.append(ssim_batch)
        if 'psnr' in metrics:
            psnr_batch = compute_pnsr(outputs, inputs)
            psnr_val.append(psnr_batch)
    if 'nrmse' in metrics:
        if len(nrmse_val) > 1:
            nrmse_val = np.mean(nrmse_val)
            nrmse_val_std = np.std(nrmse_val)
        else:
            nrmse_val = nrmse_val[0]
            nrmse_val_std = None
        results['nrmse'] = (nrmse_val, nrmse_val_std)
    if 'ssim' in metrics:
        if len(ssim_val) > 1:
            ssim_val = np.mean(ssim_val)
            ssim_val_std = np.std(ssim_val)
        else:
            ssim_val = ssim_val[0]
            ssim_val_std = None
        results['ssim'] = (ssim_val, ssim_val_std)
    if 'psnr' in metrics:
        if len(psnr_val) > 1:
            psnr_val = np.mean(psnr_val)
            psnr_val_std = np.std(psnr_val)
        else:
            psnr_val = psnr_val[0]
            psnr_val_std = None
        results['psnr'] = (psnr_val, psnr_val_std)
    return results   

# ---------------------------------------------------------
#%% Parameters for simulated images 
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


def transform_gray_norm_rand_crop(img_size, seed = 0, shuffle = True):
    
    torch.manual_seed(seed)  # reproductibility of random crop
    #
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.functional.to_grayscale,
            torchvision.transforms.RandomCrop(
                size=(img_size, img_size), pad_if_needed=True, padding_mode="edge"
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ]
    )    
    return transform

# ---------------------------------------------------------
#%% Simulated images
if mode_sim:
    # Create dataset and loader (expects class folder 'images/test/')
    #from spyrit.misc.statistics import transform_gray_norm
    if mode_sim_crop:
        transform = transform_gray_norm_rand_crop(N_rec)
    else:
        transform = transform_gray_norm(N_rec)

    # Check if the directory exists
    if not os.path.isdir(path_natural_images):
        print(f"Directory does not exist: {path_natural_images}")
    dataset = torchvision.datasets.ImageFolder(root=path_natural_images, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 7)

    x, _ = next(iter(dataloader))
    print(f'Shape of input images: {x.shape}')

    # Select image
    if img_id is not None:
        x = x[img_id:img_id+1,:,:,:]
    x = x.detach().clone()
    b,c,h,w = x.shape
    # Uncomment for MSE for unroll iterations: for mode_eval_metrics = False
    # x_gt = np.copy(x)

    # plot
    for i in range(b):
        x_plot = x[i].view(-1,h,h).cpu().numpy() 
        if b > 1:
            name_save_sim = f'sim{i}_{N_rec}_gt'
        else:
            name_save_sim = f'sim_{N_rec}_gt'
        if mode_sim_crop:
            name_save_sim = name_save_sim + '_crop'
        full_path = save_root / (name_save_sim + '.png')

        if i == 0:
            imshow_specs(x_plot[0,:,:], vmin=-1, vmax=0.4, colorbar=colorbar, path_save=full_path)
        else:
            imshow_specs(x_plot[0,:,:], vmin=vmin, vmax=vmax, colorbar=colorbar, path_save=full_path)

######################################################
# Data to evaluate metrics
if mode_eval_metrics:
    dataloader_val = data_loader(data_root_train=Path(data_root) / 'test', 
                                   data_root_val=Path(data_root) / 'val', 
                                   img_size=N_rec, 
                                   batch_size=bs)
    results_metrics = {}

# Loop over models
for model_specs in models_specs:
    # Model specifications
    net_arch = model_specs['net_arch']
    net_denoi = model_specs['net_denoi']
    model_path = model_specs['model_path']
    model_name = model_specs['model_name']
    other_specs = model_specs['other_specs']
    # LPGD
    lpgd_iter = other_specs.get('lpgd_iter', None)
    step_decay = other_specs.get('step_decay', None)
    wls = other_specs.get('wls', False)
    step_grad = other_specs.get('step_grad', False)
    x_gt = None
    # DRUNet
    noise_level = other_specs.get('noise_level', None)

    # Free memory
    if 'model' in locals():
        model.to('cpu')
        del model
    if 'denoi' in locals():
        denoi.to('cpu')
        del denoi

    # Name save
    name_save_details = f'{net_arch}_{net_denoi}'
    if net_arch == 'lpgd':
        name_save_details = name_save_details + f'_it{lpgd_iter}'
        if wls:
            name_save_details = name_save_details + '_wls'
        if step_decay != 1:
            name_save_details = name_save_details + f'_sdec{step_decay}'.replace('.','')
        if step_grad:
            name_save_details = name_save_details + '_sgrad'   
    # ---------------------------------------------------------
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
        
        #if net_arch == 'dc-net':
        #    model_name = f'{net_arch}_{net_denoi}_{net_data}_{net_order}_{net_suffix}'

        # Init data
        if 'y' in locals():
            del y

        #%% Init and load trained network
        # Covariance in hadamard domain
        Cov_rec = np.load(cov_rec_file)
        
        # Sampling order
        Ord_rec = sampling_order(N_rec, M, net_order)

        if img_id is not None:
            name_save = f'sim{img_id}'
        else:
            name_save = 'sim'
        name_save = name_save + f'_{N_rec}_N0_{N0}_M_{M}_{net_order}'      
        if mode_sim_crop:
            name_save = name_save + f'_crop'  
        # ---------------------------------------------------------
        # Init network  
        meas = HadamSplit(M, N_rec, Ord_rec)
        noise = Poisson(meas, N0) # could be replaced by anything here as we just need to recon
        prep  = SplitPoisson(N0, meas)    

        model, denoi = init_reconstruction_network(noise, prep, Cov_rec, net_arch, net_denoi,
                                                    model_path, model_name, lpgd_iter, step_decay, wls, step_grad)

        # noise error for drunet
        if net_denoi == 'drunet':
            denoi.set_noise_level(noise_level)
            name_save_details = name_save_details + f'_nlevel_{noise_level}'
        ###########################################################################   
        # Evaluate metrics
        if mode_eval_metrics:
            results = evaluate_model(model, dataloader_val, device, metrics = metrics_eval, num_batchs = num_batchs_metrics)
            print(f'Metrics for {name_save_details}: {results}')
            results_metrics[name_save_details] = results
        # ---------------------------------------------------------    
        #%% simulations
        if mode_sim and 'y' not in locals():
            x = x.view(b * c, h * w)
            y = noise(x.to(device))
        
        if mode_sim:
            with torch.no_grad():
                rec_sim, data_fidelity, mse = reconstruct(model, y, device, log_fidelity, 
                                                        step_estimation, wls, metrics=metrics)
            if 'name_save_details' in globals():
                name_save = name_save + '_' + name_save_details
            
            for i in range(b):                
                if b > 1:
                    name_save_this = name_save.replace('sim', f'sim{i}')
                full_path = save_root / (name_save_this + '.png')
                if i == 0:
                    imshow_specs(rec_sim[i], vmin=-1, vmax=0.4, colorbar=colorbar, path_save=full_path)
                else:
                    imshow_specs(rec_sim[i], vmin=vmin, vmax=vmax, colorbar=colorbar, path_save=full_path)
                # 
                if metrics:
                    if log_fidelity:
                        # Data fidelity
                        full_path = save_root / (name_save_this + '_data_fidelity.png')
                        plot_data(data_fidelity, label='GD', path_save=full_path)

                    if hasattr(model, 'mse'):
                        # MSE
                        #np.linalg.norm(x_gt-rec_sim)/np.linalg.norm(x_gt)
                        mse = model.mse
                        mse = np.array(mse)/np.linalg.norm(x_gt)
                        full_path = save_root / (name_save_this + '_nmse.png')
                        plot_data(mse, ylabel='NMSE', xlabel='Iterations', path_save=full_path)

                    print(f'MSE: {mse}')
                    # Save metrics
                    save_metrics(mse, save_root / (name_save_this + '_metrics.pkl'))

        ###########################################################################
        #%% Load expe data and unsplit
        if mode_exp:
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
                    
                    if True:  # all methods?
                        rec_gpu = model.reconstruct_expe(m)
                        rec = rec_gpu.cpu().detach().numpy().squeeze()
                    
                        #%% Plot or save 
                        full_path = save_root / (data_file_prefix + '_' + f'{M}_{N_rec}' + f'_{name_save_details}' + '.png')
                        imshow_specs(rec, vmin=None, vmax=None, colorbar=colorbar, path_save=full_path)
              
                    
print(f'Metrics for {name_save_details}: {results_metrics}')
            
                    
                
