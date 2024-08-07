# -*- coding: utf-8 -*-
"""
This script is based on reconstructs the images in Figure 8

Requirements: 
- stat/ :  covariances from oe_paper
- model/ : models from oe_paper, drunet, dfbnet
- data/ : experimental datasets and ILVRC2012_v10102019

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
import pprint 

from skimage.metrics import structural_similarity as ssim
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

# PnP from Audrey
import sys
sys.path.insert(0, '../../DFBNet-PnP/') # update if necessary!
sys.path.insert(0, '../../DFBnet-spyrit/') # update if necessary!
from models import get_model, load_model

torch.manual_seed(0)

#%% user-defined
# used for acquisition
N_acq = 64

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
metrics_eval = ['nrmse', 'ssim'] 
num_batchs_metrics = None # Number of batchs to evaluate: None: all

# We used test set for training and now use val set for evaluation 
ds_type_eval = 'val' 

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

bs = 128

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

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
                    'other_specs': {'noise_level': 35},
                    'model_path': '../../model/',
                    'model_name': 'drunet_gray.pth',
                },      
                # DFBNet
                {   'net_arch':'dfb-net',
                    'net_denoi': 'dfb',
                    'other_specs': {'mu': 3000, 'gamma': 1/N_rec**2, 'max_iter': 101, 'crit_norm': 1e-4},
                    'model_path': '../../model_pnp/DFBNet_l1_patchsize=50_varnoise0.1_feat_100_layers_20/',
                    'model_name': 'DFBNet_l1_patchsize=50_varnoise0.05_feat_100_layers_20.pth',
                }          
                ]

models_specs = models_specs[:4]
#models_specs = models_specs[-1:] # Test only DFBNet
#models_specs = models_specs[-2:-1] # Test only DRUNet
#models_specs = models_specs[-2:]

# Assess several noise values for DRUNet
mode_drunet_est_noise = False
if mode_drunet_est_noise:
    ds_type_eval = 'train'
    mode_eval_metrics = True
    num_batchs_metrics = 3
    #noise_levels = [30, 35, 40, 45, 50, 55, 60]    # N0=10
    #noise_levels = [70, 80, 90, 95, 100, 105, 110] # N0=2
    noise_levels = [20, 30, 40]         # N0=50
    models_spec_ref = {   
                    'net_arch':'pinv-net', 
                    'net_denoi':'drunet', 
                    'other_specs': {'noise_level': 55},
                    'model_path': '../../model/',
                    'model_name': 'drunet_gray.pth',
                    }                
                
    models_specs = []
    for noise_level in noise_levels:
        model_specs = models_spec_ref.copy()
        model_specs['other_specs'] = {'noise_level': noise_level}
        models_specs.append(model_specs)

# Assess several mu values for DFBNet
mode_dfbnet_est_mu = False
if mode_dfbnet_est_mu:
    ds_type_eval = 'train'
    mode_eval_metrics = False #True
    num_batchs_metrics = 3
    mu_values = [2000]
    models_spec_ref = {   
                    'net_arch':'dfb-net',
                    'net_denoi': 'dfb',
                    'other_specs': {'mu': 3000, 'gamma': 1/N_rec**2, 'max_iter': 101, 'crit_norm': 1e-4},
                    'model_path': '../../model_pnp/DFBNet_l1_patchsize=50_varnoise0.1_feat_100_layers_20/',
                    'model_name': 'DFBNet_l1_patchsize=50_varnoise0.05_feat_100_layers_20.pth',
                    }                
                
    models_specs = []
    for mu in mu_values:
        model_specs = models_spec_ref.copy()
        model_specs['other_specs'] = {'mu': mu}
        models_specs.append(model_specs)

print('Models specifications:')
pprint.pprint(models_specs)
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
    elif net_denoi == 'dfb':
        # Define DFB model
        denoi = get_dfb_model()    
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
    elif net_arch == 'dfb-net':
        from spyrit_dev import PnP

        #-- recon param
        gamma = 1/N_rec**2
        max_iter = 101
        crit_norm = 1e-4

        model = PnP(noise, prep, denoi, gamma, mu, max_iter, crit_norm)

    if net_denoi != 'I' and net_denoi != 'P0' and net_denoi != 'drunet' and net_denoi != 'dfb':
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
    if colorbar:
        add_colorbar(im, 'bottom')
    plt.axis('off')
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

def data_loader(data_root_train, data_root_val, img_size, batch_size, ds_type='val'):
    # Data loaders
    dataloaders = data_loaders_ImageNet(data_root_train, 
                                data_root_val, 
                                img_size=img_size, 
                                batch_size=batch_size)
    return dataloaders[ds_type]

def compute_nrmse(x, x_gt, dim=[2,3]):
    # Compute relative error across pixels
    if isinstance(x, np.ndarray):
        nrmse_val = nrmse(x, x_gt)
    else:
        nrmse_val = torch.linalg.norm(x - x_gt, dim=dim)/ torch.linalg.norm(x_gt, dim=dim)
    return nrmse_val

def compute_ssim(x, x_gt):
    if not isinstance(x, np.ndarray):
        x = x.cpu().detach().numpy().squeeze()
        x_gt = x_gt.cpu().detach().numpy().squeeze()
    ssim_val = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        ssim_val[i] = ssim(x[i], x_gt[i], data_range=x[i].max() - x[i].min())
    return torch.tensor(ssim_val)

def compute_metric_batch(images, targets, metric='nrmse', operation='sum'):
    """
    Compute mean and variance of a metric
    """
    if metric == 'nrmse':
        metric_batch = compute_nrmse(images, targets)
    elif metric == 'ssim':
        metric_batch = compute_ssim(images, targets)
    else:
        raise ValueError(f'Metric {metric} not supported')
    
    if operation == 'sum':
        # Sum over all images in the batch
        metric_batch_sum = torch.sum(metric_batch)
        
        # Sum of squares to compute variance
        metric_batch_sq = torch.sum(metric_batch**2)
        return metric_batch_sum, metric_batch_sq
    elif operation == 'mean':
        metric_batch = torch.mean(metric_batch)
        return metric_batch
    else:
        raise ValueError(f'Operation {operation} not supported')
    

def eval_model_metrics_batch_cum(model, dataloader, device, metrics = ['nrmse', 'ssim'], num_batchs = None):
    """
    Compute metrics meand and variance for a dataset, accumulating across batches
    """
    model.eval()
    results = {}
    n = 0
    for i, (inputs, _) in enumerate(dataloader):
        if num_batchs is not None and i >= num_batchs:
            break
        inputs = inputs.to(device)
        outputs = model(inputs)
        for metric in metrics:
            # Accumulate sum and sum of squares across batches
            results_batch_sum, results_batch_sq = compute_metric_batch(outputs, inputs, metric)
            results[metric] = results.get(metric, 0)  + results_batch_sum.cpu().detach().numpy().item()
            results[metric + '_var'] = results.get(metric + '_var', 0) + results_batch_sq.cpu().detach().numpy().item()

        n = n + inputs.shape[0]
    for metric in metrics:
        # Compute mean and variance
        results[metric] = results[metric] / n
        results[metric + '_var'] = results[metric + '_var'] / n - results[metric]**2
    return results   

def mean_walsh(dataloader, device, n_loop=1):
    """
    nloop > 1 is relevant for dataloaders with random crops such as that
    provided by data_loaders_ImageNet

    """
    import spyrit.misc.walsh_hadamard as wh

    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (b, c, nx, ny) = inputs.shape
    tot_num = len(dataloader) * b

    # Init
    n = 0
    H = wh.walsh_matrix(nx).astype(np.float32, copy=False)
    mean = torch.zeros((nx, ny), dtype=torch.float32)

    # Send to device (e.g., cuda)
    mean = mean.to(device)
    H = torch.from_numpy(H).to(device)

    # Compute Mean
    # Accumulate sum over all images in dataset
    for i in range(n_loop):
        torch.manual_seed(i)
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            trans = wh.walsh2_torch(inputs, H)
            mean = mean.add(torch.sum(trans, 0))
            # print
            n = n + inputs.shape[0]
            print(f"Mean:  {n} / (less than) {tot_num*n_loop} images", end="\n")
            # test
            # print(f' | {inputs[53,0,33,49]}', end='\n')
        print("", end="\n")

    # Normalize
    mean = mean / n
    mean = torch.squeeze(mean)

    return mean

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
def get_dfb_model():
    #-- load denoiser
    n_channel, n_feature, n_layer = 1, 100, 20

    model_dir = f'../../model_pnp/DFBNet_l1_patchsize=50_varnoise0.1_feat_{n_feature}_layers_{n_layer}/'
    model, net_name, clip_val, lr = get_model('DFBNet', n_channel,  n_feature, n_layer)
    model = load_model(pth = model_dir + net_name + '.pth', 
                        net = model, 
                        n_ch = n_channel, 
                        features = n_feature, 
                        num_of_layers = n_layer)
    model.module.update_lip((1,50,50))
    model.eval()    
    return model  

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
    # We used test set for training and now use val set for evaluation 
    dataloader_val = data_loader(data_root_train=Path(data_root) / 'test', 
                                   data_root_val=Path(data_root) / 'val', 
                                   img_size=N_rec, 
                                   batch_size=bs,
                                   ds_type=ds_type_eval)
    results_metrics = {}

# Loop over models
for model_specs in models_specs:
    # Model specifications
    net_arch = model_specs['net_arch']
    net_denoi = model_specs['net_denoi']
    model_path = model_specs['model_path']
    model_name = model_specs['model_name']
    other_specs = model_specs['other_specs']
    print(f'Model: {net_arch} - {net_denoi}')
    # LPGD
    lpgd_iter = other_specs.get('lpgd_iter', None)
    step_decay = other_specs.get('step_decay', None)
    wls = other_specs.get('wls', False)
    step_grad = other_specs.get('step_grad', False)
    x_gt = None
    # DRUNet
    noise_level = other_specs.get('noise_level', None)
    # DFBNet
    mu = other_specs.get('mu', None)

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
        meas = HadamSplit(M, N_rec, torch.from_numpy(Ord_rec))
        noise = Poisson(meas, N0) # could be replaced by anything here as we just need to recon
        prep  = SplitPoisson(N0, meas)  
        prep.set_expe()
        model, denoi = init_reconstruction_network(noise, prep, Cov_rec, net_arch, net_denoi,
                                                    model_path, model_name, lpgd_iter, step_decay, wls, step_grad)

        # noise error for drunet
        if net_denoi == 'drunet':
            denoi.set_noise_level(noise_level)
            name_save_details = name_save_details + f'_nlevel_{noise_level}'
        # params for DFBNet
        if net_arch == 'dfb-net':
            name_save_details = name_save_details + f'_mu_{int(mu)}'
        ###########################################################################   
        # Evaluate metrics
        if mode_eval_metrics:
            results = eval_model_metrics_batch_cum(model, dataloader_val, device, metrics = metrics_eval, num_batchs = num_batchs_metrics)
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
            #for i in range(0,1):    
                #if b > 1:
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
            for data_file_prefix in data_file_prefix_list[1:2]:
                
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
                    # Warning: Reconstruction reconstruct_expe handled inside reconstruct!!!!
                    if net_arch == 'lpgd' or net_arch == 'dfb-net':
                        rec_gpu = model.reconstruct(m, exp=True)
                    else:
                        model.prep.set_expe()
                        rec_gpu = model.reconstruct_expe(m)
                    rec = rec_gpu.cpu().detach().numpy().squeeze()
                
                    #%% Plot or save 
                    full_path = save_root / (data_file_prefix + '_' + f'{M}_{N_rec}' + f'_{name_save_details}' + '.png')
                    plt.close('all')
                    imshow_specs(rec, vmin=None, vmax=None, colorbar=True, path_save=full_path)
              
if mode_eval_metrics:                    
    print(f'Metrics for {name_save_details}: {results_metrics}')

    # Save metrics
    save_metrics(results_metrics, save_root / f'N0_{N0}_metrics_sim_test.pkl')
    df = pd.DataFrame(results_metrics)
    df.to_csv(save_root / f'N0_{N0}_metrics_sim_test.csv')

            
                    
                
