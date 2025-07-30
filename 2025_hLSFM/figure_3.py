# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 17:38:00 2025

@author: ducros
"""
#%% to debug needs to run
import collections
collections.Callable = collections.abc.Callable

fig_folder = './figure/'

#%% Load experimental hadamard patterns (after split compensation)
from pathlib import Path
from spyrit.misc.disp import add_colorbar, noaxis
import matplotlib.pyplot as plt
import numpy as np

M = 64
N = 512
data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/'
mat_folder = '/Reconstruction/Mat_rc/'

# load
H_exp = np.load(Path(data_folder + mat_folder) / f'hadamard_matrix_{M}_{N}.npy')
H_exp /= H_exp[0,16:500].mean()
   
#
Nl, Nh, Nc = 512, 64, 1280  # shape of preprocessed data
 
# Spectral binning **around** the central channel. The preprocessed data were binned 
# spectrally by a factor of two. Choosing c_step = 5 leads to spectral bins with 
# 2*(2*5) channels.

c_step = 5
#lambda_central_list = np.arange(510, 598, 12, dtype=int)  # Central channel in nanometer
lambda_central_list = 510, 551, 594  # Central channel in nanometer


lambda_all = np.linspace(500, 608, Nc)

#%% Init physics operators for Hadamard patterns | Alternative
import torch
from spyrit.core.meas  import Linear
from spyrit.core.recon import PseudoInverse

M = 64
N = 512

linop = Linear(torch.from_numpy(H_exp), pinv=True, meas_shape = (1,512)) 
# meas_shape = (1,512) allows to work in 1D, i.e., the forward model
# applies to last dimension only. meas_shape = (512, 1) works as well.

recon = PseudoInverse()

#%% Pinv
# raw data
data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/'
Run = 'RUN0004' 

# save
save_rec = True
save_fig = True     # Either save or display
ext_fig = 'pdf'
dpi_fig = 600

# Load net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
linop.to(device) 

# Load prep data
save_folder = '/Preprocess/'
filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_pos.npy'
prep_pos = np.load(Path(data_folder+save_folder) / filename)

filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_neg.npy'
prep_neg =  np.load(Path(data_folder+save_folder) / filename)

# spectral dimension comes first
prep_pos = np.moveaxis(prep_pos, -1, 0)
prep_neg = np.moveaxis(prep_neg, -1, 0)

# param #2
nc, nl, nh = prep_neg.shape
y = prep_pos - prep_neg #/np.expand_dims(prep_pos[:,:,0], axis=2)
y = torch.from_numpy(y)

print(f'Loaded: {filename[:-8]}')
print(f'Pos data: range={prep_pos.max() - prep_pos.min()} counts; mean={prep_pos.mean()} counts')  

y = y.to(device)
y = y.view(-1,1,N,M)

# init output
rec_had = np.zeros((len(lambda_central_list), N, N))

for i_lambda, lambda_central in enumerate(lambda_central_list):
    with torch.no_grad():
        c_central = np.argmin((lambda_all-lambda_central)**2) # Central channel
        m = y[c_central-c_step:c_central+c_step].sum(0, keepdim=True).to(device, torch.float32)
        print(f'reconstructing spectral bin from channels: {c_central-c_step}--{c_central+c_step}')
        rec_gpu = recon(m, linop)
        rec = rec_gpu.cpu().detach().numpy().squeeze()
        rec = np.moveaxis(rec, 0, -1) # spectral channel is now the last axis
        rec = np.flip(rec,0)
        rec = np.fliplr(rec)
        rec = np.rot90(rec,-1)
        rec_had[i_lambda] = rec
    
    if save_rec:
        save_folder = 'Reconstruction/hypercube'  
        Path(data_folder+save_folder).mkdir(parents=True, exist_ok=True)
            
        save_filename = f'had_{lambda_central}_{N}x{N}.npy'
        np.save(Path(data_folder+save_folder) / save_filename, rec)

            
# Plot
for i_lambda, lambda_central in enumerate(lambda_central_list):
    fig, axs = plt.subplots(1, 1, figsize=(5,5))
    im = axs.imshow(rec_had[i_lambda])
    axs.set_title(f'{lambda_central} nm')
    add_colorbar(im, 'bottom')
        
    noaxis(axs)
        
    if save_fig:
        save_filename = f'had_{lambda_central}_{N}x{N}.{ext_fig}'
        plt.savefig(Path(data_folder+save_folder)/save_filename, bbox_inches='tight', dpi=dpi_fig)
        plt.close(fig)


#%% Tikhonet + unet
from spyrit.core.nnet import Unet
from spyrit.core.train import load_net

div = 1.5
alpha = 50

nbin = 20*4
mudark = 105.0
sigdark = 5.0
gain = 2.6

# raw data
# recon = f'tikhonet{alpha}_div{div}_exp'
# save_folder = f'Reconstruction/hypercube/tikhonet{alpha}_div{div}/'

# covariance prior in the image domain
stat_folder = './stat/'
cov_file   = f'Cov_1_{N}x{N}.npy'
mean_file   = f'Average_1_{N}x{N}.npy'
mean  = np.load(Path(stat_folder) / mean_file)
sigma = np.load(Path(stat_folder) / cov_file)

# Load net
net_prefix = f'tikho-net_unet_imagenet_ph_{alpha}'
net_suffix = 'N_512_M_128_epo_20_lr_0.001_sss_10_sdr_0.5_bs_20_reg_1e-07'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from spyrit.core.prep import SplitPoissonRaw
from spyrit.core.noise import Poisson
from spyrit.core.recon import TikhoNet 
from spyrit.core.meas import LinearSplit

linop = LinearSplit(torch.from_numpy(H_exp), pinv=True, meas_shape=(1,512))

prep   = SplitPoissonRaw(alpha, linop)
prep.set_expe(gain, mudark, sigdark, nbin)
noise  = Poisson(linop, alpha)
recnet = TikhoNet(noise, prep, torch.from_numpy(sigma/div), Unet())  

title = './model/' + net_prefix + '_exp_' + net_suffix + '.pth'
load_net(title, recnet, device, False)
recnet.eval()   # Mandantory when batchNorm is used
recnet.to(device)  


# Load prep data
save_folder = '/Preprocess/'
filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_pos.npy'
prep_pos = np.load(Path(data_folder+save_folder) / filename)

filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_neg.npy'
prep_neg =  np.load(Path(data_folder+save_folder) / filename)

# spectral dimension comes first
prep_pos = np.moveaxis(prep_pos, -1, 0)
prep_neg = np.moveaxis(prep_neg, -1, 0)

# spectral binning of both pos/neg for a few central channels
prep_pos = torch.from_numpy(prep_pos)
prep_neg = torch.from_numpy(prep_neg)

y_pos = torch.empty((len(lambda_central_list),N,M))
y_neg = torch.empty((len(lambda_central_list),N,M))

for i_lambda, lambda_central in enumerate(lambda_central_list):
    with torch.no_grad():
        c_central = np.argmin((lambda_all-lambda_central)**2) # Central channel
        
        y_pos[i_lambda] = prep_pos[c_central-c_step:c_central+c_step].sum(0, keepdim=True)
        y_neg[i_lambda] = prep_neg[c_central-c_step:c_central+c_step].sum(0, keepdim=True)
        #.to(device, torch.float32)
        print(f'reconstructing spectral bin from channels: {c_central-c_step}--{c_central+c_step}')
        
# param #2
background  = (2**15-1)*nbin
y_pos = y_pos - background
y_neg = y_neg - background

y = torch.empty((len(lambda_central_list), N, 2*M))
y[:,:,::2]  = y_pos #/np.expand_dims(prep_pos[:,:,0], axis=2)
y[:,:,1::2] = y_neg #/np.expand_dims(prep_pos[:,:,0], axis=2)

print(f'Loaded: {filename[:-8]}')
print(f'Pos data: range={prep_pos.max() - prep_pos.min()} counts; mean={prep_pos.mean()} counts')  

y = y.to(device)
y = y.view(-1,1,N,2*M) 

with torch.no_grad():          
    rec_gpu, beta_gpu = recnet.reconstruct_expe(y)

rec_ticko = rec_gpu.cpu().detach().numpy().squeeze()
beta = beta_gpu.cpu().detach().numpy().squeeze()
print(beta)
                
rec_ticko = np.fliplr(rec_ticko)

    
#%% Tikhonet + unet: plot and save 
save_rec = True
save_fig = True

for i_lambda, lambda_central in enumerate(lambda_central_list):        
    
    fig, axs = plt.subplots(1, 1, figsize=(5,5))
    im = axs.imshow(rec_ticko[i_lambda])
    axs.set_title(f'{lambda_central} nm')
    add_colorbar(im, 'bottom')        
    noaxis(axs)
    
    if save_fig:
        save_filename = f'had_tikhonet_{lambda_central}_{N}x{N}.{ext_fig}'
        plt.savefig(Path(data_folder+save_folder)/save_filename, bbox_inches='tight', dpi=dpi_fig)
        
    if save_rec:
        save_folder = 'Reconstruction/hypercube'  
        Path(data_folder+save_folder).mkdir(parents=True, exist_ok=True)
            
        save_filename = f'had_tikhonet_{lambda_central}_{N}x{N}.npy'
        np.save(Path(data_folder+save_folder) / save_filename, rec_ticko[i_lambda])
        plt.close(fig)
    

#%% Tikhonov alone

# Denoiser as identity
recnet.denoi = torch.nn.Identity()

# Reconstruct
with torch.no_grad():          
    rec_gpu, beta_gpu = recnet.reconstruct_expe(y)
    
del recnet

rec_ticko = rec_gpu.cpu().detach().numpy().squeeze()
beta = beta_gpu.cpu().detach().numpy().squeeze()
print(beta)
                
rec_ticko = np.fliplr(rec_ticko)

#%% Tikhonov alone: plot and save 
save_rec = True
save_fig = True

for i_lambda, lambda_central in enumerate(lambda_central_list):        
    
    fig, axs = plt.subplots(1, 1, figsize=(5,5))
    im = axs.imshow(rec_ticko[i_lambda])
    axs.set_title(f'{lambda_central} nm')
    add_colorbar(im, 'bottom')        
    noaxis(axs)
    
    if save_fig:
        save_filename = f'had_tikhonov_{lambda_central}_{N}x{N}.{ext_fig}'
        plt.savefig(Path(data_folder+save_folder)/save_filename, bbox_inches='tight', dpi=dpi_fig)
        
    if save_rec:
        save_folder = 'Reconstruction/hypercube'  
        Path(data_folder+save_folder).mkdir(parents=True, exist_ok=True)
            
        save_filename = f'had_tikhonov_{lambda_central}_{N}x{N}.npy'
        np.save(Path(data_folder+save_folder) / save_filename, rec_ticko[i_lambda])
        plt.close(fig)
        
#%% Pinv for CANONICAL patterns
from pathlib import Path
from spyrit.core.meas  import Linear
from spyrit.core.recon import PseudoInverse
import torch

# save
save_rec = True
save_fig = True

# Load experimental canonical patterns (after split compensation)
mat_folder = '/Reconstruction/Mat_rc/'
H_exp = np.load(Path(data_folder + mat_folder) / f'canonical_diff_matrix_{M}_{N}.npy')
#H_exp /= H_exp[0,16:500].mean()

# Init measurement operator
M = 64
N = 512
linop = Linear(torch.from_numpy(H_exp), pinv=True, meas_shape = (1,512))
recon = PseudoInverse()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
linop.to(device) 

# raw data
data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/'
Run_can = 'RUN0006' 
Run_dark = 'RUN0003'  

# Load prep data
Nl, Nh, Nc = 512, 64, 1280 # shape of preprocessed data

save_folder = '/Preprocess/'
filename = f'{Run_can}_Can_{Nl}_{Nh}_{Nc}_can.npy'
prep_can = np.load(Path(data_folder+save_folder) / filename)

filename = f'{Run_dark}_Can_{Nl}_{Nh}_{Nc}_dark.npy'
prep_dark = np.load(Path(data_folder+save_folder) / filename)

# The mean of the dark stack is substracted
prep_diff = prep_can - prep_dark.mean(1,keepdims=True)
prep_diff = np.moveaxis(prep_diff, -1, 0)
y = torch.from_numpy(prep_diff)

print(f'Loaded: {filename[:-8]}')
print(f'Pos data: range={prep_diff.max() - prep_diff.min()} counts; mean={prep_diff.mean()} counts')  

y = y.to(device)
y = y.view(-1,1,N,M) 

# init output
rec_can = np.zeros((len(lambda_central_list), N, N))

# Reconstruct all spectral bins
for i_lambda, lambda_central in enumerate(lambda_central_list):
    
    with torch.no_grad():
        c_central = np.argmin((lambda_all-lambda_central)**2) # Central channel
        
        m = y[c_central-c_step:c_central+c_step].sum(0, keepdim=True).to(device, torch.float32)
        print(f'reconstructing spectral bin from channels: {c_central-c_step}--{c_central+c_step}')
        rec_gpu = recon(m, linop)
        rec = rec_gpu.cpu().detach().numpy().squeeze()
                    
        rec = np.moveaxis(rec, 0, -1) # spectral channel is now the last axis
        rec = np.flip(rec,0)
        rec = np.fliplr(rec)
        rec = np.rot90(rec,-1)
        rec_can[i_lambda] = rec
        
        if save_rec:
            save_folder = 'Reconstruction/hypercube'  
            Path(data_folder+save_folder).mkdir(parents=True, exist_ok=True)
            
            save_filename = f'can_{lambda_central}_{N}x{N}.npy'
            np.save(Path(data_folder+save_folder) / save_filename, rec)

# Plot or save
for i_lambda, lambda_central in enumerate(lambda_central_list):
        
        fig, axs = plt.subplots(1, 1, figsize=(5,5))
        im = axs.imshow(rec_can[i_lambda])
        axs.set_title(f'{lambda_central} nm')
        add_colorbar(im, 'bottom')
            
        noaxis(axs)
            
        if save_fig:
            save_filename = f'can_{lambda_central}_{N}x{N}.{ext_fig}'
            plt.savefig(Path(data_folder+save_folder)/save_filename, bbox_inches='tight', dpi=dpi_fig)
            plt.close(fig)
    

#%% Peak signal to noise ratio

def psnr(image, x, y, plot=True):
    """Calculates the Peak Signal-to-Noise Ratio (PSNR) of an image.

    Args:
        image: The input image.
        x: A tuple specifying the starting and ending row indices of the background.
        y: A tuple specifying the starting and ending column indices of the background.

    Returns:
        The calculated PSNR value.

    This function calculates the PSNR using the formula:

    PSNR = 20*log10(MAX/STD),

    where:
        MAX: The maximum pixel value in the image.
        STD: The standard deviation within the background region.

    The PSNR provides a measure of image quality, with higher values indicating better image quality.
    """
    imax = image.max()
    std = np.std(image[x[0]:x[1],y[0]:y[1]])
    out = 20*np.log10(imax/std)
    
    if plot:
        
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots()
        plt.imshow(image)
        plt.title(f'psnr: {out:.2f} dB')
        plt.colorbar()
        ax.add_patch(Rectangle((y[0], x[0]), y[1]-y[0]+1, x[1]-x[0]+1, linewidth=1, edgecolor='r', facecolor='none'))
    
    return out

#%% Main plot with PSNR comparison
from matplotlib.patches import Rectangle

data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/'
save_folder = 'Reconstruction/hypercube'

fs = 11 # font size

N = 512

x_noise = [330, 480]
y_noise = [50, 300]

psnr_had = np.zeros(len(lambda_central_list))
psnr_can = np.zeros(len(lambda_central_list))
psnr_had_tikhonov = np.zeros(len(lambda_central_list))
psnr_had_tikhonet = np.zeros(len(lambda_central_list))

f, ax = plt.subplots(4,3, figsize=(9,12))

for i_lambda, lambda_central in enumerate(lambda_central_list):
    
    # pushbroom / canonical
    save_filename = f'can_{lambda_central}_{N}x{N}.npy'
    rec = np.load(Path(data_folder+save_folder) / save_filename)
    psnr_can[i_lambda] = psnr(rec, x_noise, y_noise, False)
  
    im = ax[0,i_lambda].imshow(rec, cmap="gray")
    ax[0,i_lambda].set_title(f'{psnr_can[i_lambda]:.2f} dB', fontsize=fs)
    add_colorbar(im, 'right')
    
    # Hadamard pinv
    save_filename = f'had_{lambda_central}_{N}x{N}.npy'
    rec = np.load(Path(data_folder+save_folder) / save_filename)
    psnr_had[i_lambda] = psnr(rec, x_noise, y_noise, False)
    
    im = ax[1,i_lambda].imshow(rec, cmap="gray")
    ax[1,i_lambda].set_title(f'{psnr_had[i_lambda]:.2f} dB', fontsize=fs)
    add_colorbar(im, 'right')
    
    # Hadamard TiknoNet
    save_filename = f'had_tikhonov_{lambda_central}_{N}x{N}.npy'
    rec = np.load(Path(data_folder+save_folder) / save_filename)
    psnr_had_tikhonov[i_lambda] = psnr(rec, x_noise, y_noise, False)
    
    im = ax[2,i_lambda].imshow(rec, cmap="gray")
    ax[2,i_lambda].set_title(f'{psnr_had_tikhonov[i_lambda]:.2f} dB', fontsize=fs)
    add_colorbar(im, 'right')
    
    # Hadamard TiknoNet
    save_filename = f'had_tikhonet_{lambda_central}_{N}x{N}.npy'
    rec = np.load(Path(data_folder+save_folder) / save_filename)
    psnr_had_tikhonet[i_lambda] = psnr(rec, x_noise, y_noise, False)
    
    im = ax[3,i_lambda].imshow(rec, cmap="gray")
    ax[3,i_lambda].set_title(f'{psnr_had_tikhonet[i_lambda]:.2f} dB', fontsize=fs)
    add_colorbar(im, 'right')
    
    for j in range(4):
        ax[j,i_lambda].set_xticks([])
        ax[j,i_lambda].set_yticks([]) 
    
# Column labels    
method_list = ("Push-broom\nPseudoinverse", 
                "Hadamard\nPseudoinverse", 
                "Hadamard\nTikhonov", 
                "Hadamard\nTikhonov-Net")
for j in range(4):
    ax[j,0].set_ylabel(method_list[j], fontsize=fs)
    
# Row labels    
for i in range(len(lambda_central_list)):
    ax[0,i].set_title(
        fr"$\lambda = {lambda_central_list[i]}$ nm"+'\n'+ax[0,i].get_title(),
        fontsize=fs)

# SNR region    
ax[0,0].add_patch(Rectangle(
    (y_noise[0], x_noise[0]), y_noise[1]-y_noise[0]+1, x_noise[1]-x_noise[0]+1, 
    linewidth=1, edgecolor='r', facecolor='none')
    )

plt.savefig('figure_3.pdf', bbox_inches='tight', dpi=dpi_fig)


with np.printoptions(precision=1, suppress=True):
    print(psnr_had-psnr_can)
    print(psnr_had_tikhonov-psnr_had)
    print(psnr_had_tikhonet-psnr_had)