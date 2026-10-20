# -*- coding: utf-8 -*-
"""
The script generates Fig. 5 of the paper

@author: ducros
"""
# -*- coding: utf-8 -*-

#%%
import torch
import torchvision
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt

from spyrit.misc.statistics import transform_gray_norm
from spyrit.misc.disp import add_colorbar, noaxis
from spyrit.core.meas import HadamSplit2d
from spyrit.core.noise import Gaussian, Poisson, PoissonGaussian

h = 128  # image size hxh

i_img = 1
tot   = 1       # total acquisition time in s
alpha = 1e6     # intensity in photons/pixel/s or None 
sigma = 17      # gaussian std
noise_list = ['G','P','PG']     # Noise model
i_seed = 0

path_image = Path('../spyrit/tutorial/images/')

masked_type = 'skew'        #'skew' or 'low'
N_pixel = 2**12             # only for 'low'

fig_folder = Path('figures')

save_tag = True

fs = 9                 # Font size
dpi_fig = 600
cbar_pos = 'bottom'     # colorbar position

if save_tag:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['lines.linewidth'] = 1

#%%
def get_indices_lowest_2d(tensor_2d, n):
  """
  Returns the row and column indices of the n lowest entries in a 2D PyTorch tensor.

  Args:
    tensor_2d: The input 2D PyTorch tensor.
    n: The number of lowest entries to find the indices of.

  Returns:
    A tuple containing two 1D LongTensors:
      - row_indices: The row indices of the n lowest entries.
      - col_indices: The column indices of the n lowest entries.
    If n is greater than or equal to the tensor size, it returns the
    row and column indices of all elements in sorted order.
  """
  if n >= tensor_2d.numel():
    sorted_flattened_indices = torch.argsort(tensor_2d.flatten())
    row_indices = sorted_flattened_indices // tensor_2d.size(1)
    col_indices = sorted_flattened_indices % tensor_2d.size(1)
    return row_indices, col_indices
  else:
    flattened_tensor = tensor_2d.flatten()
    _, flattened_indices = torch.topk(flattened_tensor, k=n, largest=False)
    row_indices = flattened_indices // tensor_2d.size(1)
    col_indices = flattened_indices % tensor_2d.size(1)
    return row_indices, col_indices


def mse_raster(N, f_mean, sigma, t=1, gamma=1):
    
    fref =  sigma**2 / gamma**2 / t
    return 10 * torch.log10(1 / t * (N*f_mean + N**2*fref))

def mse_split(N, f_mean, sigma, M=None, t=1, gamma=1):
    
    if M is None:
        M = N
    fref =  sigma**2 / gamma**2 / t
    return 10 * torch.log10(2 / t * (N*f_mean + 8*M *fref))

def mse_smatrix(N, f_mean, sigma, M=None, t=1, gamma=1):
    
    if M is None:
        M = N
    fref =  sigma**2 / gamma**2 / t
    return 10 * torch.log10(2 / t * (N*f_mean + 2*M *fref))

#%%
# Create a transform for natural images to normalized grayscale image tensors
transform = transform_gray_norm(img_size=h)

# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=path_image, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=7)

x, _ = next(iter(dataloader))
print(f"Shape of input images: {x.shape}")

# Select image
#x = x[i_img : i_img + 5, :, :, :]
x = x.detach().clone()
x = (x+1)/2
print(f'images in ({x.min()}, {x.max()})')

b, c, h, w = x.shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = x.to(device=device)


#%% Define mask
# indices
if masked_type == 'skew':
    start = 45
    mask_width = 64 # a power of two
    f1,l1 = start,start + mask_width
    
    base = np.arange(f1,l1)
    ind = base
    for _ in range(h-1):
        base = base + h-1
        ind = np.append(ind, base)
        
    ind_array = np.unravel_index(ind, (h,h))
    ind_array = tuple(torch.from_numpy(array).to(device=device) for array in ind_array)
    
elif masked_type == 'low':
    ind_array = get_indices_lowest_2d(x[i_img,0], N_pixel)
    
elif masked_type == 'high':
    ind_array = get_indices_lowest_2d(-x[i_img,0], N_pixel)

N_pixel = len(ind_array[0])

# mask
mask = torch.zeros((h,h), device=device, dtype=x.dtype)
mask[ind_array] = 1

# masked image
x_mask = x*mask

mean_mask = x[i_img,0,mask==1].mean()

#%%    
fref = sigma**2/tot
fbar = alpha*x[i_img].mean()
fbarref = fbar/fref

print('== Numbers ==')
print(f'fref = {fref} counts')
print(f'fbar = {fbar} counts')
print(f'fbar/fref = {fbarref}')
print(f'N_pixel = {N_pixel}')

mse_sp = mse_split(N_pixel, alpha*x[i_img,0].mean().cpu(), sigma)
mse_rs = mse_raster(N_pixel, alpha*x[i_img,0].mean().cpu(), sigma)
mse_sm = mse_smatrix(N_pixel, alpha*x[i_img,0].mean().cpu(), sigma)

gain_sp = mse_rs - mse_sp
gain_sm = mse_rs - mse_sm

print(f'Expected boost from split Hadamard: {gain_sp:0.2f} dB')
print(f'Expected boost from S-matrix Hadamard: {gain_sp:0.2f} dB')


#%% create main figure

f, ax = plt.subplots(5,4, figsize=(7,10))
    
#%% LOOP OVER NOISE MODELS
for i_noise, noise in enumerate(noise_list):
    
    print('************')
    print(i_noise, noise)
    print('************')
    
    #--- Gaussian or Poisson-Gausssian noise
    if noise == 'G':
        noise_model = Gaussian(sigma=sigma)
        title = 'Gaussian: '
        
    elif noise == 'P':
        noise_model = Poisson()
        title = 'Poisson: '
        
    elif noise == 'PG':
        noise_model = PoissonGaussian(sigma=sigma)
        title = 'Poisson-Gaussian: '
        
    #%% 2D Hadamard full
    # Available orders: https://oeis.org/A090658
    # Check Table 2 and 3 in Appendix. https://arxiv.org/pdf/2411.18897
    
    meas_op = HadamSplit2d(h, device=device)
    
    #--- Noiseless measurements
    print('== Hadamard 2D full ==')
    y_nonoise = meas_op(x)
    
    # counts
    K = 2*h**2
    dt = tot / K
    K_H2dF = K
    
    # counts
    norm = alpha*dt
    
    #--- Simulation
    meas_op = HadamSplit2d(h, device=device, noise_model = noise_model)
    param = fr'$\alpha$ = {alpha:.2}, $\delta t$ = {dt:.1e}, $\sigma$ = {sigma}, $K$={K}'
    
    torch.manual_seed(i_seed)
    y = meas_op(norm*x)
    
    print('max of meas:', y.max())
    
    #--- Pseudo-inverse reconstruction operator
    from spyrit.core.prep import Unsplit
    prep = Unsplit()
    prep = prep.to(device=device)
    
    x_H2dF = meas_op.fast_pinv(prep(y))
    x_H2dF /= norm
    
    #--- PSNR
    from spyrit.misc.metrics import psnr_torch
    
    # error
    err_H2dF = x - x_H2dF
    
    # PSNR in mask
    # ground-truth
    ppp_H2dF = psnr_torch(x[i_img,0], 
                          x_H2dF[i_img,0], 
                          img_dyn = mean_mask, 
                          mask = mask
                          ).cpu().item()
    print(ppp_H2dF)
    
    #%% Arbitrary shape -- Identity matrix 1D
    from spyrit.core.meas import FreeformLinear
    
    print('== Raster Scan ==')
    
    # Hadamard
    H = torch.eye(N_pixel)
    
    # forward
    K = N_pixel
    dt = tot / K
    K_I1d = K
    
    # counts
    norm = alpha*dt
    
    #--- Simulation
    meas_1d = FreeformLinear(H,
                    meas_shape = (h,h), 
                    index_mask = torch.stack(ind_array),
                    noise_model = noise_model,
                    dtype = x.dtype, # Check why default dtype not working here ???
                    device = device
                    )
    
    torch.manual_seed(i_seed)
    y = meas_1d(norm*x)
    y_clean = meas_1d.measure(norm*x)
    y_std = torch.sqrt(torch.maximum(y_clean, torch.tensor(1.0)))
    
    print('max of meas:', y.max())
    
    # Direct reconstruction operator
    x_rec_2 = y         # x_rec_2 = y / y_std
    x_rec_2 /= norm
    
    x_I1d = torch.zeros_like(x)
    x_I1d[:,:,ind_array[0],ind_array[1]] = x_rec_2
    
    # ground-truth
    x_true =  x_mask
    
    # PSNR
    print(f'PSNRs (in dB): {psnr_torch(x_true,x_I1d).T}')
    
    # Error
    err_I1d = x_true - x_I1d
    ppp_I1d = psnr_torch(x_true[i_img,0], 
                         x_I1d[i_img,0], 
                         img_dyn = mean_mask, 
                         mask = mask
                         ).cpu().item()
        
    #%% Arbitrary shape -- Hadamard matrix 1D
    from spyrit.core.torch import walsh_matrix
    from spyrit.core.meas import FreeformLinearSplit
    
    print('== Hadamard 1D ==')
    
    # Hadamard
    H = walsh_matrix(N_pixel)
    
    # forward
    K = 2*N_pixel
    dt = tot / K
    K_H1d = K
    
    # counts
    norm = alpha*dt
    
    param = fr'$\alpha$ = {alpha:.2}, $\delta t$ = {dt:.1e}, $\sigma$ = {sigma}, $K$={K}'
    
    #--- Simulation
    meas_1d = FreeformLinearSplit(H, 
                        meas_shape = (h,h), 
                        index_mask = torch.stack(ind_array),
                        noise_model = noise_model,
                        dtype = x.dtype, # Check why default dtype not working here ???
                        device = device)
    
    torch.manual_seed(i_seed)
    y = meas_1d(norm*x)
    
    print('max of meas:', y.max())
    
    #--- Reconstrcution
    from spyrit.core.torch import ifwht
    prep = Unsplit()
    
    x_rec_2 = ifwht(prep(y)) 
    x_rec_2 /= norm
    
    x_H1d = torch.zeros_like(x)
    x_H1d[:,:,ind_array[0],ind_array[1]] = x_rec_2
    
    # Ground-truth
    x_true =  x_mask
    
    # PSNR
    print(f'PSNRs (in dB): {psnr_torch(x_true,x_H1d).T}')
    
    # Error
    err_H1d = x_true - x_H1d
    ppp_H1d = psnr_torch(x_true[i_img,0],
                         x_H1d[i_img,0], 
                         img_dyn = mean_mask, 
                         mask=mask).cpu().item()
    
    #%% Arbitrary shape -- S matrix
    from spyrit.misc.walsh_hadamard import walsh_S_matrix, ifwalsh_S_torch
    
    print('== S-matrix 1D ==')
    
    # S matrix
    K = N_pixel-1
    H = torch.from_numpy(walsh_S_matrix(K))
    K_S1d = K
    
    # Integration time
    dt = tot / K
    
    # ROI indices
    ind_array_0_S = ind_array[0][:-1]
    ind_array_1_S = ind_array[1][:-1]
    x_sub = x[:, :, ind_array_0_S, ind_array_1_S]
    
    # counts
    norm = alpha*dt
    
    param = fr'$\alpha$ = {alpha:.2}, $\delta t$ = {dt:.1e}, $\sigma$ = {sigma}, $K$={K}'
    
    #--- Simulation
    meas_1d = FreeformLinear(H, 
                        meas_shape = (h,h), 
                        index_mask = torch.stack((ind_array_0_S, ind_array_1_S)),
                        noise_model = noise_model,
                        dtype = x.dtype, # Check why default dtype not working here ???
                        device = device) 
    
    torch.manual_seed(i_seed)
    y = meas_1d(norm*x)
    
    print('max of meas:', y.max())
    
    #--- Reconstrcution
    x_rec_2 = ifwalsh_S_torch(y)
    x_rec_2 /= norm
    
    x_S1d = torch.zeros_like(x)
    x_S1d[:,:,ind_array_0_S,ind_array_1_S] = x_rec_2
    
    # ground-truth
    x_true =  torch.zeros_like(x)
    x_true[:,:,ind_array_0_S,ind_array_1_S] = x[:,:,ind_array_0_S,ind_array_1_S]
    
    # PSNR
    print(f'PSNRs (in dB): {psnr_torch(x_true,x_S1d).T}')
    
    # plot
    err_S1d = x_true - x_S1d
    ppp_S1d = psnr_torch(x_true[i_img,0], 
                         x_S1d[i_img,0], 
                         img_dyn = mean_mask, 
                         mask = mask
                         ).cpu().item()
    
    #%% Masked 2D Hadamard 
    print('== Hadamard 2D masked ==')
    
    # counts
    K = 2*h**2
    dt = tot / K
    K_H2dM = K
    
    # counts
    norm = alpha*dt
    
    param = fr'$\alpha$ = {alpha:.2}, $\delta t$ = {dt:.1e}, $\sigma$ = {sigma}, $K$={K}'
    
    # Simulation
    meas_op = HadamSplit2d(h, device=device, noise_model=noise_model)
    
    torch.manual_seed(i_seed)
    y = meas_op(norm*x_mask)
    
    print('max of meas:', y.max())
    
    # Preprocessing operator
    prep = Unsplit().to(device=device)
    
    # Pseudo inverse reconstruction
    x_H2dM = meas_op.fast_pinv(prep(y))
    x_H2dM /= norm
    
    # Masking
    x_H2dM = x_H2dM * mask
    
    # Ground-truth
    x_true =  x_mask
    
    # Error 
    err_H2dM = x_true - x_H2dM
    ppp_H2dM = psnr_torch(x_true[i_img,0],
                          x_H2dM[i_img,0], 
                          img_dyn = mean_mask, 
                          mask = mask
                          ).cpu().item() #, img_dyn=1.0, mask=mask
        
    #%% plot all    
    im = ax[0,i_noise+1].imshow(x_H2dF[i_img, 0, :, :].cpu(), cmap="gray")
    ax[0,i_noise+1].set_title(f"{title} \n FH2 ({ppp_H2dF:0.1f} dB)", fontsize=fs)
    cbar = add_colorbar(im, cbar_pos)
    cbar.ax.tick_params(labelsize=fs-3)
    
    im = ax[1,i_noise+1].imshow(x_I1d[i_img, 0, :, :].cpu(), cmap="gray")
    ax[1,i_noise+1].set_title(f"RS ({ppp_I1d:0.1f} dB)", fontsize=fs)
    cbar = add_colorbar(im, cbar_pos)
    cbar.ax.tick_params(labelsize=fs-3)
    
    im = ax[2,i_noise+1].imshow(x_H2dM[i_img, 0, :, :].cpu(), cmap="gray")
    ax[2,i_noise+1].set_title(f"MH2 ({ppp_H2dM:0.1f} dB)", fontsize=fs)
    cbar = add_colorbar(im, cbar_pos)
    cbar.ax.tick_params(labelsize=fs-3)
    
    im = ax[3,i_noise+1].imshow(x_H1d[i_img, 0, :, :].cpu(), cmap="gray")
    ax[3,i_noise+1].set_title(f"H1 ({ppp_H1d:0.1f} dB)", fontsize=fs)
    cbar = add_colorbar(im, cbar_pos)
    cbar.ax.tick_params(labelsize=fs-3)
    
    im = ax[4,i_noise+1].imshow(x_S1d[i_img, 0, :, :].cpu(), cmap="gray")
    ax[4,i_noise+1].set_title(f"S1 ({ppp_S1d:0.1f} dB)", fontsize=fs)
    cbar = add_colorbar(im, cbar_pos)
    cbar.ax.tick_params(labelsize=fs-3)  
    
    #plt.pause(10)
    
#%% final plot and save

im = ax[0,0].imshow(x[i_img,0,:,:].cpu(), cmap="gray")
ax[0,0].set_title("Original image", fontsize=fs)
cbar = add_colorbar(im, cbar_pos)
cbar.ax.tick_params(labelsize=fs-3)

im = ax[1,0].imshow(x_mask[i_img,0,:,:].cpu(), cmap="gray")
ax[1,0].set_title("Masked image", fontsize=fs)
cbar = add_colorbar(im, cbar_pos)
cbar.ax.tick_params(labelsize=fs-3)

ax[2,0].set_visible(False)
ax[3,0].set_visible(False)
ax[4,0].set_visible(False)

noaxis(ax[0])
noaxis(ax[1])
noaxis(ax[2])
noaxis(ax[3])
noaxis(ax[4])

plt.tight_layout()

if save_tag:
    plt.savefig(fig_folder / 'figure_6.pdf', transparent=True, dpi=600)

print(f'Actual boost from split Hadamard: {ppp_H1d-ppp_I1d:0.2f} dB')
print(f'Actual boost from S-matrix Hadamard: {ppp_S1d-ppp_I1d:0.2f} dB')