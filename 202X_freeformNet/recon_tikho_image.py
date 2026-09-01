# S-matrix reconstruction with and without denoising networks
# Based on class FreeformLinear
#%%
%matplotlib qt6
import torch
import torchvision
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt

from spyrit.misc.statistics import transform_gray_norm
from spyrit.misc.disp import add_colorbar, noaxis
from spyrit.core.meas import FreeformLinear
from spyrit.core.noise import Gaussian, Poisson, PoissonGaussian
from spyrit.misc.metrics import psnr_torch
from spyrit.misc.walsh_hadamard import walsh_S_matrix, ifwalsh_S_torch
from spyrit.core.inverse import Tikhonov
import spyrit.core.nnet as nnet

h = 128  # image size hxh

i_img = 1
tot   = 1               # total acquisition time in s
alpha = 1e5             # intensity in photons/pixel/s or None 
sigma_meas = 17         # Gaussian std, not used for Poisson noise
noise_type = 'P'        # Noise model among ['G','P','PG']
i_seed = 0

path_image = Path('../../spyrit/tutorial/images/')

masked_type = 'skew'        # Among'skew', 'low', 'all'
N_pixel = 2**12             # only for 'low'

masked_type = 'all'        # Among'skew', 'low', 'all'

fig_folder = Path('figures')

fs = 9                 # Font size
dpi_fig = 600
cbar_pos = 'bottom'     # colorbar position

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

elif masked_type == 'all':
    ind_array = torch.meshgrid(torch.arange(h), torch.arange(h))
    ind_array = tuple(torch.flatten(array).to(device=device) for array in ind_array)

N_pixel = len(ind_array[0])

# mask
mask = torch.zeros((h,h), device=device, dtype=x.dtype)
mask[ind_array] = 1

# masked image
x_mask = x*mask

mean_mask = x[i_img,0,mask==1].mean()

#%%    
fref = sigma_meas**2/tot
fbar = alpha*x[i_img].mean()
fbarref = fbar/fref

print('== Numbers ==')
print(f'fref = {fref} counts')
print(f'fbar = {fbar} counts')
print(f'fbar/fref = {fbarref}')
print(f'N_pixel = {N_pixel}')

mse_sp = mse_split(N_pixel, alpha*x[i_img,0].mean().cpu(), sigma_meas)
mse_rs = mse_raster(N_pixel, alpha*x[i_img,0].mean().cpu(), sigma_meas)
mse_sm = mse_smatrix(N_pixel, alpha*x[i_img,0].mean().cpu(), sigma_meas)

gain_sp = mse_rs - mse_sp
gain_sm = mse_rs - mse_sm

print(f'Expected boost from split Hadamard: {gain_sp:0.2f} dB')
print(f'Expected boost from S-matrix Hadamard: {gain_sp:0.2f} dB')
    
#%% NOISE MODELS
#--- Gaussian or Poisson-Gausssian noise
if noise_type == 'G':
    noise_model = Gaussian(sigma=sigma_meas)
    title = 'Gaussian: '
    
elif noise_type == 'P':
    noise_model = Poisson()
    title = 'Poisson: '
    
elif noise_type == 'PG':
    noise_model = PoissonGaussian(sigma=sigma_meas)
    title = 'Poisson-Gaussian: '
    

#%% Arbitrary shape -- S matrix
from spyrit.misc.walsh_hadamard import walsh_S_matrix, ifwalsh_S_torch

print('== S-matrix 1D ==')

# # Case 1: Full 1D S matrix with ROI
# K = N_pixel-1
# H = torch.from_numpy(walsh_S_matrix(K))
# K_S1d = K

# # Case 1: ROI indices 
# ind_array_0_S = ind_array[0][:-1]
# ind_array_1_S = ind_array[1][:-1]
# x_sub = x[:, :, ind_array_0_S, ind_array_1_S]

# Case 2: S matrix in 2D subsampled by factor of 4, keeping the lowest frequencies.
from spyrit.misc.walsh_hadamard import walsh_matrix_2d
S_full = 0.5*(1-walsh_matrix_2d(h))     # S-matrix with extra row and column of zeros
S_full = S_full.reshape((h, h, h**2))
H = S_full[:h//2,:h//2,:]
H = H.reshape(h**2//4, h**2)
H = H[1:,:]         # removes first row that has only zeros
H = torch.from_numpy(H).to(device=device, dtype=x.dtype)
K = H.shape[0]

# Case 2: No ROI in the case of a subsampled 2D S transform. All pixels are kept.
ind_array_0_S = ind_array[0]
ind_array_1_S = ind_array[1]
x_sub = x[:, :, ind_array_0_S, ind_array_1_S]

# Integration time
dt = tot / K

# counts
norm = alpha*dt

param = fr'$\alpha$ = {alpha:.2}, $\delta t$ = {dt:.1e}, $\sigma$ = {sigma_meas}, $K$={K}'

#--- Simulation
meas = FreeformLinear(H, 
                    meas_shape = (h,h), 
                    index_mask = torch.stack((ind_array_0_S, ind_array_1_S)),
                    noise_model = noise_model,
                    dtype = x.dtype, # Check why default dtype not working here ???
                    device = device) 

#torch.manual_seed(i_seed)
y = meas(norm*x)

print('max of meas:', y.max())

# #--- Case 1: Reconstruction
# x_rec_2 = ifwalsh_S_torch(y)
# x_rec_2 /= norm
# x_pinv = meas.unvectorize(x_rec_2)

#--- Case 2: Reconstruction
H_inv = torch.linalg.pinv(H)
x_rec_2 = torch.einsum('nm,bcm->bcn', H_inv, y)
x_rec_2 /= norm
x_pinv = meas.unvectorize(x_rec_2)

# ground-truth
x_true =  torch.zeros_like(x)
x_true[:,:,ind_array_0_S,ind_array_1_S] = x[:,:,ind_array_0_S,ind_array_1_S]

# PSNR
print(f'PSNRs (in dB): {psnr_torch(x_true,x_pinv).T}')

ppp_pinv = psnr_torch(x_true[i_img,0], x_pinv[i_img,0], 
                        img_dyn = mean_mask, mask = mask).cpu().item()

#%% Tikhonov reconstruction
from spyrit.core.inverse import Tikhonov
cov_path = Path('../20XX_stat/stat/ILSVRC2012_v10102019_ccrop/Cov_im2_128x128.pt')

# covariance of the freeform image. Only row and column corresponding to the 
# freeform pixels are kept.
sigma_im = torch.load(cov_path, map_location=device)  
sigma_im = sigma_im.view((h,h,h,h))  
sigma_im = meas.vectorize(sigma_im)     # vectorize columns
sigma_im = sigma_im.movedim(-1, 0)      # move last dimension to front
sigma_im = meas.vectorize(sigma_im)     # vectorize rows
sigma_im *= (norm**2)     # normalization in agreement with y = meas(norm*x)

# Shrink the covariance matrix. This becomes useful when the noise level 
# decreases (not so clear so far). The shrinkage parameter can be tuned to 
# optimize the reconstruction quality. 

# 1. Vanilla Ledoit-Wolf (isotropic) shrinkage. The covariance matrix is shrunk 
# towards a scaled identity matrix (mean variance).
# http://www.ledoit.net/Review_Paper_2020_JFEc.pdf
# https://scikit-learn.org/stable/modules/covariance.html#shrunk-covariance

shrink = 0.5 # between 0 and 1. 0: no shrinkage, 1: only diagonal elements are kept

sigma_diag = torch.diagonal(sigma_im, dim1=-2, dim2=-1)
mu_shrink = torch.mean(sigma_diag)
# delta_shrink_2 = torch.linalg.matrix_norm(
#           sigma_im - mu_shrink*torch.eye(N_pixel, N_pixel, device=device)
#           )**2

sigma_im = (1-shrink)*sigma_im + shrink*mu_shrink*torch.eye(
                                            N_pixel, N_pixel, device=device)

# 2. Modified Ledoit-Wolf shrinkage. The covariance matrix is shrunk 
# towards the diagonal of the sample covariance matrix.

#sigma_im = (1-shrink)*sigma_im + shrink*torch.diag_embed(sigma_diag)

# covariance of the noise
if noise_type == 'G':
    gamma = (sigma_meas**2)*torch.eye(K, K, device=device)
elif noise_type == 'P':
    gamma = torch.diag_embed(y)
elif noise_type == 'PG':
    gamma = torch.diag_embed(y + sigma_meas**2)

recon = Tikhonov(meas, sigma_im)
x_tik = recon(y, gamma)
x_tik /= norm

# plot
plt.figure()
plt.imshow(x_tik[i_img, 0, :, :].cpu(), cmap="gray")
plt.gca().set_title(f"Tikhonov (shrink = {shrink})", fontsize=fs)
plt.colorbar(location='bottom')

#%% Tikhonov reconstruction with mean prior
x_0 = 0.5       # mean of the freeform image
y_tik = y - x_0 * N_pixel/2   # Note: only works when H is an S-matrix

recon = Tikhonov(meas, sigma_im, reshape_output=False)
x_tik = x_0 + recon(y_tik, gamma)
x_tik = meas.unvectorize(x_tik)
x_tik /= norm

# %%
# Denoiser
# ====================================================================
from collections import OrderedDict
from spyrit.core.prep import Rerange
from spyrit.core.train import load_net

model_folder = "../2025_spyrit_v3/model/"  # reconstruction models
model_folder_full = Path.cwd() / Path(model_folder)

model_dcnet = "dc-net_unet_imagenet_rect_N0_10_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07_light.pth"
model_pinvnet = "pinv-net_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07_retrained_light.pth"

# Load the denoiser
rerange = Rerange((0, 1), (-1, 1))
denoiser = OrderedDict(
    {"rerange": rerange, 
    "denoi": nnet.Unet(), 
    "rerange_inv": rerange.inverse()}
)
denoiser = torch.nn.Sequential(denoiser)
# this function loads the model into the '.denoi' key present in the second
# argument. It fails if it does not find the '.denoi' key.
load_net(model_folder_full / model_dcnet, denoiser, device, False)

# Denoiser evaluation
denoiser.eval()
denoiser.to(device=device)

# Reconstruct
with torch.no_grad():
    x_tik_dcnet = denoiser(x_tik)
    x_pinv_dcnet = denoiser(x_pinv)

# Loads the denoiser of the pinv-net model.
load_net(model_folder_full / model_pinvnet, denoiser, device, False)

# Reconstruct
with torch.no_grad():
    x_tik_pinvnet = denoiser(x_tik)
    x_pinv_pinvnet = denoiser(x_pinv)

# PSNR evaluation for all methods
ppp_pinv = psnr_torch(x_true[i_img,0], x_pinv[i_img,0], 
                        img_dyn = mean_mask, mask = mask).cpu().item()
ppp_tik = psnr_torch(x_true[i_img,0], x_tik[i_img,0], 
                        img_dyn = mean_mask, mask = mask).cpu().item()
ppp_pinv_pinvnet = psnr_torch(x_true[i_img,0], x_pinv_pinvnet[i_img,0], 
                        img_dyn = mean_mask, mask = mask).cpu().item()
ppp_pinv_dcnet = psnr_torch(x_true[i_img,0], x_pinv_dcnet[i_img,0], 
                        img_dyn = mean_mask, mask = mask).cpu().item()
ppp_tik_pinvnet = psnr_torch(x_true[i_img,0], x_tik_pinvnet[i_img,0], 
                        img_dyn = mean_mask, mask = mask).cpu().item()
ppp_tik_dcnet = psnr_torch(x_true[i_img,0], x_tik_dcnet[i_img,0], 
                        img_dyn = mean_mask, mask = mask).cpu().item()

# %% Plot
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

plots = {
    (0, 0): (x_mask[i_img, 0, :, :].cpu(), "Ground truth"),
    (0, 1): (x_pinv[i_img, 0, :, :].cpu(), 
            f"Pseudoinverse ({ppp_pinv:0.2f} dB)"),
    (0, 2): (x_pinv_pinvnet[i_img, 0, :, :].cpu(), 
            f"Pseudoinverse + pinvnet ({ppp_pinv_pinvnet:0.2f} dB)"),
    (0, 3): (x_pinv_dcnet[i_img, 0, :, :].cpu(), 
            f"Pseudoinverse + dcnet ({ppp_pinv_dcnet:0.2f} dB)"),
    (1, 1): (x_tik[i_img, 0, :, :].cpu(), 
            f"Tikhonov ({ppp_tik:0.2f} dB)"),
    (1, 2): (x_tik_pinvnet[i_img, 0, :, :].cpu(), 
            f"Tikhonov + pinvnet ({ppp_tik_pinvnet:0.2f} dB)"),
    (1, 3): (x_tik_dcnet[i_img, 0, :, :].cpu(), 
            f"Tikhonov + dcnet ({ppp_tik_dcnet:0.2f} dB)"),
}

for ax in axes.ravel():
    ax.axis("off")

for (row, col), (img, title) in plots.items():
    im = axes[row, col].imshow(img, cmap="gray")
    axes[row, col].set_title(title, fontsize=fs)
    axes[row, col].axis("off")
    fig.colorbar(im, ax=axes[row, col], location="bottom", fraction=0.046, pad=0.04)

axes[1, 0].set_visible(False)
plt.tight_layout()