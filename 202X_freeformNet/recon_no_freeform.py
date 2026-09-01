# Hadamard split reconstruction with and without denoising networks
# Based on class HadamSplit2d

#%%
%matplotlib qt6
import torch
import torchvision
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt

from spyrit.misc.statistics import transform_gray_norm
from spyrit.misc.disp import add_colorbar, noaxis
from spyrit.core.meas import Linear
from spyrit.core.noise import Gaussian, Poisson, PoissonGaussian
from spyrit.misc.metrics import psnr_torch
from spyrit.misc.walsh_hadamard import walsh_S_matrix, ifwalsh_S_torch
from spyrit.core.inverse import Tikhonov
import spyrit.core.nnet as nnet

h = 128  # image size hxh

i_img = 1
tot   = 1               # total acquisition time in s
alpha = 1e5             # intensity in photons/pixel/s or None 
sigma_gauss = 17         # Gaussian std, not used for Poisson noise
noise_type = 'N'        # Noise model among ['G','P','PG']
i_seed = 0

path_image = Path('../../spyrit/tutorial/images/')

fig_folder = Path('figures')

fs = 9                 # Font size
dpi_fig = 600
cbar_pos = 'bottom'     # colorbar position

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

device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = x.to(device=device)
    
#%% NOISE MODELS
#--- Gaussian or Poisson-Gausssian noise
if noise_type == 'G':
    noise_model = Gaussian(sigma=sigma_gauss)
    title = 'Gaussian: '
    
elif noise_type == 'P':
    noise_model = Poisson()
    title = 'Poisson: '
    
elif noise_type == 'PG':
    noise_model = PoissonGaussian(sigma=sigma_gauss)
    title = 'Poisson-Gaussian: '

else:
    noise_model = torch.nn.Identity()

#%% 
# Send to GPU if available
from spyrit.core.meas import HadamSplit2d

M = h**2//4

Ord_rec = torch.ones(h, h)
Ord_rec[:, h // 2 :] = 0
Ord_rec[h // 2 :, :] = 0
meas = HadamSplit2d(h, M, Ord_rec, noise_model=noise_model, device=device)
meas = meas.to(device=device)

# Integration time
K = 2*M
dt = tot / K

# counts
norm = alpha*dt

#torch.manual_seed(i_seed)
y = meas(norm*x)
print('max of meas:', y.max())

from spyrit.core.prep import Unsplit
prep = Unsplit()

#--- Case 2: Reconstruction
H_pinv = meas.H.T/h**2
x_rec_2 = torch.einsum('nm,bcm->bcn', H_pinv, prep(y))
x_rec_2 /= norm
x_pinv = meas.unvectorize(x_rec_2)

# ground-truth
x_true = x

# PSNR
print(f'PSNRs (in dB): {psnr_torch(x_true,x_pinv).T}')

ppp_pinv = psnr_torch(x_true[i_img,0], x_pinv[i_img,0]).cpu().item()

#%% DC reconstruction
from spyrit.core.recon import DCNet
from spyrit.core.prep import UnsplitRescale
cov_path = Path('../20XX_stat/stat/ILSVRC2012_v10102019_ccrop/Cov_im2_128x128.pt')

# covariance in the measurement domain
meas_full  = HadamSplit2d(h, h**2, None, device=device)
meas_full  = meas_full.to(device=device)

sigma_im   = torch.load(cov_path, map_location=device)  # image domain covariance matrix
sigma_im   = sigma_im.view((h,h,h,h))
sigma_meas = meas_full.measure_H(sigma_im)   # H @ Sigma
sigma_meas = sigma_meas.movedim(-1, 0)       # (H @ Sigma)^T
sigma_meas = meas_full.measure_H(sigma_meas) # H @ (H @ Sigma)^T
sigma_meas = sigma_meas.T                    # (H @ (H @ Sigma)^T)^T = H @ Sigma @ H^T

sigma_im   = sigma_im.view((h**2,h**2))

del meas_full

sigma_meas *= (norm**2)     # normalization in agreement with y = meas(norm*x)
                            # TODO: check if this is correct, or if preprocessing
                            # already solved this
prep_dc = UnsplitRescale(norm)
dcnet = DCNet(meas, prep_dc, sigma_meas, device=device)
dcnet.eval()

x_dc = dcnet.reconstruct(y)

# plot
plt.figure()
plt.imshow(x_dc[i_img, 0, :, :].cpu(), cmap="gray")
plt.gca().set_title(f"Denoised completion", fontsize=fs)
plt.colorbar(location='bottom')

#%% Tikhonov reconstruction with mean prior
recon = Tikhonov(meas, sigma_im, reshape_output=False)

# covariance of the noise
if noise_type == 'G':
    gamma = (sigma_gauss**2)*torch.eye(M, M, device=device)
elif noise_type == 'P':
    gamma = torch.diag_embed(y)
elif noise_type == 'PG':
    gamma = torch.diag_embed(y + sigma_gauss**2)

# x_0 = 0.5                   # mean of the freeform image
y_tik = prep_dc(y)
#y_tik = y_tik - x_0 * h**2//2   # Note: works when H is a split Hadamard matrix
#y_tik[:,:,0] = x_0 * h**2//2
#x_tik = x_0 + recon(y_tik, gamma)
x_tik = recon(y_tik, gamma)
x_tik = meas.unvectorize(x_tik)

#x_tik /= norm

# plot
plt.figure()
plt.imshow(x_tik[i_img, 0, :, :].cpu(), cmap="gray")
plt.gca().set_title(f"Denoised completion", fontsize=fs)
plt.colorbar(location='bottom')

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
    x_dc_dcnet = denoiser(x_dc)
    x_tik_dcnet = denoiser(x_tik)
    x_pinv_dcnet = denoiser(x_pinv)

# Loads the denoiser of the pinv-net model.
load_net(model_folder_full / model_pinvnet, denoiser, device, False)

# Reconstruct
with torch.no_grad():
    x_dc_pinvnet = denoiser(x_dc)
    x_tik_pinvnet = denoiser(x_tik)
    x_pinv_pinvnet = denoiser(x_pinv)

# PSNR evaluation for all methods
ppp_pinv = psnr_torch(x_true[i_img,0], x_pinv[i_img,0]).cpu().item()
ppp_tik = psnr_torch(x_true[i_img,0], x_tik[i_img,0]).cpu().item()
ppp_dc = psnr_torch(x_true[i_img,0], x_dc[i_img,0]).cpu().item()

ppp_pinv_pinvnet = psnr_torch(x_true[i_img,0], x_pinv_pinvnet[i_img,0]).cpu().item()
ppp_tik_pinvnet = psnr_torch(x_true[i_img,0], x_tik_pinvnet[i_img,0]).cpu().item()
ppp_dc_pinvnet = psnr_torch(x_true[i_img,0], x_dc_pinvnet[i_img,0]).cpu().item()

ppp_pinv_dcnet = psnr_torch(x_true[i_img,0], x_pinv_dcnet[i_img,0]).cpu().item()    
ppp_tik_dcnet = psnr_torch(x_true[i_img,0], x_tik_dcnet[i_img,0]).cpu().item()
ppp_dc_dcnet = psnr_torch(x_true[i_img,0], x_dc_dcnet[i_img,0]).cpu().item()

# %% Plot
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

plots = {
    (0, 0): (x[i_img, 0, :, :].cpu(), "Ground truth"),
    (0, 1): (x_pinv[i_img, 0, :, :].cpu(), 
            f"Pseudoinverse ({ppp_pinv:0.2f} dB)"),
    (0, 2): (x_pinv_pinvnet[i_img, 0, :, :].cpu(), 
            f"Pseudoinverse + pinvnet ({ppp_pinv_pinvnet:0.2f} dB)"),
    (0, 3): (x_pinv_dcnet[i_img, 0, :, :].cpu(), 
            f"Pseudoinverse + dcnet ({ppp_pinv_dcnet:0.2f} dB)"),
    (1, 1): (x_tik[i_img, 0, :, :].cpu(), 
            f"Denoised completion ({ppp_tik:0.2f} dB)"),
    (1, 2): (x_tik_pinvnet[i_img, 0, :, :].cpu(), 
            f"Denoised completion + pinvnet ({ppp_tik_pinvnet:0.2f} dB)"),
    (1, 3): (x_tik_dcnet[i_img, 0, :, :].cpu(), 
            f"Tikhonov + dcnet ({ppp_tik_dcnet:0.2f} dB)"),
    (2, 1): (x_dc[i_img, 0, :, :].cpu(), 
            f"Denoised completion ({ppp_dc:0.2f} dB)"),
    (2, 2): (x_dc_pinvnet[i_img, 0, :, :].cpu(), 
            f"Denoised completion + pinvnet ({ppp_dc_pinvnet:0.2f} dB)"),
    (2, 3): (x_dc_dcnet[i_img, 0, :, :].cpu(), 
            f"Denoised completion + dcnet ({ppp_dc_dcnet:0.2f} dB)"),
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
