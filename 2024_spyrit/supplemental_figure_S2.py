# %% 
# Imports
# --------------------------------------------------------------------
from pathlib import Path

import torch
import torchvision
import matplotlib.pyplot as plt

import spyrit.core.meas as meas
import spyrit.core.noise as noise
import spyrit.core.prep as prep
import spyrit.misc.statistics as stats
import utility_dpgd as dpgd


# %% 
# General
# --------------------------------------------------------------------
# Experimental data
image_folder = 'data/images/'       # images for simulated measurements
model_folder = 'model/'             # reconstruction models
stat_folder  = 'stat/'              # statistics
recon_folder = 'recon/supplemental_figure_S2/'    # reconstructed images

# Full paths
image_folder_full = Path.cwd() / Path(image_folder)
model_folder_full = Path.cwd() / Path(model_folder)
stat_folder_full  = Path.cwd() / Path(stat_folder)
recon_folder_full = Path.cwd() / Path(recon_folder)
recon_folder_full.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# %% 
# Load images
# --------------------------------------------------------------------

img_size = 128 # image size

print("Loading image...")
# crop to desired size, set to black and white, normalize
transform = stats.transform_gray_norm(img_size)

# define dataset and dataloader. `image_folder_full` should contain
# a class folder with the images
dataset = torchvision.datasets.ImageFolder(
    image_folder_full, 
    transform=transform
    )

dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=2, 
    shuffle=False
    )

# select the two images
x, _ = next(iter(dataloader))
x_dog = x[1]
c, h, w = x_dog.shape
print("Image shape:", x_dog.shape)

x_dog_plot = x_dog.view(-1, h, h).cpu().numpy()
# save image as original
plt.imsave(recon_folder_full / f'sim1_{img_size}_gt.png', x_dog_plot[0, :, :], cmap='gray')


# %% 
# Simulate measurements for three image intensities
# --------------------------------------------------------------------
# Measurement parameters
alpha_list = [2, 10, 50] # Poisson law parameter for noisy image acquisitions
n_alpha = len(alpha_list)
M = img_size**2 // 4  # Number of measurements (here, 1/4 of the pixels)

# Measurement and noise operators
Ord_rec = torch.ones((img_size, img_size))
Ord_rec[:,img_size//2:] = 0
Ord_rec[img_size//2:,:] = 0

meas_op = meas.HadamSplit(M, h, Ord_rec)
noise_op = noise.Poisson(meas_op, alpha_list[0])
prep_op = prep.SplitPoisson(alpha_list[0], meas_op)

# Vectorized images
x_dog = x_dog.view(1, h * w)

# Measurement vectors
y_dog = torch.zeros(n_alpha, 2*M)

for ii, alpha in enumerate(alpha_list): 
    noise_op.alpha = alpha
    torch.manual_seed(0) # for reproducibility
    # only need to measure, preprocessing is done in the dpgd, later
    y_dog[ii,:] = noise_op(x_dog)

# Send to GPU if available
y_dog = y_dog.to(device)


# %% 
# DPGD-PnP
# ====================================================================
# load denoiser
n_channel, n_feature, n_layer = 1, 100, 20
model_name = 'DFBNet_l1_patchsize=50_varnoise0.1_feat_100_layers_20.pth'
denoi = dpgd.load_model(pth = (model_folder_full / model_name).as_posix(), 
                    n_ch = n_channel, 
                    features = n_feature, 
                    num_of_layers = n_layer)

denoi.module.update_lip((1,50,50))
denoi.eval()

# Reconstruction hyperparameters
gamma = 1/img_size**2
max_iter = 101
mu_list = {
    2: [4000, 5500, 6000, 8000, 10000],
    10: [1000, 2500, 3000, 4000, 10000],
    50: [500, 1000, 1500, 2000, 4000]
}
crit_norm = 1e-4

# Init 
dpgdnet = dpgd.DualPGD(noise_op, prep_op, denoi, gamma, 10000, max_iter, crit_norm)
dpgdnet.to(device)
x_dog_dpgd = torch.zeros(1, 1, img_size, img_size)

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):
        
        # set noise level for measurement operator
        dpgdnet.prep.alpha = alpha
        # set noise level for PnP denoiser
        for mu in mu_list[alpha]:
            dpgdnet.mu = mu
            x_dog_dpgd =  dpgdnet.reconstruct(y_dog[ii:ii+1, :])

            # save
            filename = f'sim1_{img_size}_N0_{alpha}_M_{M}_rect_dfb-net_dfb_mu_{mu}.png'
            full_path = recon_folder_full / filename
            plt.imsave(full_path, x_dog_dpgd[0,0].cpu().detach().numpy(), cmap='gray')

# %%
