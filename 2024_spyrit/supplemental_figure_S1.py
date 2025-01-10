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
import spyrit.core.recon as recon
import spyrit.misc.statistics as stats
import spyrit.external.drunet as drunet


# %%
# General
# --------------------------------------------------------------------
# Experimental data
image_folder = "data/images/"  # images for simulated measurements
model_folder = "model/"  # reconstruction models
stat_folder = "stat/"  # statistics
recon_folder = "recon/supplemental_figure_S1/"  # reconstructed images

# Full paths
image_folder_full = Path.cwd() / Path(image_folder)
model_folder_full = Path.cwd() / Path(model_folder)
stat_folder_full = Path.cwd() / Path(stat_folder)
recon_folder_full = Path.cwd() / Path(recon_folder)
recon_folder_full.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# %%
# Load images
# --------------------------------------------------------------------

img_size = 128  # image size

print("Loading image...")
# crop to desired size, set to black and white, normalize
transform = stats.transform_gray_norm(img_size)

# define dataset and dataloader. `image_folder_full` should contain
# a class folder with the images
dataset = torchvision.datasets.ImageFolder(image_folder_full, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False)

# select the two images
x, _ = next(iter(dataloader))
x_dog, x_panther = x[1], x[2]
c, h, w = x_dog.shape
print("Image shape:", x_dog.shape)

x_dog_plot = x_dog.view(-1, h, h).cpu().numpy()
x_panther_plot = x_panther.view(-1, h, h).cpu().numpy()
# save image as original
plt.imsave(
    recon_folder_full / f"sim1_{img_size}_gt.png", x_dog_plot[0, :, :], cmap="gray"
)
plt.imsave(
    recon_folder_full / f"sim2_{img_size}_gt.png", x_panther_plot[0, :, :], cmap="gray"
)


# %%
# Simulate measurements for three image intensities
# --------------------------------------------------------------------
# Measurement parameters
alpha_list = [2, 10, 50]  # Poisson law parameter for noisy image acquisitions
n_alpha = len(alpha_list)
M = img_size**2 // 4  # Number of measurements (here, 1/4 of the pixels)

# Measurement and noise operators
Ord_rec = torch.ones((img_size, img_size))
Ord_rec[:, img_size // 2 :] = 0
Ord_rec[img_size // 2 :, :] = 0

meas_op = meas.HadamSplit(M, h, Ord_rec)
noise_op = noise.Poisson(meas_op, alpha_list[0])
prep_op = prep.SplitPoisson(2, meas_op)

# Vectorized images
x_dog = x_dog.view(1, h * w)
x_panther = x_panther.view(1, h * w)

# Measurement vectors
y_dog = torch.zeros(n_alpha, 2 * M)
y_panther = torch.zeros(n_alpha, 2 * M)

for ii, alpha in enumerate(alpha_list):
    noise_op.alpha = alpha
    torch.manual_seed(0)  # for reproducibility
    y_dog[ii, :] = noise_op(x_dog)
    torch.manual_seed(0)  # for reproducibility
    y_panther[ii, :] = noise_op(x_panther)

# Send to GPU if available
y_dog = y_dog.to(device)
y_panther = y_panther.to(device)


# %%
# Pinv - PnP
# ====================================================================
model_name = "drunet_gray.pth"
noise_levels = {
    2: [80, 90, 100, 110, 115],
    10: [20, 25, 30, 40, 50],
    50: [10, 15, 20, 25, 30],
}  # noise levels from 0 to 255 for each alpha for PnP

# Initialize network
denoi = drunet.DRUNet()
pinvnet = recon.PinvNet(noise_op, prep_op, denoi)
pinvnet.eval()

# load_net(model_folder_full / model_name, pinvnet, device, False)
pinvnet.denoi.load_state_dict(
    torch.load(model_folder_full / model_name, weights_only=True), strict=False
)
pinvnet.denoi.eval()
pinvnet = pinvnet.to(device)

# Reconstruct and save
x_pinvnet = torch.zeros(1, 1, img_size, img_size)

with torch.no_grad():
    for ii, alpha in enumerate(alpha_list):

        # set noise level for measurement operator
        pinvnet.prep.alpha = alpha
        # set noise level for PnP denoiser
        for nu in noise_levels[alpha]:

            pinvnet.denoi.set_noise_level(nu)
            x_dog_pinvnet = pinvnet.reconstruct(y_dog[ii : ii + 1, :])
            x_panther_pinvnet = pinvnet.reconstruct(y_panther[ii : ii + 1, :])

            # save
            filename_dog = (
                f"sim1_{img_size}_N0_{alpha}_M_{M}_rect_pinv-net_drunet_nlevel_{nu}.png"
            )
            filename_panther = (
                f"sim2_{img_size}_N0_{alpha}_M_{M}_rect_pinv-net_drunet_nlevel_{nu}.png"
            )
            full_path_dog = recon_folder_full / filename_dog
            full_path_panther = recon_folder_full / filename_panther

            plt.imsave(
                full_path_dog, x_dog_pinvnet[0, 0].cpu().detach().numpy(), cmap="gray"
            )
            plt.imsave(
                full_path_panther,
                x_panther_pinvnet[0, 0].cpu().detach().numpy(),
                cmap="gray",
            )

# %%
