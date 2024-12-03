"""Script that analyses the superresolution effect and its limitation. 

Superresolution happens when a moving object is mesured at different times,
leading to a higher resolution than the one of the camera. A Siemens star is
used.

An original image of 80x80 pixels is used, and then measured at different
scales using subsampling at 64x64 resolution (and not directly at a 32x32
resolution for instance). The reconstructed image is thus at 80x80 resolution.

This simulation is done using a fixed size image (80x80 pixels) and simulating
its movement across multiple frames. The number of frames depends on the number
of measurements, so that the first and last frame are always identical (i.e.
represent the same overall movement, but with a different sampling rate). See
_fig2 for a simulation with a finer image, deformed and then subsampled to
the measurement size.
"""

# %%

import time
import pathlib

import torch
import torchvision
import matplotlib.pyplot as plt
# import matplotlib.animation as animation

import spyrit.core.warp as warp


# %%
## PARAMETERS
# =============================================================================
# image
image_size = 128 # reconstruction size
image_shape = (image_size, image_size)
n_branches = 16 # number of branches in the Siemens star

# deformation
torch_seed = 0
n_frames = 2 * 64*64 # *2 because of splitting
deform_mode = 'bilinear' # choose between 'bilinear' and 'bicubic'
alpha = 150 # magnitude of displacements 
sigma = 10 # smoothness of displacements
spacing = 3 # defines the distance in frames between 2 generated elastic deformations

# where to load the reference image from
load_path = pathlib.Path(r"C:\Users\phan\Documents\SPYRIT\deep_dyn_recon\reference")
# where to save figs
save_path = pathlib.Path(r"C:\Users\phan\Documents\SPYRIT\deep_dyn_recon\deformations")

# use gpu ?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # force cpu
print(f"Using device: {device}")
# =============================================================================


# %%
# Load the image

image_name = f"siemens_star_{image_size}x{image_size}_{n_branches}_branches.png"
image_path = load_path / image_name

if not image_path.exists():
    from aux_functions import save_siemens_star
    print(f"Image {image_name} not found in {load_path}, creating it")
    save_siemens_star(image_path, figsize=image_size, n=n_branches)

x = torchvision.io.read_image(load_path / image_name, torchvision.io.ImageReadMode.GRAY)
print("Successfully loaded Siemens star")

# rescale x from [0, 255] to [-1, 1]
x = x.detach().float().to(device)
x = 2 * (x / 255) - 1
c, h, w = x.shape
print(f"Shape of input image: {x.shape}")


# %%
# warp the image and create an animation object
torch.manual_seed(torch_seed)
Elastic = warp.ElasticDeformation(alpha, sigma, image_shape, n_frames, spacing).to(device)
warped_img = Elastic(x, mode=deform_mode)

# show 4 sample images
samples = [0, n_frames//4, n_frames//2, (3*n_frames)//4]
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for i in range(4):
    axs[i//2, i%2].imshow(warped_img[samples[i], 0, :, :].cpu(), cmap='gray')
    axs[i//2, i%2].set_title(f"Frame {samples[i]}")
plt.show()


# %%
# Save deformation field if all is correct

save_name = f"ElasticDeform_{image_size}px_{n_frames}frames_{alpha}alpha_{sigma}sigma_{spacing}spacing_{torch_seed}seed"
# save warp as fig and deformation field
save_datetime = time.strftime("%Y%m%d-%H%M%S")
torch.save(Elastic, save_path / f"{save_name}.pt")
print("Elastic deformation saved in", save_path / f"{save_name}.pt")

# fig, ax = plt.subplots()
# ax.imshow(warped_img[0, 0, :, :].cpu(), cmap='gray')

# def update_frame(frame_num):
#     ax.imshow(warped_img[int(frame_num), 0, :, :].cpu(), cmap='gray')
#     fig.canvas.draw_idle()

# ani = animation.FuncAnimation(fig, update_frame, frames=warped_img.shape[0], interval=1000/24, repeat=True)
# ani.save(save_path / f"{save_name}.gif", writer='pillow', fps=60)


# %%
