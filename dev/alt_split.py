"""
Author: Romain Phan
Date: 21/06/2024
Current spyrit version: 2.3.0
Used Spyrit version: 2.3.0
"""

#%%
# Imports
import os

import torch
import torchvision
import matplotlib.pyplot as plt

from spyrit.misc.disp import imagesc
from spyrit.misc.statistics import transform_gray_norm
from spyrit.core.meas import Linear
from spyrit.core.noise import Poisson
from spyrit.core.prep import DirectPoisson
from spyrit.core.torch import walsh2_matrix


###############################################################################
# This script compares different image reconstructions with split matrices
# given noisy measurements.
#
# Given a Walsh-Hadamard matrix H, it is decomposed into its positive and
# negative parts, H_pos and H_neg as follows:
#     H_pos = max(0, H)
#     H_neg = max(0, -H)
# Note that H = H_pos - H_neg and H_pos as well as H_neg are positive.
#
# Three different methods are compared in this script. Each uses a different
# measurement matrix. Poisson noise is added to the measurements. The three
# matrices are named H1, H2 and H3. They are build as follows:
#     1. H1_i = {H_pos_i if i is even
#               {H_neg_i if i is odd
#       This matrix has double the number of rows as H and has positive values
#
#     2. H2_i = {H_pos_i if i is even
#               {-H_neg_i if i is odd
#       This matrix has double the number of rows as H and has positive and
#       negative values
#
#     3. H3 = H = H_pos - H_neg is the original measurement matrix. 
###############################################################################


#%%
# Set parameters
# ----------------- Parameters ----------------- #
n = 32  # image size nxn
# specify here the relative path to a folder containing a batch of images
image_folder_rel_path = "tutorial/images/"
i = 1  # Image index (modify to change the image)
# ---------------------------------------------- #

# Load an image
spyritPath = os.getcwd()
imgs_path = os.path.join(spyritPath, image_folder_rel_path)

# Create a transform for natural images to normalized grayscale image tensors
transform = transform_gray_norm(img_size=n)
# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=i+1)
# laod all images as one dataset
x, _ = next(iter(dataloader))
# print(f"Shape of input images: {x.shape}")

# Select image number i
x = x[i, :, :, :]
x = x.detach().clone()
b, c, h, w = 1, *x.shape
x = x.view(b*c, h*w)
# x = (x + 1) / 2 # bring image to positive values

# view image
imagesc(x.view(h, w).numpy(), title="Original Image")

#%%
# Create the three measurement matrices and their measurement operators
H = walsh2_matrix(n) # H = H.T
H_pos = torch.nn.functional.relu(H)
H_neg = torch.nn.functional.relu(-H)
H_ones = torch.ones_like(H)

# use these matrices to simulate the measurements
meas_op_pos = Linear(H_pos)
meas_op_neg = Linear(H_neg)
meas_op_ones = Linear(H_ones)
# noisy measurements
alpha = 500
noise_op_pos = Poisson(meas_op_pos, alpha)
noise_op_neg = Poisson(meas_op_neg, alpha)
noise_op_ones = Poisson(meas_op_ones, alpha)
# simulate measurements
m_pos = noise_op_pos(x)
m_neg = noise_op_neg(x)
m_ones = noise_op_ones(x)

# Four measurement matrices used for reconstruction
H1 = H
H2 = torch.cat((H_pos, -H_neg), 1).view(2* n**2, n**2)
H3 = torch.cat((H_pos, H_neg), 1).view(2* n**2, n**2)
H4 = H_pos
H5 = H
# four measurement operators
meas_op1 = Linear(H1)
meas_op2 = Linear(H2)
meas_op3 = Linear(H3)
meas_op4 = Linear(H4)
meas_op5 = Linear(H5)


# %%
# build 5 measurement vectors from the noisy measurements
# these are built by hand so they use the same noise realization
m1 = m_pos - m_neg
m2 = torch.cat((m_pos.T, -m_neg.T), 1).view(1, 2* n**2)
m3 = torch.cat((m_pos.T, m_neg.T), 1).view(1, 2* n**2)
m4 = m_pos
m5 = (m_pos - m_ones).view(1, n**2)

# preprocessing the measurements
y1 = DirectPoisson(alpha, meas_op1)(m1)
y2 = DirectPoisson(alpha, meas_op2)(m2)
y3 = DirectPoisson(alpha, meas_op3)(m3)
y4 = DirectPoisson(alpha, meas_op4)(m4)
y5 = DirectPoisson(alpha, meas_op5)(m5)

# Manual pseudo-inverse reconstruction
x_hat1 = meas_op1.pinv(y1, reg='L1', eta=1e-6)
x_hat2 = meas_op2.pinv(y2, reg='L1', eta=1e-6)
x_hat3 = meas_op3.pinv(y3, reg='L1', eta=1e-6)
x_hat4 = meas_op4.pinv(y4, reg='L1', eta=1e-6)
x_hat5 = meas_op5.pinv(y5, reg='L1', eta=1e-6)

# Create a figure with 6 subplots, one for original image and 5 for reconstructions
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

# Plot original image
axs[0].imshow(x.view(h, w).numpy(), cmap="gray")
# Plot the 5 reconstructions
axs[1].imshow(x_hat1.view(h, w).numpy(), cmap="gray")
axs[2].imshow(x_hat2.view(h, w).numpy(), cmap="gray")
axs[3].imshow(x_hat3.view(h, w).numpy(), cmap="gray")
axs[4].imshow(x_hat4.view(h, w).numpy(), cmap="gray")
axs[5].imshow(x_hat5.view(h, w).numpy(), cmap="gray")

# show a single scale bar
fig.colorbar(axs[0].imshow(x.view(h, w).numpy(), cmap="gray"), ax=axs[0])
fig.colorbar(axs[1].imshow(x_hat1.view(h, w).numpy(), cmap="gray"), ax=axs[1])
fig.colorbar(axs[2].imshow(x_hat2.view(h, w).numpy(), cmap="gray"), ax=axs[2])
fig.colorbar(axs[3].imshow(x_hat3.view(h, w).numpy(), cmap="gray"), ax=axs[3])
fig.colorbar(axs[4].imshow(x_hat4.view(h, w).numpy(), cmap="gray"), ax=axs[4])
fig.colorbar(axs[5].imshow(x_hat5.view(h, w).numpy(), cmap="gray"), ax=axs[5])

# difference between original image and reconstructions
diff1 = torch.linalg.norm(x - x_hat1)
diff2 = torch.linalg.norm(x - x_hat2)
diff3 = torch.linalg.norm(x - x_hat3)
diff4 = torch.linalg.norm(x - x_hat4)
diff5 = torch.linalg.norm(x - x_hat5)
# titles
axs[0].set_title("Original Image")
axs[1].set_title("Using H\nDiff:{:.2f}".format(diff1))
axs[2].set_title("Using [H_pos, -H_neg]\nDiff:{:.2f}".format(diff2))
axs[3].set_title("Using [H_pos, H_neg]\nDiff:{:.2f}".format(diff3))
axs[4].set_title("Using (H+1)/2\nDiff:{:.2f}".format(diff4))
axs[5].set_title("Using H/2\nDiff:{:.2f}".format(diff5))

# suptitle
fig.suptitle(
    "Different image reconstructions and L2-distance to original image",
    fontsize=24)

plt.show()
# %%
