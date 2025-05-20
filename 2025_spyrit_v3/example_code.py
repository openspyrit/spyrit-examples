"""
This scripts generates plots presented in figure 2 of

SPyRiT: AN OPEN SOURCE PACKAGE FOR SINGLE-PIXEL IMAGING BASED ON DEEP LEARNING
"""

# %%
import torch
import matplotlib.pyplot as plt

from spyrit.core.meas import Linear, LinearSplit, HadamSplit2d
from spyrit.core.noise import Gaussian, Poisson
from spyrit.core.prep import Unsplit, UnsplitRescale
from spyrit.core.inverse import PseudoInverse
from spyrit.core.torch import walsh_matrix_2d

check = True # check with true image

#%% Load image
if check:
    import torchvision
    
    image_path = "./data/images/figure_2/ILSVRC2012_test_00000002.jpeg"
    x = torchvision.io.read_image(image_path, torchvision.io.ImageReadMode.GRAY)
    # Resize image
    x = torchvision.transforms.functional.resize(x, (64, 64))
    # Reshape image
    x = x.reshape(1, 1, 64, 64)
    print(f"Shape of input image: {x.shape}")
    x = x.detach()
    x = x.to(dtype=torch.float)
    x /= 255
    x = x.squeeze()

# %% Linear, noiseless, pseudo-inverse
if not check:
    # 4 images of shape (64, 64) with values in (0,1)
    x = torch.empty(4, 64, 64).uniform_(0, 1)
# Hadamard matrix in "2D" with shape (64*64, 64*64)
H = walsh_matrix_2d(64)
# Linear operator in 2D, working on images with shape (64, 64)
meas_op = Linear(H, (64,64))
# Measurement vectors with shape (4, 4096)
y = meas_op(x)
# Pseudo inverse solution with shape (4, 64, 64)
pinv = PseudoInverse(meas_op)
x_rec = pinv(y)

# plot
if check:
    plt.figure()
    plt.imshow(x_rec.cpu().numpy(), cmap='gray')
    plt.colorbar()

# %% LinearSplit, Gaussian noise, pseudo-inverse

# Linear operator in 2D, working on images with shape (64, 64)
meas_op = LinearSplit(H, (64,64))
# Gaussian noise
meas_op.noise_model = Gaussian(2)
# Measurement vectors with shape (4, 8192)
y = meas_op(x)
# Preprocessed measurement vectors with shape (4, 4096)
prep = Unsplit()
m = prep(y)     # y+ - y-
# Pseudo inverse solution with shape (4, 64, 64)
pinv = PseudoInverse(meas_op)
x_rec = pinv(m)

# plot
if check:
    plt.figure()
    plt.imshow(x_rec.cpu().numpy(), cmap='gray')
    plt.colorbar()

# %% HadamSplit2d, Subsampling x4, Poisson noise, pseudo-inverse

# Low-frequency sampling map with shape (64, 64)
sampling_map = torch.ones((64, 64))
sampling_map[:, 64 // 2 :] = 0
sampling_map[64 // 2 :, :] = 0
meas_op = HadamSplit2d(64, 64**2//4, order=sampling_map, reshape_output=True)
# Poisson noise
meas_op.noise_model = Poisson(100)
# Measurement vectors with shape (4, 2048)
y = meas_op(x)
# Preprocessed measurement vectors with shape (4, 1024)
prep = UnsplitRescale(100)   # (y+ - y-)/alpha 
m = prep(y)
# HadamSplit2d has a fast_pinv() method to get the pseudo inverse solution.
x_rec = meas_op.fast_pinv(m)  # shape is (4, 64, 64)

# plot
if check:
    plt.figure()
    plt.imshow(x_rec.cpu().numpy(), cmap='gray')
    plt.colorbar()