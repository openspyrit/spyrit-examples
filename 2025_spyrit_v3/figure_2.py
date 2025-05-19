"""Python file for generating plots presented in figure 2 in the Optics Express 
main paper:

SPyRiT: AN OPEN SOURCE PACKAGE FOR SINGLE-PIXEL IMAGING BASED ON DEEP LEARNING
"""

# %%
import torch
import torchvision
import matplotlib.pyplot as plt

import spyrit.core.meas as meas
import spyrit.core.noise as noise
import spyrit.core.prep as prep
import spyrit.core.inverse as inverse
import spyrit.core.torch as spytorch

import aux_functions as aux


# %% PARAMS
# Generic parameters for the script
# -----------------------------------------------------------------------------

# Image size
h = 64
# matrix size for display
h_disp = 16
# number of measurements
M = h**2
# subsampling factor, when used
sub_x = 2
sub_y = 2
sub = sub_x * sub_y

# noise parameter (poisson distribution)
alpha = 100

# figure size (in inches)
figsize = (2, 2)
figsize_dbl = (2, 4)
# dots per inch (resolution)
dpi = 300
# measurement domain colormap
cmap_meas = plt.cm.cividis

# set the seed for reproducibility
seed = 404
# -----------------------------------------------------------------------------


# %% GET IMAGE
# Load image
image_path = "./data/images/figure_2/ILSVRC2012_test_00000002.jpeg"
x = torchvision.io.read_image(image_path, torchvision.io.ImageReadMode.GRAY)
# Resize image
x = torchvision.transforms.functional.resize(x, (h, h)).reshape(1, 1, h, h)
print(f"Shape of input image: {x.shape}")

# Select image
x = x.detach().clone()
x = x / 255
b, c, height, width = x.shape

# Plot Ground-truth image
orig_minmax = (0, 1)
aux.imagesc_mod(x[0, 0, :, :], figsize=figsize, dpi=dpi, minmax=orig_minmax)


# %% SQUARE SAMPLING MAP
Sampling_map = torch.ones((h, h))
Sampling_map[:, h // sub_x :] = 0
Sampling_map[h // sub_y :, :] = 0


# %% MEASUREMENT OPERATORS
# define 3 measurement operators :
# 1. Linear, no noise
# 2. LinearSplit, Gaussian noise
# 3. HadamSplit2d, no noise, rectangular subsampling

H = spytorch.walsh_matrix_2d(h)
meas_op1 = meas.Linear(H, (h,h))
meas_op2 = meas.LinearSplit(H, (h,h))
meas_op3 = meas.HadamSplit2d(h, M // sub, order=Sampling_map)

# plot the mask matrices
mask3 = (meas_op3.indices.argsort() < meas_op3.M).reshape(h, h)

# because op2 & op3 are split, we double the masks
mask3 = torch.cat((mask3, mask3), dim=0)

aux.imagesc_mod(mask3.cpu().numpy(), figsize=figsize_dbl, dpi=dpi, minmax=(-1, 1))

# %% MATRICES H
# show 1D walsh-ordered hadamard matrices for each case
H_disp1 = spytorch.walsh_matrix(h_disp)
H_disp2 = torch.cat((H_disp1, -H_disp1), dim=0)
H_disp2[H_disp2 == -1] = 0  # take the positive part

order = [0, 1, 2, 3]  # custom order for the subsampling
zer = torch.zeros((h_disp - len(order), h_disp))
H_disp3 = torch.cat((H_disp1[order], zer, -H_disp1[order], zer), dim=0)
H_disp3[H_disp3 == -1] = 0  # take the positive part

aux.imagesc_mod(H_disp1, figsize=figsize,     dpi=dpi, minmax=(-1, 1))
aux.imagesc_mod(H_disp2, figsize=figsize_dbl, dpi=dpi, minmax=(-1, 1))
aux.imagesc_mod(H_disp3, figsize=figsize_dbl, dpi=dpi, minmax=(-1, 1))


# %% MEASUREMENTS
# Poisson noise
torch.manual_seed(seed) # for reproducibility
meas_op2.noise_model = noise.Gaussian(2)
meas_op3.noise_model = noise.Poisson(alpha)

# measure
y1 = meas_op1(x)
y2 = meas_op2(x)
y3 = meas_op3(x)

# plots
y1_plot = y1.reshape(h, h)
y2_plot = aux.split_meas2img(y2, meas_op2) #meas_op2.unvectorize(y2)
y3_plot = aux.split_meas2img(y3, meas_op3)

# for split measurements, center measurements for better visualization
y2_plot = aux.center_measurements(y2_plot)
y3_plot = aux.center_measurements(y3_plot)

norm = "symlog"
aux.imagesc_mod(y1_plot, figsize=figsize, dpi=dpi, norm=norm, colormap=cmap_meas)
aux.imagesc_mod(y2_plot, figsize=figsize_dbl, dpi=dpi, norm=norm, colormap=cmap_meas)
aux.imagesc_mod(y3_plot, figsize=figsize_dbl, dpi=dpi, norm=norm, colormap=cmap_meas)


# %% PREPROCESSING
# preprocess measurements
prep_op2 = prep.Unsplit()
prep_op3 = prep.UnsplitRescale(alpha)

# Generate measurements using the measurement operators
m1 = y1
m2 = prep_op2(y2)
m3 = prep_op3(y3)

# because we have defined a different sampling order, we call meas2img to
# show the measurements in the same order as the mask
# show the mask as nan for custom color

m2_plot = meas_op2.unvectorize(m2)
m2_plot[m2_plot == 0] = torch.tensor(float("nan"))

m3_plot = spytorch.meas2img(m3, meas_op3.order)
m3_plot[m3_plot == 0] = torch.tensor(float("nan"))

m1_plot = m1.reshape(h, h)
m2_plot = m2_plot.reshape(h, h)
m3_plot = m3_plot.reshape(h, h)

# plot the preprocessed measurements
aux.imagesc_mod(m1_plot, figsize=figsize, dpi=dpi, norm=norm, colormap=cmap_meas)
aux.imagesc_mod(m2_plot, figsize=figsize, dpi=dpi, norm=norm, colormap=cmap_meas)
aux.imagesc_mod(m3_plot, figsize=figsize, dpi=dpi, norm=norm, colormap=cmap_meas)


# %% RECONSTRUCTION
# Reconstruct the image using the measurements (takes a few seconds)
pinv = inverse.PseudoInverse(meas_op1)

z1 = pinv(m1)
z2 = pinv(m2)
z3 = meas_op3.fast_pinv(m3) # We use fast pinv here


# %% PLOT 
aux.imagesc_mod(z1.view(h, h), minmax=orig_minmax, figsize=figsize)#, dpi=dpi, minmax=orig_minmax)
aux.imagesc_mod(z2.view(h, h), minmax=orig_minmax, figsize=figsize)#, dpi=dpi, minmax=orig_minmax)
aux.imagesc_mod(z3.view(h, h), minmax=orig_minmax, figsize=figsize)#, dpi=dpi, minmax=orig_minmax)
