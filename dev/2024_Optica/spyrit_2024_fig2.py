"""Python file for generating plots presented in figure 2 in the paper:

SPyRiT: AN OPEN SOURCE PACKAGE FOR SINGLE-PIXEL IMAGING BASED ON DEEP LEARNING
"""
#%%
import numpy as np
import torch
import torchvision
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import spyrit.core.meas as meas
import spyrit.core.noise as noise
import spyrit.core.prep as prep
import spyrit.core.torch as spytorch
import spyrit.misc.statistics as stats
import spyrit.misc.sampling as samp
from spyrit.misc.load_data import download_girder

def imagesc_mod(img,
               title='',
               figsize=(5, 5),
               colormap=plt.cm.gray,
               title_fontsize=16,
               dpi=100,
               minmax=None,
               showscale=False,
               **kwargs):
    """
    Plot images with a custom colormap and a custom color for 'nan' values.
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    # clean the axes
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    # set min max for the colormap
    if minmax is None:
        minmax = (img[~img.isnan()].min(), img[~img.isnan()].max())
    # define the color for 'nan' values
    colormap.set_bad(color='grey')
    plt.imshow(img, cmap=colormap, vmin=minmax[0], vmax=minmax[1], **kwargs)
    plt.title(title, fontsize=title_fontsize)
    if showscale:
        # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        plt.colorbar(orientation="vertical")
    plt.show()
    
def split_meas2img(measurements, meas_operator):
    r"""
    Generates a 2D image from split measurements acquired from a LinearSplit or
    HadamSplit operator.
    
    /!\ The measurements must be in the alternating positive / negative format.
    
    Using spyrit 2.3.2
    """
    M = meas_operator.M
    N = meas_operator.N
    h,w = meas_operator.meas_shape
    # using 'nan' so that we can show them with a custom color (see imagesc_mod)
    img_pos = torch.full((N,), torch.tensor(float('nan'))) # even rows
    img_neg = torch.full((N,), torch.tensor(float('nan'))) # odd rows
    
    # split the measurements in pos/neg, then apply meas2img to each
    meas = measurements.view(2*M)
    meas_pos = meas[0::2]
    meas_neg = meas[1::2]
    
    # fill img_pos and img_neg with the measurements
    img_pos[meas_operator.indices[:M]] = meas_pos
    img_neg[meas_operator.indices[:M]] = meas_neg
    
    # concatenate and reshape the images
    img = torch.cat((img_pos.reshape(h,w), img_neg.reshape(h,w)), dim=0)
    
    return img

def center_measurements(measurements):
    r"""
    Centers the measurements so that the max value is the opposite of the min
    value. This is useful for visualization purposes.
    """
    max_val = measurements[~measurements.isnan()].max()
    min_val = measurements[~measurements.isnan()].min()
    return measurements - (max_val + min_val) / 2


#%% PARAMS
# Generic parameters for the script
# -----------------------------------------------------------------------------

# Image size
h = 64
# matrix size for display
h_disp = 16
# number of measurements
M = h**2
# subsampling factor, when used
sub = 4
# noise parameter (poisson distribution)
alpha = 100

# figure size (in inches)
figsize = (5, 5)
figsize_dbl = (5, 10)
# dots per inch (resolution)
dpi = 300
# measurement domain colormap
cmap_meas = plt.cm.cividis

# set the seed for reproducibility
seed = 404


#%% GET IMAGE
# Get image

# download image from girder server
url = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
dataID = "668e986a7d138728d4806d7a"
local_folder = "./image_sample/"
class_folder = "test/"
data_name = "ILSVRC2012_test_00000002.jpeg"
image_abs_path = download_girder(url, dataID, local_folder+class_folder, data_name)

# Create a transform for natural images to normalized grayscale image tensors
transform = stats.transform_gray_norm(img_size=h)
# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=local_folder, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=7)

x, _ = next(iter(dataloader))
x = x[0, :, :, :]
print(f"Shape of input image: {x.shape}")

# Select image
x = x.detach().clone()
c, height, width = x.shape

# Plot Ground-truth image
x_plot = x.view(h, h).cpu().numpy()
orig_minmax = (-1, 1)
imagesc_mod(x_plot, figsize=figsize, dpi=dpi, minmax=orig_minmax)


# %% GET COV MATRIX
# download the covariance matrix and get the file path
url = "https://pilot-warehouse.creatis.insa-lyon.fr/api/v1"
dataId_64 = "63935b624d15dd536f0484a5" 
data_folder = "./stat/"
file_abs_path = download_girder(url, dataId_64, data_folder)

# load it into variables
cov = np.load(file_abs_path)
var = torch.from_numpy(stats.Cov2Var(cov))
cov = torch.from_numpy(cov)



#%% MEASUREMENT OPERATORS
# define 3 measurement operators :
# 1. full (meas.Linear)
# 2. Split, poisson noise
# 3. Split, no noise, variance subsampling
H = spytorch.walsh2_matrix(h)
meas_op1 = meas.Linear(H)
meas_op2 = meas.HadamSplit(M, h)
meas_op3 = meas.HadamSplit(M//sub, h, Ord=var)

# plot the mask matrices
mask1 = (meas_op1.indices.argsort() < meas_op1.M).reshape(h, h)
mask2 = (meas_op2.indices.argsort() < meas_op2.M).reshape(h, h)
mask3 = (meas_op3.indices.argsort() < meas_op3.M).reshape(h, h)

# because op2 & op3 are split, we double the masks
mask2 = torch.cat((mask2, mask2), dim=0)
mask3 = torch.cat((mask3, mask3), dim=0)

imagesc_mod(mask1.cpu().numpy(), figsize=figsize, dpi=dpi, minmax=(-1, 1))
imagesc_mod(mask2.cpu().numpy(), figsize=figsize_dbl, dpi=dpi, minmax=(-1, 1))
imagesc_mod(mask3.cpu().numpy(), figsize=figsize_dbl, dpi=dpi, minmax=(-1, 1))



# %% MATRICES H
# show 1D walsh-ordered hadamard matrices for each case
H_disp1 = spytorch.walsh_matrix(h_disp)
H_disp2 = torch.cat((H_disp1, -H_disp1), dim=0)
H_disp2[H_disp2 == -1] = 0 # take the positive part

order = [0, 1, 7, 6] # custom order for the subsampling
zer = torch.zeros((h_disp-len(order), h_disp))
H_disp3 = torch.cat((H_disp1[order], zer, -H_disp1[order], zer), dim=0)
H_disp3[H_disp3 == -1] = 0 # take the positive part

imagesc_mod(H_disp1, figsize=figsize, dpi=dpi, minmax=(-1, 1))
imagesc_mod(H_disp2, figsize=figsize_dbl, dpi=dpi, minmax=(-1, 1))
imagesc_mod(H_disp3, figsize=figsize_dbl, dpi=dpi, minmax=(-1, 1))



# %% MEASUREMENTS
# set seed for reproducibility in Poisson noise
torch.manual_seed(seed)

# corrupt using poisson noise
noise_op1 = noise.NoNoise(meas_op1)
noise_op2 = noise.Poisson(meas_op2, alpha)
noise_op3 = noise.Poisson(meas_op3, alpha)

# vectorize image for measurements
x = x.view(-1, h*h)
# measure
y1 = noise_op1(x)
y2 = noise_op2(x)
y3 = noise_op3(x)

# plots
y1_plot = y1.reshape(h, h)
y2_plot = split_meas2img(y2, meas_op2)
y3_plot = split_meas2img(y3, meas_op3)

# for split measurements, center measurements for better visualization
y2_plot = center_measurements(y2_plot)
y3_plot = center_measurements(y3_plot)

norm = 'symlog'
imagesc_mod(y1_plot, figsize=figsize, dpi=dpi, norm=norm, colormap=cmap_meas)
imagesc_mod(y2_plot, figsize=figsize_dbl, dpi=dpi, norm=norm, colormap=cmap_meas)
imagesc_mod(y3_plot, figsize=figsize_dbl, dpi=dpi, norm=norm, colormap=cmap_meas)



# %% PREPROCESSING
# preprocess measurements
prep_op1 = prep.DirectPoisson(1, meas_op1)
prep_op2 = prep.SplitPoisson(alpha, meas_op2)
prep_op3 = prep.SplitPoisson(alpha, meas_op3)

# Generate measurements using the measurement operators
m1 = prep_op1(y1)
m2 = prep_op2(y2)
m3 = prep_op3(y3)

# because we have defined a different sampling order, we call meas2img to
# show the measurements in the same order as the mask
# show the mask as nan for custom color
m2_plot = torch.from_numpy(samp.meas2img(m2, meas_op2.Ord.numpy()))
m2_plot[m2_plot == 0] = torch.tensor(float('nan'))

m3_plot = torch.from_numpy(samp.meas2img(m3, meas_op3.Ord.numpy()))
m3_plot[m3_plot == 0] = torch.tensor(float('nan'))

# plot the preprocessed measurements
imagesc_mod(m1.view(h, h).cpu(), figsize=figsize, dpi=dpi, norm=norm, colormap=cmap_meas)
imagesc_mod(m2_plot.reshape(h, h), figsize=figsize, dpi=dpi, norm=norm, colormap=cmap_meas)
imagesc_mod(m3_plot.reshape(h, h), figsize=figsize, dpi=dpi, norm=norm, colormap=cmap_meas)



# %% RECONSTRUCTION
# Reconstruct the image using the measurements
z1 = meas_op1.pinv(m1)
z2 = meas_op2.pinv(m2)
z3 = meas_op3.pinv(m3)



#%%
# plot the reconstructions
imagesc_mod(z1.view(h, h).cpu().numpy(), figsize=figsize, dpi=dpi, minmax=orig_minmax)
imagesc_mod(z2.view(h, h).cpu().numpy(), figsize=figsize, dpi=dpi, minmax=orig_minmax)
imagesc_mod(z3.view(h, h).cpu().numpy(), figsize=figsize, dpi=dpi, minmax=orig_minmax)
