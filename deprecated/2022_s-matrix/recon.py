# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 19:02:22 2022

@author: ducros
"""
#%%
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
import spyrit.misc.walsh_hadamard as wh

#%% 
def add_colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def noaxis(axs):
    for ax in axs:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

#%%
#- Acquisition
img_size = 64 # image size
batch_size = 512

#- Model and data paths
data_root = Path('./data/')

#- Save plot using type 1 font
plt.rcParams['pdf.fonttype'] = 42

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)

transform = torchvision.transforms.Compose(
    [torchvision.transforms.functional.to_grayscale,
     torchvision.transforms.Resize((img_size, img_size)),
     torchvision.transforms.ToTensor(),
     #])
     torchvision.transforms.Normalize([0.5], [0.5])])

trainset = \
    torchvision.datasets.STL10(root=data_root, split='train+unlabeled',download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)

testset = \
    torchvision.datasets.STL10(root=data_root, split='test',download=True, transform=transform)
testloader =  torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)

dataloaders = {'train':trainloader, 'val':testloader}

#%% Walsh ordered 2D on STL-10
inputs, classes = next(iter(dataloaders['val']))
img = inputs[89, 0, :, :];
img = img.cpu().detach().numpy()
#img = (img - img.min())/(img.max()-img.min())

img_size = 64
had = wh.walsh2(img)
im1 = wh.iwalsh2(had)
err = img - im1
print(np.linalg.norm(err)/np.linalg.norm(img))

f, axs = plt.subplots(1, 4, figsize=(12,8),  dpi= 100)
im = axs[0].imshow(img, cmap='gray')
add_colorbar(im)
im = axs[1].imshow(had, cmap='gray')
add_colorbar(im)
im = axs[2].imshow(im1, cmap='gray')
add_colorbar(im)
im = axs[3].imshow(err, cmap='gray')
add_colorbar(im)
axs[1].set_title("Walsh transform")
axs[2].set_title("inverse transform")
axs[3].set_title("difference")
noaxis(axs)

#%%
f, ax = plt.subplots(1, 2, figsize=(12,8),  dpi= 100)
ax[0].hist(img.ravel(), bins = 50)
ax[1].hist(had.ravel()[1:], bins = 200, log = True)

#%% S-Walsh ordered "2D" on STL-10
mes = wh.fwalsh2_S(img)
im2 = wh.ifwalsh2_S(mes)
er2 = img - im2
print(np.linalg.norm(er2/np.linalg.norm(img)))

mes[0,0] = np.nan
er2[0,0] = np.nan

f, axs = plt.subplots(1, 4, figsize=(12,8),  dpi= 100)
im = axs[0].imshow(img, cmap='gray') 
add_colorbar(im)
im = axs[1].imshow(mes, cmap='gray')
add_colorbar(im)
im = axs[2].imshow(im2, cmap='gray')
add_colorbar(im)
im = axs[3].imshow(er2, cmap='gray')
add_colorbar(im)
noaxis(axs)
axs[1].set_title("S-Walsh transform")
axs[2].set_title("inverse transform")
axs[3].set_title("difference")

#%%
f, ax = plt.subplots(1, 2, figsize=(12,8),  dpi= 100)
ax[0].hist(img.ravel(), bins = 50)
ax[1].hist(mes.ravel()[1:],bins = 200, log = True)