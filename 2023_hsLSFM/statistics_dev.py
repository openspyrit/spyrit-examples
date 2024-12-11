# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 10:08:04 2023

@author: ducros
"""
#%%
import torch
import torchvision
import spyrit.misc.walsh_hadamard as wh
from pathlib import Path
import numpy as np

def data_loaders_ImageNet(train_root, val_root=None, img_size=64, 
                          batch_size=512, seed=7, shuffle=False): 
    """ 
    Args:
        Both 'train_root' and 'val_root' need to have images in a subfolder
        shuffle=True to shuffle train set only (test set not shuffled)
        
    The output of torchvision datasets are PILImage images in the range [0, 1].
    We transform them to Tensors in the range [-1, 1]. Also RGB images are 
    converted into grayscale images.   
    """

    torch.manual_seed(seed) # reproductibility of random crop
    #    
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.functional.to_grayscale,
         torchvision.transforms.Resize(img_size),
         torchvision.transforms.CenterCrop(img_size),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize([0.5], [0.5])
        ])
    
    # train set
    trainset = torchvision.datasets.ImageFolder(root=train_root, transform=transform)
    trainloader =  torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
    
    # validation set (if any)
    if val_root is not None:
        valset = torchvision.datasets.ImageFolder(root=val_root, transform=transform)
        valloader =  torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    else:
        valloader = None
        
    dataloaders = {'train':trainloader, 'val':valloader}
    
    return dataloaders

def mean_walsh1(dataloader, device, n_loop=1):
    """ 
    nloop > 1 is relevant for dataloaders with random crops such as that 
    provided by data_loaders_ImageNet
        
    """
    
    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (b, _, nx, ny) = inputs.shape
    tot_num = len(dataloader)*b
    
    # Init
    n = 0
    H = wh.walsh_matrix(ny).astype(np.float32, copy=False)
    mean = torch.zeros(ny, dtype=torch.float32)
    
    # Send to device (e.g., cuda)
    mean = mean.to(device)
    H = torch.from_numpy(H).to(device)
    
    # Compute Mean 
    # Accumulate sum over all the image columns in the database
    for i in range(n_loop):
        torch.manual_seed(i)
        for inputs,_ in dataloader:
            inputs = inputs.to(device)
            inputs = inputs.view(-1,nx,ny)
            #
            trans = wh.walsh_torch(inputs,H)
            mean = mean.add(trans.sum((0,1)))  # Accumulate over images and rows
            # print
            n = n + inputs.shape[0]
            print(f'Mean:  {n} / (less than) {tot_num*n_loop} images', end='\n')
        print('', end='\n')
        
    # Normalize
    mean = mean/n/nx
    mean = torch.squeeze(mean)
    
    return mean 

def cov_walsh1(dataloader, mean, device, n_loop=1):
    """ 
    nloop > 1 is relevant for dataloaders with random crops such as that 
    provided by data_loaders_ImageNet
        
    """
    
    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (b, _, nx, ny) = inputs.shape
    tot_num = len(dataloader)*b
    
    H = wh.walsh_matrix(ny).astype(np.float32, copy=False)
    H = torch.from_numpy(H).to(device)
    
    # Covariance --------------------------------------------------------------
    # Init
    n = 0
    cov = torch.zeros((ny,ny), dtype=torch.float32)
    cov = cov.to(device)
    
    # Accumulate (im - mu)^T*(im - mu) over all images in dataset.
    # Each row is assumed to be an observation, so we have to transpose
    for i in range(n_loop):
        torch.manual_seed(i)
        for inputs,_ in dataloader:
            inputs = inputs.to(device)
            (b, c, _, _) = inputs.shape
            inputs = inputs.view(-1,nx,ny) # shape (b*c, nx, ny)
            #
            trans = wh.walsh_torch(inputs,H) # shape (b*c, nx, ny)
            trans = (trans - mean).mT
            #
            cov = torch.addbmm(cov, trans, trans.mT)
            # print
            n += inputs.shape[0]
            print(f'Cov:  {n} / (less than) {tot_num*n_loop} images', end='\n')
        print('', end='\n')
    
    # Normalize
    cov = cov/(n-1)/(nx-1)
    
    return cov

def stat_walsh1(dataloader, device:torch.device, root, n_loop:int=1):
    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (_, _, nx, ny) = inputs.shape

    #--------------------------------------------------------------------------
    # 1. Mean
    #--------------------------------------------------------------------------
    mean = mean_walsh1(dataloader, device, n_loop=n_loop)
    
    # Save
    if n_loop==1:
        path = root / Path('Average_walsh1_{}x{}'.format(nx,ny)+'.npy')
    else:
        path = root / Path('Average_walsh1_{}_{}x{}'.format(n_loop,nx,ny)+'.npy')
        
    if not root.exists():
        root.mkdir()
    np.save(path, mean.cpu().detach().numpy())
    #--------------------------------------------------------------------------
    # 2. Covariance
    #-------------------------------------------------------------------------
    cov = cov_walsh1(dataloader, mean, device, n_loop=n_loop)
        
    # Save
    if n_loop==1:
        path = root / Path('Cov_walsh1_{}x{}'.format(nx,ny)+'.npy')
    else:
        path = root / Path('Cov_walsh1_{}_{}x{}'.format(n_loop,nx,ny)+'.npy')
        
    if not root.exists():
        root.mkdir()
    np.save(path, cov.cpu().detach().numpy())
    
    return mean, cov