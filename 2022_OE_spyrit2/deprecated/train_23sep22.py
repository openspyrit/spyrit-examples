# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:13:34 2023

@author: ducros

git checkout towards_v2
git checkout -b tmp-23sep22 49ea3d9a3793686ebb0ec8c41ed414b286198c26
"""
from __future__ import print_function, division
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import argparse
from pathlib import Path
import pickle


from spyrit.learning.model_Had_DCAN import Permutation_Matrix, Weight_Decay_Loss
from spyrit.learning.nets import train_model, Train_par, save_net
from spyrit.restructured.Updated_Had_Dcan import *
from spyrit.misc.statistics import Cov2Var, data_loaders_ImageNet, data_loaders_stl10
import spyrit.misc.walsh_hadamard as wh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Acquisition
    parser.add_argument("--img_size",   type=int,   default=64,   help="Height / width dimension")
    parser.add_argument("--M",          type=int,   default=512,  help="Number of patterns")
    parser.add_argument("--subs",       type=str,   default="var",  help="Among 'var','rect','rand'")
    
    # Network and training
    parser.add_argument("--data",       type=str,   default="imagenet", help="stl10 or imagenet")
    parser.add_argument("--model_root", type=str,   default='./model/', help="Path to model saving files")
    parser.add_argument("--data_root",  type=str,   default="./data/ILSVRC2012_v10102019", help="Path to the dataset")
    
    parser.add_argument("--N0",         type=float, default=10,   help="Mean maximum total number of photons")
    parser.add_argument("--stat_root",  type=str,   default="stats", help="Path to precomputed data")
    parser.add_argument("--arch",       type=str,   default="dc-net", help="Choose among 'dc-net','pinv-net',")
    parser.add_argument("--denoi",      type=str,   default="cnn", help="Choose among 'cnn','cnnbn', 'unet'")
    parser.add_argument("--no_denoi",   default=False, action='store_true', help="No denoising layer")


    # Optimisation
    parser.add_argument("--num_epochs", type=int,   default=30,   help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=1024, help="Size of each training batch")
    parser.add_argument("--reg",        type=float, default=1e-7, help="Regularisation Parameter")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--step_size",  type=int,   default=10,   help="Scheduler Step Size")
    parser.add_argument("--gamma",      type=float, default=0.5,  help="Scheduler Decrease Rate")
    parser.add_argument("--checkpoint_model", type=str, default="", help="Optional path to checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=0, help="Interval between saving model checkpoints"
    )
    opt = parser.parse_args()
    opt.model_root = Path(opt.model_root)
    opt.data_root = Path(opt.data_root)
    
    if opt.data == 'stl10':
        opt.data_root = './data/'
    
    print(opt)
    
    
    
    #==========================================================================
    # 0. Setting up parameters for training
    #==========================================================================
    # The device of the machine, number of workers...
    # 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    
    #==========================================================================
    # 1. Loading and normalizing data
    #==========================================================================
    if opt.data == 'stl10':
        dataloaders = data_loaders_stl10(opt.data_root, 
                                        img_size=opt.img_size, 
                                        batch_size=opt.batch_size, 
                                        seed=7,
                                        shuffle=True)        
    elif opt.data == 'imagenet':
        dataloaders = data_loaders_ImageNet(opt.data_root / 'test', 
                                        opt.data_root / 'val', 
                                        img_size=opt.img_size, 
                                        batch_size=opt.batch_size, 
                                        seed=7,
                                        shuffle=True)
   
    #==========================================================================
    # 2. Statistics of the training images
    #==========================================================================
    print('Loading covariance and mean')
    if opt.img_size == 64:
        my_average_file = Path(opt.stat_root) / ('Average_{}x{}'.format(opt.img_size, opt.img_size)+'.npy')
        my_cov_file = Path(opt.stat_root) / ('Cov_{}x{}'.format(opt.img_size, opt.img_size)+'.npy')
    else:
        my_average_file = Path(opt.stat_root) / ('Average_8_{}x{}'.format(opt.img_size, opt.img_size)+'.npy')
        my_cov_file = Path(opt.stat_root) / ('Cov_8_{}x{}'.format(opt.img_size, opt.img_size)+'.npy')
        
    Mean = np.load(my_average_file)
    Cov  = np.load(my_cov_file)

    #==========================================================================
    # 3. Subsampling
    #==========================================================================
    if opt.subs == 'var':
        print('Subsampling: high variance')
        Ord = Cov2Var(Cov)
    # /!\ not tested    
    elif opt.subs == 'rect':
        print('Subsampling: low frequency (rect)')
        import numpy as np
        import math
        Ord = np.ones((opt.img_size,opt.img_size))
        n_sub = math.ceil(opt.M**0.5)
        Ord[:,n_sub:] = 0
        Ord[n_sub:,:] = 0
    # /!\ not tested    
    elif opt.subs == 'rand':
        print('Subsampling: random')
        import numpy as np
        np.random.seed(0)
        Ord = np.random.rand(opt.img_size,opt.img_size)
        
    Perm = Permutation_Matrix(Ord)
    H =  wh.walsh2_matrix(opt.img_size)
    Hperm = Perm @ H
    Pmat = Hperm[:opt.M,:]

    #==========================================================================
    # 3. Define a Neural Network
    #==========================================================================
    FO = Split_Forward_operator_ft_had(Pmat, Perm, opt.img_size, opt.img_size)
    Noi = Bruit_Poisson_approx_Gauss(opt.N0, FO)
    Prep = Split_diag_poisson_preprocess(opt.N0, opt.M, opt.img_size**2)
    
    # Image-domain denoising layer
    if opt.denoi == 'cnn':      # CNN no batch normalization
        Denoi = ConvNet()
    elif opt.denoi == 'cnnbn':  # CNN with batch normalization
        Denoi = ConvNetBN()
    elif opt.denoi == 'unet':   # Unet
        Denoi = Unet()
    
    # Global Architecture
    if opt.arch == 'dc-net':        # Denoised Completion Network
        Cov_perm = Perm @ Cov @ Perm.T
        DC = Generalized_Orthogonal_Tikhonov(sigma_prior = Cov_perm, 
                                             M = opt.M, 
                                             N = opt.img_size**2)
        model = DC2_Net(Noi, Prep, DC, Denoi)
        
    elif opt.arch == 'pinv-net':    # Pseudo Inverse Network
        DC = Pinv_orthogonal()
        model = Pinv_Net(Noi, Prep, DC, Denoi)
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    if opt.checkpoint_model:
        model.load_state_dict(torch.load(opt.checkpoint_model))
    
    #==========================================================================
    # 4. Define a Loss function optimizer and scheduler
    #==========================================================================
    # Penalization defined in DCAN.py
    loss = nn.MSELoss();
    criterion = Weight_Decay_Loss(loss);
    optimizer = optim.Adam(model.parameters(), lr=opt.lr);
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)
    
    #==========================================================================
    # 5. Train the network
    #==========================================================================
    # We  loop over our data iterator, feed the inputs to the
    model, train_info = train_model(model, criterion, \
            optimizer, scheduler, dataloaders, device, opt.model_root, num_epochs=opt.num_epochs,\
            disp=True, do_checkpoint=opt.checkpoint_interval)
    
    #==========================================================================
    # 6. Saving the model so that it can later be utilized
    #==========================================================================
    #- network's architecture
    train_type = 'N0_{:g}'.format(opt.N0) 
        
    #- training parameters
    suffix = 'N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
           opt.img_size, opt.M, opt.num_epochs, opt.lr, opt.step_size,\
           opt.gamma, opt.batch_size, opt.reg)

    title = opt.model_root / f'{opt.arch}2_{opt.denoi}_{opt.data}_{train_type}_{suffix}'    
    print(title)
    
    Path(opt.model_root).mkdir(parents=True, exist_ok=True)
   
    if opt.checkpoint_interval:
       Path(title).mkdir(parents=True, exist_ok=True)
       
    save_net(title, model)
    
    #- save training history
    params = Train_par(opt.batch_size, opt.lr, opt.img_size,reg=opt.reg);
    params.set_loss(train_info);
    train_path = opt.model_root / f'TRAIN_{opt.arch}2_{opt.denoi}_{opt.data}_{train_type}_{suffix}.pkl'
    
    with open(train_path, 'wb') as param_file:
        pickle.dump(params,param_file)
    torch.cuda.empty_cache()