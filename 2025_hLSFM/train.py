# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:38:18 2023

@author: ducros
"""
from __future__ import print_function, division
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
import argparse
from pathlib import Path
import pickle

from spyrit.core.noise import PoissonApproxGauss, Poisson
from spyrit.core.meas import LinearSplit
#from spyrit.core.prep import SplitPoisson
from spyrit.core.prep import SplitPoissonRaw

from spyrit.core.train import train_model, Train_par, save_net, Weight_Decay_Loss
from spyrit.core.nnet import Unet, ConvNet, ConvNetBN
from spyrit.misc.walsh_hadamard import walsh_matrix

from meas_dev import Hadam1Split
#from recon_dev import DC1Net, Pinv1Net, Tikho1Net
from spyrit.core.recon import Pinv1Net, TikhoNet 
from statistics_dev import data_loaders_ImageNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Acquisition
    parser.add_argument("--pattern_root", type=str, default=None, 
                        help="Path to measurement patterns or None")
    parser.add_argument("--img_size",   type=int,   default=512,  help="Height / width dimension")
    parser.add_argument("--M",          type=int,   default=128,  help="Number of patterns")
    
    # Network and training
    parser.add_argument("--data_root",  type=str,   default="./data/ILSVRC2012_v10102019", help="Path to the training dataset")
    parser.add_argument("--data",       type=str,   default="imagenet", help="imagenet")
    parser.add_argument("--stat_root",  type=str,   default="./stat/", help="Path to precomputed data")
    parser.add_argument("--arch",       type=str,   default="pinv-net", help="Choose among 'pinv-net'")
    parser.add_argument("--model_root", type=str,   default='./model/', help="Path to model saving files")
    
    parser.add_argument("--alpha",      type=float, default=10,   help="Mean maximum total number of photons")
    parser.add_argument("--denoi",      type=str,   default="unet", help="Choose among 'cnn','cnnbn', 'unet'")

    # Optimisation
    parser.add_argument("--num_epochs", type=int,   default=1,   help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=64, help="Size of each training batch")
    parser.add_argument("--reg",        type=float, default=1e-7, help="Regularisation Parameter")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--step_size",  type=int,   default=10,   help="Scheduler Step Size")
    parser.add_argument("--gamma",      type=float, default=0.5,  help="Scheduler Decrease Rate")
    parser.add_argument("--checkpoint_model", type=str, default="", help="Optional path to checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=0, help="Interval between saving model checkpoints"
    )
    opt = parser.parse_args()
    if opt.pattern_root == 'None':
        opt.pattern_root = None
    if opt.pattern_root is not None:
        opt.pattern_root = Path(opt.pattern_root)
    opt.model_root = Path(opt.model_root)
    opt.data_root = Path(opt.data_root)
    
    print(opt)
    
    #==========================================================================
    # 0. Setting up parameters for training
    #==========================================================================
    # The device of the machine, number of workers...
    # 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    
    #==========================================================================
    # 1. Loading patterns or readind size
    #==========================================================================
    if opt.pattern_root is not None:
        H =  np.load(opt.pattern_root)
        H /= H[0,16:500].mean()
        M, img_size = H.shape
        print(f'Patterns: max={H.max()}, min={H.min()}, mean={H.mean()}')
    else:
        M = opt.M 
        img_size = opt.img_size
    
    print(f'{M} measurements, {img_size} columns per image')
    
    dataloaders = data_loaders_ImageNet(opt.data_root / 'test', 
                                    opt.data_root / 'val', 
                                    img_size=img_size, 
                                    batch_size=opt.batch_size, 
                                    seed=7,
                                    shuffle=True)
    #==========================================================================
    # 2. Statistics of the training images
    #==========================================================================
    if opt.arch == 'dc-net' or opt.arch == 'tikho-net':
        print('Loading covariance')#' and mean')
        #my_average_file = Path(opt.stat_root) / f'Average_1_{img_size}x{img_size}.npy'
        my_cov_file = Path(opt.stat_root) / f'Cov_1_{img_size}x{img_size}.npy'
            
        #Mean = np.load(my_average_file)
        sigma  = np.load(my_cov_file)
    #==========================================================================
    # 3. Define a Neural Network
    #==========================================================================
    if opt.pattern_root is None or opt.arch == 'dc-net':
        meas = Hadam1Split(M, img_size)
        patt_type = 'Hadam1'
    else:
        #meas = LinearSplit(H, pinv=True)
        meas = LinearSplit(torch.from_numpy(H), pinv=True, meas_shape=(1,img_size)) 
        patt_type = 'exp'
    
    #prep = SplitPoisson(opt.alpha, meas)
    prep  = SplitPoissonRaw(opt.alpha, meas)
    #noise = Poisson(meas, opt.alpha)
    noise = PoissonApproxGauss(meas, opt.alpha) # faster than Poisson
       
    # Image-domain denoising layer
    if opt.denoi == 'cnn':      # CNN no batch normalization
        denoi = ConvNet()
    elif opt.denoi == 'cnnbn':  # CNN with batch normalization
        denoi = ConvNetBN()
    elif opt.denoi == 'unet':   # Unet
        denoi = Unet()
    
    # -- Global Architecture
    # a. Pseudo Inverse network
    if opt.arch == 'pinv-net':   
        model = Pinv1Net(noise, prep, denoi)
    # b. Denoised completion network    
    elif opt.arch == 'dc-net':
        H = walsh_matrix(img_size)
        model = DC1Net(noise, prep, H @ sigma @ H, denoi)
    # c. Tikhonov network
    elif opt.arch == 'tikho-net':
        #model = Tikho1Net(noise, prep, sigma, denoi)
        model = TikhoNet(noise, prep, torch.from_numpy(sigma), denoi) 
            
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
    train_type = 'ph_{:g}'.format(opt.alpha) 
        
    #- training parameters
    suffix = '{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
           patt_type, img_size, M, opt.num_epochs, opt.lr, opt.step_size,\
           opt.gamma, opt.batch_size, opt.reg)

    title = opt.model_root / f'{opt.arch}_{opt.denoi}_{opt.data}_{train_type}_{suffix}'    
    print(title)
    
    Path(opt.model_root).mkdir(parents=True, exist_ok=True)
   
    if opt.checkpoint_interval:
       Path(title).mkdir(parents=True, exist_ok=True)
       
    save_net(title, model)
    
    #- save training history
    params = Train_par(opt.batch_size, opt.lr, img_size,reg=opt.reg);
    params.set_loss(train_info);
    train_path = opt.model_root / f'TRAIN_{opt.arch}_{opt.denoi}_{opt.data}_{train_type}_{suffix}.pkl'
    
    with open(train_path, 'wb') as param_file:
        pickle.dump(params,param_file)
    torch.cuda.empty_cache()