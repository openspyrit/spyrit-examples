# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:25:43 2022

@author: ducros
"""

from __future__ import print_function, division
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.profiler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import argparse
from pathlib import Path
import pickle
import os
import datetime

from spyrit.core.noise import PoissonApproxGauss
from spyrit.core.meas import HadamSplit
from spyrit.core.prep import SplitPoisson
from spyrit.core.recon import DCNet, PinvNet, LearnedPGD
from spyrit.core.train import train_model, Train_par, save_net, Weight_Decay_Loss
from spyrit.core.nnet import Unet, ConvNet, ConvNetBN
from spyrit.misc.statistics import Cov2Var, data_loaders_ImageNet, data_loaders_stl10, data_loaders_img_folder

from spyrit.core.noise import NoNoise
from spyrit.core.prep import DirectPoisson
from spyrit.core.noise import Poisson

# pip install -e git+https://github.com/openspyrit/spas.git@v1.4#egg=spas
# python3 ./spyrit-examples/2022_OE_spyrit2/download_data.py

def get_meas_operator(opt, Ord):
    """
    Define the most used measurement definitions
    """
    if opt.meas == 'hadam-split':
        # Hadamard split operator given Ord
        meas_op = HadamSplit(opt.M, opt.img_size, Ord)
    elif opt.meas == 'hadam-pos':
        # Hadamard positive operator with cartesian subsampling
        from spyrit.misc.walsh_hadamard import walsh2_matrix
        from spyrit.misc.sampling import Permutation_Matrix
        from spyrit.core.meas import Linear

        F = walsh2_matrix(opt.img_size)
        F = np.where(F>0, F, 0)

        Sampling_map = np.ones((opt.img_size,opt.img_size))
        M_xy = math.ceil(opt.M**0.5)
        Sampling_map[:,M_xy:] = 0
        Sampling_map[M_xy:,:] = 0

        Perm = Permutation_Matrix(Sampling_map)
        F = Perm@F 
        H = F[:opt.M,:]

        meas_op = Linear(H, pinv=True)  
    return meas_op

def get_noise_operator(opt, meas_op):
    """
    Define the most used noise definitions
    """ 
    if opt.N0 == 1:  # if opt.noise == 'no_noise':
        # Noiseless case
        noise_op = NoNoise(meas_op)        
    else:
        # Poisson noise
        if opt.noise == 'gauss-approx':
            noise_op = PoissonApproxGauss(meas_op, opt.N0) # faster than Poisson
        elif opt.noise == 'poisson':
            noise_op = Poisson(meas_op, opt.N0)        
    return noise_op

def get_prep_operator(opt, meas_op):
    """
    Define the most used preprocessing definitions
    """
    if opt.prep == 'split-poisson':
        prep_op = SplitPoisson(opt.N0, meas_op) 
    elif opt.prep == 'dir-poisson':
        from spyrit.core.prep import DirectPoisson
        prep_op = DirectPoisson(opt.N0, meas_op)   # "Undo" the NoNoise operator
    return prep_op

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Forward model
    parser.add_argument("--meas",       type=str, default="hadam-pos",   help="Measurement type: 'hadam-split', 'hadam-pos'")
    parser.add_argument("--noise",      type=str, default="poisson",   help="Noise types: 'poisson', 'gauss-approx', 'no-noise'")
    parser.add_argument("--prep",       type=str, default="dir-poisson",   help="Preprocessing types: 'dir-poisson', 'split-poisson'")

    # Acquisition
    parser.add_argument("--img_size",   type=int,   default=64,   help="Height / width dimension")
    parser.add_argument("--M",          type=int,   default=512,  help="Number of patterns")
    parser.add_argument("--subs",       type=str,   default="rect",  help="Among 'var','rect'")

    # Network and training
    parser.add_argument("--data",       type=str,   default="stl10", help="stl10, imagenet, folder")
    parser.add_argument("--model_root", type=str,   default='./model/', help="Path to model saving files")
    parser.add_argument("--data_root",  type=str,   default="./data/", help="Path to the dataset")
    
    parser.add_argument("--N0",         type=float, default=10,   help="Mean maximum total number of photons")
    parser.add_argument("--stat_root",  type=str,   default="", help="Path to precomputed data covariance and mean")
    parser.add_argument("--arch",       type=str,   default="dc-net", help="Choose among 'dc-net','pinv-net', 'upgd")
    parser.add_argument("--denoi",      type=str,   default="unet", help="Choose among 'cnn','cnnbn', 'unet'")
    parser.add_argument("--device",     type=str,   default="", help="Choose among 'cuda','cpu'")
    #parser.add_argument("--no_denoi",   default=False, action='store_true', help="No denoising layer")

    # Specific models parameters
    parser.add_argument("--x0",         type=int,   default=0,    help="Initial estimate. 0: zero, 1: as given in the reconstruction class")
    parser.add_argument("--unfold_iter",   type=int,   default=3,    help="Number of unrolled iterations")
#    parser.add_argument("--unfold_step_size",   type=str, default="custom", help="Step size parameter. Default to custom 1/N")
    parser.add_argument("--unfold_step_grad", type=boolean_string,   default=False, help="Learnable step size")
    parser.add_argument("--unfold_step_decay",   type=float, default=1, help="Step size decay")
    parser.add_argument("--wls",        type=boolean_string,   default=False, help="Weighted least squares")

    # Optimisation
    parser.add_argument("--num_epochs", type=int,   default=30,   help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=512, help="Size of each training batch")
    parser.add_argument("--reg",        type=float, default=1e-7, help="Regularisation Parameter")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--step_size",  type=int,   default=10,   help="Scheduler Step Size")
    parser.add_argument("--gamma",      type=float, default=0.5,  help="Scheduler Decrease Rate")
    parser.add_argument("--checkpoint_model", type=str, default="", help="Optional path to checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=0, help="Interval between saving model checkpoints")
    
    # Tensorboard
    parser.add_argument("--tb_path",    type=str,   default=False, help="Relative path for Tensorboard experiment tracking logs")
    parser.add_argument("--tb_prof",    type=str,   default=False, help="Profiler for code with Tensorboard")

    opt = parser.parse_args()
    if os.path.exists(opt.model_root) is False:
        os.makedirs(opt.model_root)
    opt.model_root = Path(opt.model_root)
    opt.data_root = Path(opt.data_root)
    
    # Define other parameters (for testing)
    if False:
        opt.meas = 'hadam-pos'
        opt.noise = 'no-noise' # noise type
        opt.prep = 'dir-poisson'    # preprocessing type
        opt.N0 = 1.0        # ph/pixel m
        opt.M = opt.img_size**2 // 4
        opt.arch = 'pinv-net' # Network architecture
        opt.denoi = 'cnn' 
        name_run = "stdl10_hadampos"
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        opt.tb_path = f'runs/runs_{name_run}_n{int(opt.N0)}_m{opt.M}/{now}'

    print(opt)

    #==========================================================================
    # 0. Setting up parameters for training
    #==========================================================================
    # The device of the machine, number of workers...
    # 
    if opt.device: 
        device = torch.device(opt.device)
    else:
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
                                        shuffle=True, download=True)   

        #now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        #opt.tb_path = f'runs/runs_stdl10_n100_m1024/{now}'

    elif opt.data == 'imagenet':
        dataloaders = data_loaders_ImageNet(opt.data_root / 'test', 
                                        opt.data_root / 'val', 
                                        img_size=opt.img_size, 
                                        batch_size=opt.batch_size, 
                                        seed=7,
                                        shuffle=True)
    elif opt.data == 'folder':
        # Check if there is a train folder within the data_root
        data_root = os.path.join(opt.data_root, 'train')
        if not os.path.isdir(data_root):
            data_root = opt.data_root
        # Check if there is a val folder within the data_root
        data_val_root = os.path.join(opt.data_root, 'val')
        if not os.path.isdir(data_val_root):
            print('There is no val folder within the data_root, splitting data in the train folder!')
            data_val_root = None
        dataloaders = data_loaders_img_folder(data_root=data_root,
                                        data_val_root=data_val_root, 
                                        img_size=opt.img_size, 
                                        batch_size=opt.batch_size, 
                                        shuffle=True)


    #==========================================================================
    # 2. Statistics of the training images
    #==========================================================================
    if opt.stat_root:
        # Load covariance and mean from path
        print('Loading covariance and mean')
        if opt.img_size == 64:
            my_average_file = Path(opt.stat_root) / ('Average_{}x{}'.format(opt.img_size, opt.img_size)+'.npy')
            my_cov_file = Path(opt.stat_root) / ('Cov_{}x{}'.format(opt.img_size, opt.img_size)+'.npy')
        else:
            my_average_file = Path(opt.stat_root) / ('Average_8_{}x{}'.format(opt.img_size, opt.img_size)+'.npy')
            my_cov_file = Path(opt.stat_root) / ('Cov_8_{}x{}'.format(opt.img_size, opt.img_size)+'.npy')
            
        Mean = np.load(my_average_file)
        Cov  = np.load(my_cov_file)
    else:
        # Covariance not provided, set it to the identity
        if opt.arch == 'dc-net':
            Cov = np.eye(opt.img_size**2)
            print("Seting Cov matrix to the identity: Not optimal for dc-net!!!")

    #==========================================================================
    # 3. Subsampling
    #==========================================================================
    if opt.subs == 'var':
        print('Subsampling: high variance')
        Ord = Cov2Var(Cov)
    #     
    elif opt.subs == 'rect':
        print('Subsampling: low frequency (rect)')
        import numpy as np
        import math
        Ord = np.ones((opt.img_size,opt.img_size))
        n_sub = math.ceil(opt.M**0.5)
        Ord[:,n_sub:] = 0
        Ord[n_sub:,:] = 0
    else:
        Ord = None

    #==========================================================================
    # 3. Define a Neural Network
    #==========================================================================
    # Measurement, noise and preprocessing operators
    meas_op = get_meas_operator(opt, Ord)
    noise_op = get_noise_operator(opt, meas_op)
    prep_op = get_prep_operator(opt, meas_op)
    
    # Image-domain denoising layer
    if opt.denoi == 'cnn':      # CNN no batch normalization
        denoi = ConvNet()
    elif opt.denoi == 'cnnbn':  # CNN with batch normalization
        denoi = ConvNetBN()
    elif opt.denoi == 'unet':   # Unet
        denoi = Unet()
    elif opt.denoi == 'cnn-diff':   # Diff CNN per iteration
        if opt.x0 == 0:
            unfold_iter = opt.unfold_iter
        else:
            unfold_iter = opt.unfold_iter + 1
        denoi = nn.ModuleList([ConvNet() for _ in range(unfold_iter)])
    
    # Global Architecture
    if opt.arch == 'dc-net':        # Denoised Completion Network
        model = DCNet(noise_op, prep_op, Cov, denoi)
        
    elif opt.arch == 'pinv-net':    # Pseudo Inverse Network
        model = PinvNet(noise_op, prep_op, denoi)

    elif opt.arch == 'lpgd':        # Learned Proximal Gradient Descent
        model = LearnedPGD(noise_op, prep_op, denoi,
                     iter_stop=opt.unfold_iter, step_grad=opt.unfold_step_grad, step_decay=opt.unfold_step_decay,
                     wls=opt.wls)  

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    if opt.checkpoint_model:
        model.load_state_dict(torch.load(opt.checkpoint_model))
        print(f'Loaded model from {opt.checkpoint_model}')

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
            disp=True, do_checkpoint=opt.checkpoint_interval, tb_path=opt.tb_path)
    
    #==========================================================================
    # 6. Saving the model so that it can later be utilized
    #==========================================================================
    #- network's architecture
    train_type = 'N0_{:g}'.format(opt.N0) 
        
    #- training parameters
    suffix = 'm_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
           opt.meas, opt.img_size, opt.M, opt.num_epochs, opt.lr, opt.step_size,\
           opt.gamma, opt.batch_size, opt.reg)
    # suffix for UPGD iterations
    if opt.arch == 'lpgd':
        suffix += '_uit_{}'.format(opt.unfold_iter)
        if opt.unfold_step_grad:
            suffix += '_sgrad'
        if opt.unfold_step_decay != 1:
            suffix += '_sdec{}'.format(opt.unfold_step_decay).replace('.','-')
    if opt.checkpoint_model:
        suffix += '_cont'
    title = opt.model_root / f'{opt.arch}_{opt.denoi}_{opt.data}_{train_type}_{suffix}'    
    print(title)
    
    Path(opt.model_root).mkdir(parents=True, exist_ok=True)
   
    if opt.checkpoint_interval:
       Path(title).mkdir(parents=True, exist_ok=True)
       
    save_net(title, model)
    
    #- save training history
    params = Train_par(opt.batch_size, opt.lr, opt.img_size,reg=opt.reg);
    params.set_loss(train_info);
    train_path = opt.model_root / f'TRAIN_{opt.arch}_{opt.denoi}_{opt.data}_{train_type}_{suffix}.pkl'
    
    with open(train_path, 'wb') as param_file:
        pickle.dump(params,param_file)
    torch.cuda.empty_cache()

