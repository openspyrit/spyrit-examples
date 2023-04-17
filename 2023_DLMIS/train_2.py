# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 11:54:05 2021

@author: ducros
"""
# In a terminal
# python train.py --angle_nb 40

# In spyder :
# runfile('train.py', args='--angle_nb 40')
#%%
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import transforms
import argparse
from pathlib import Path

from model_radon import Linear, radonSpecifyAngles
import pickle
from spyrit.core.prep import DirectPoisson
from spyrit.core.recon import PinvNet
from spyrit.core.noise import NoNoise
from spyrit.core.train import train_model, Weight_Decay_Loss, save_net, Train_par
from spyrit.core.nnet import ConvNet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Acquisition
    parser.add_argument("--img_size",   type=int,   default=64,    help="Height=width of image (in pixels)")
    parser.add_argument("--pixel_size", type=int,   default=64,    help="Size of the detector (in pixels)")
    parser.add_argument("--angle_nb",   type=int,   default=60,    help="Number of view angles")
    # Network and training
    parser.add_argument("--data_root",  type=str,   default='./data/', help="Path to SLT-10 dataset")
    parser.add_argument("--model_root", type=str,   default='./models/', help="Path to model saving files")
    parser.add_argument("--intensity_max",  type=float,   default=float('inf'), help="maximum photons/pixel")
    parser.add_argument("--intensity_sig",  type=float,   default=0.5, help="std of maximun photons/pixel")
    
    # Optimisation
    parser.add_argument("--num_epochs", type=int,   default=100,    help="Number of epochs")
    parser.add_argument("--batch_size", type=int,   default=512,    help="Batch size")
    parser.add_argument("--reg",        type=float, default=1e-7,   help="Regularisation Parameter")
    parser.add_argument("--lr",         type=float, default=1e-3,   help="Learning Rate")
    parser.add_argument("--step_size",  type=int,   default=10,     help="Scheduler Step Size")
    parser.add_argument("--gamma",      type=float, default=0.5,    help="Scheduler Decrease Rate")
    parser.add_argument("--checkpoint_model", type=str, default="", help="Optional path to checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=0, help="Interval between saving model checkpoints"
    )
    opt = parser.parse_args()
    opt.model_root = Path(opt.model_root)
    print(opt)

    #%% =======================================================================
    # 0. Setting up parameters for training
    # =========================================================================
    # The device of the machine, number of workers...s
    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    #%% =======================================================================
    # 1. Load and normalize training data
    # =========================================================================
    transform = transforms.Compose(
    [transforms.functional.to_grayscale,
     transforms.Resize((opt.img_size, opt.img_size)),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.ImageFolder(root=opt.data_root+"train", transform=transform)
    trainloader = \
        torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=False)
    
    testset = torchvision.datasets.ImageFolder(root=opt.data_root+"test", transform=transform)
    testloader = \
        torch.utils.data.DataLoader(testset, batch_size=opt.batch_size,shuffle=False)
    
    dataloaders = {'train': trainloader, 'val': testloader}
    #inputs, labels = next(iter(dataloaders['val']))
    #inputs = inputs.to(device)

    
    #%% =======================================================================
    # 2. Load forward matrix with full angle data
    # =========================================================================    
    fmat_root = opt.data_root + 'matrices/'
    radon_matrix_path = fmat_root + 'Q{}_D{}.npy'.format(opt.img_size, opt.pixel_size)

    A = np.load(radon_matrix_path)
    
    
    #%% =======================================================================
    # 3. Define a Neural Network
    # ========================================================================= 
    Areduced = radonSpecifyAngles(A, opt.angle_nb)
    meas = Linear(Areduced, True, reg = 1e-5)
    meas.h, meas.w = opt.img_size, opt.img_size
   
    noise = NoNoise(meas)        
    prep = DirectPoisson(1.0, meas)
    denoi = ConvNet()
   
    model = PinvNet(noise, prep, denoi)
        
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    if opt.checkpoint_model:
        model.load_state_dict(torch.load(opt.checkpoint_model))
    
    #%% =======================================================================
    # 4. Define a Loss function optimizer and scheduler
    # =========================================================================
    # Penalization defined in DCAN.py
    loss = nn.MSELoss();
    criterion = Weight_Decay_Loss(loss);
    optimizer = optim.Adam(model.parameters(), lr=opt.lr);
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)    

    #%% =======================================================================
    # 5. Train the network
    # =========================================================================
    #- training parameters
    suffix = 'Q_{}_D_{}_T_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(
           opt.img_size, opt.pixel_size, opt.angle_nb, 
           opt.num_epochs, opt.lr, opt.step_size,
           opt.gamma, opt.batch_size, opt.reg
           )
    title = opt.model_root / ('NET_' + suffix)
    
    print(title)
    Path(opt.model_root).mkdir(parents=True, exist_ok=True)
    
    if opt.checkpoint_interval:
        Path(title).mkdir(parents=True, exist_ok=True)
    #
    model, train_info = train_model(model, criterion, \
            optimizer, scheduler, dataloaders, device, title, num_epochs=opt.num_epochs,\
            disp=True, do_checkpoint=opt.checkpoint_interval)

    #%% =======================================================================
    # 6. Saving the model
    # =========================================================================
    #- network's architecture
    save_net(title, model)

    #- save training history
    params = Train_par(opt.batch_size, opt.lr, opt.img_size,reg=opt.reg);
    params.set_loss(train_info);
    train_path = opt.model_root / ('TRAIN_' + suffix +'.pkl')
    with open(train_path, 'wb') as param_file:
        pickle.dump(params,param_file)