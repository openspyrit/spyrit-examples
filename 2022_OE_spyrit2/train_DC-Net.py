from __future__ import print_function, division
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import argparse
import sys
from pathlib import Path

from spyrit.learning.model_Had_DCAN import Permutation_Matrix, Weight_Decay_Loss
from spyrit.learning.nets import *
from spyrit.restructured.Updated_Had_Dcan import * 
from spyrit.misc.statistics import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Acquisition
    parser.add_argument("--img_size",   type=int,   default=64,   help="Height / width dimension")
    parser.add_argument("--M",          type=int,   default=512,  help="Number of patterns")
    parser.add_argument("--N0",         type=float, default=10,   help="Mean maximum total number of photons")
    # Network and training
    
    parser.add_argument("--data_train_root",  type=str,   default="data", help="Path to Imagenet train dataset")
    parser.add_argument("--data_val_root",    type=str,   default="data", help="Path to Imagenet validation dataset")
    parser.add_argument("--stat_root", type=str, default="stats", help="Path to precomputed data")
    parser.add_argument("--model_root",type=str, default='./model/', help="Path to model saving files")
    
    # Optimisation
    parser.add_argument("--num_epochs", type=int,   default=2,   help="Number of training epochs")
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
    print(opt)
    
    
    ########################################################################
    # 0. Setting up parameters for training
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # The device of the machine, number of workers...
    # 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    
    ########################################################################
    # 1. Loading and normalizing STL10
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]. Also
    # RGB images transformed into grayscale images.
    transform = transforms.Compose(
        [transforms.functional.to_grayscale,
         torchvision.transforms.RandomCrop(opt.img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    

    # trainset = \
    #       torchvision.datasets.ImageFolder(root=opt.data_train_root, transform=transform, target_transform=None, is_valid_file=None)

    
    # testset = \
    #       torchvision.datasets.ImageFolder(root=opt.data_val_root, transform=transform, target_transform=None, is_valid_file=None)


    trainset = \
          torchvision.datasets.STL10(root=opt.data_train_root, split='train+unlabeled',download=False, transform=transform)
    trainloader = \
        torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,shuffle=False)
 
    testset = \
            torchvision.datasets.STL10(root=opt.data_val_root, split='test',download=False, transform=transform)
    
    testloader = \
         torch.utils.data.DataLoader(testset, batch_size=opt.batch_size,shuffle=False)
    
    dataloaders = {'train':trainloader, 'val':testloader}
    num_img = {'train' : len(trainset), 'val' : len(testset)}
   
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 2. Statistics of the training images
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    my_average_file = Path(opt.stat_root) / ('Average_{}x{}'.format(opt.img_size, opt.img_size)+'.npy')
    my_cov_file = Path(opt.stat_root) / ('Cov_{}x{}'.format(opt.img_size, opt.img_size)+'.npy')
    
    print('Loading covariance and mean')
    Mean = np.load(my_average_file)
    Cov  = np.load(my_cov_file)

    H =  wh.walsh2_matrix(opt.img_size)
    Ord = Cov2Var(Cov)
    Perm = Permutation_Matrix(Ord)
    Hperm = Perm @ H
    Pmat = Hperm[:opt.M,:]
    Cov_perm = Perm @ Cov @ Perm.T

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 3. Define a Neural Network
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    FO = Split_Forward_operator_ft_had(Pmat, Perm)
    Noi = Bruit_Poisson_approx_Gauss(opt.N0, FO)
    Prep = Split_diag_poisson_preprocess(opt.N0, opt.M, opt.img_size**2)
    DC = Generalized_Orthogonal_Tikhonov(sigma_prior = Cov_perm, M = opt.M, 
                                         N = opt.img_size**2)
    Denoi = ConvNet()
    model = DC_Net(Noi, Prep, DC, Denoi)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    if opt.checkpoint_model:
        model.load_state_dict(torch.load(opt.checkpoint_model))
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 4. Define a Loss function optimizer and scheduler
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Penalization defined in DCAN.py
    
    loss = nn.MSELoss();
    criterion = Weight_Decay_Loss(loss);
    optimizer = optim.Adam(model.parameters(), lr=opt.lr);
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 5. Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    # We  loop over our data iterator, feed the inputs to the
    model, train_info = train_model(model, criterion, \
            optimizer, scheduler, dataloaders, device, opt.model_root, num_epochs=opt.num_epochs,\
            disp=True, do_checkpoint=opt.checkpoint_interval)
    
    #%%#####################################################################
    # 6. Saving the model so that it can later be utilized
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #- network's architecture

    train_type = '_N0_{:g}'.format(opt.N0)
        
    #- training parameters
    suffix = '_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
           opt.img_size, opt.M, opt.num_epochs, opt.lr, opt.step_size,\
           opt.gamma, opt.batch_size, opt.reg)

    title = opt.model_root / ('DC_Net_Conv_net' + train_type+suffix)    
    print(title)
    
    Path(opt.model_root).mkdir(parents=True, exist_ok=True)
   
    if opt.checkpoint_interval:
       Path(title).mkdir(parents=True, exist_ok=True)
       
    save_net(title, model)
    
    #- save training history
    params = Train_par(opt.batch_size, opt.lr, opt.img_size,reg=opt.reg);
    params.set_loss(train_info);
    train_path = opt.model_root / ('TRAIN_DC_Net_Conv_net'+train_type+suffix+'.pkl')
    with open(train_path, 'wb') as param_file:
        pickle.dump(params,param_file)
    torch.cuda.empty_cache()