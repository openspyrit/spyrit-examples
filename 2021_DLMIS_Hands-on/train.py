from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import argparse
import sys
from pathlib import Path
import scipy.io as sio

spyritfolder = 'D:/Creatis/Programmes/Python/spyrit'

sys.path.insert(1, spyritfolder);
from function.learning.model_Had_DCAN import compNet, Stat_had, Weight_Decay_Loss
from function.learning.nets import *
from function.misc.disp import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    spyritfolder = Path(spyritfolder)

    # Acquisition
    parser.add_argument("--img_size",   type=int,   default=64,     help="Height / width dimension")
    parser.add_argument("--CR",         type=int,   default=333,    help="Number of patterns")
    # Network and training
    parser.add_argument("--data_root",  type=str,   default=spyritfolder/'data', help="Path to SLT-10 dataset")
    parser.add_argument("--net_arch",   type=int,   default=0,      help="Network architecture (variants for the FCL)")
    parser.add_argument("--precompute_root", type=str, default='./models/', help="Path to precomputed data")
    parser.add_argument("--precompute", type=bool,  default=False,  help="Tells if the precomputed data is available")
    parser.add_argument("--model_root", type=str,   default='./models/', help="Path to model saving files")
    # Optimisation
    parser.add_argument("--num_epochs", type=int,   default=60,      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=256,    help="Size of each training batch")
    parser.add_argument("--reg",        type=float, default=1e-7,   help="Regularisation Parameter")
    parser.add_argument("--lr",         type=float, default=1e-3,   help="Learning Rate")
    parser.add_argument("--step_size",  type=int,   default=20,     help="Scheduler Step Size")
    parser.add_argument("--gamma",      type=float, default=0.2,    help="Scheduler Decrease Rate")
    parser.add_argument("--checkpoint_model", type=str, default="", help="Optional path to checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=0, help="Interval between saving model checkpoints"
    )
    opt = parser.parse_args()
    opt.precompute_root = Path(opt.precompute_root)
    opt.model_root = Path(opt.model_root)
    print(opt)

    #%% =======================================================================
    # 0. Setting up parameters for training
    # =========================================================================
    # The device of the machine, number of workers...
    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #%% =======================================================================
    # 1. Load and normalize STL10
    # =========================================================================
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]. Also
    # RGB images transformed into grayscale images.
    transform = transforms.Compose(
        [transforms.functional.to_grayscale,
         transforms.Resize((opt.img_size, opt.img_size)),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

    trainset = \
        torchvision.datasets.STL10(root=opt.data_root, split='train+unlabeled',download=True, transform=transform)
    trainloader = \
        torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,shuffle=False)

    testset = \
        torchvision.datasets.STL10(root=opt.data_root, split='test',download=True, transform=transform)
    testloader = \
        torch.utils.data.DataLoader(testset, batch_size=opt.batch_size,shuffle=False)

    dataloaders = {'train':trainloader, 'val':testloader}

    #%% =======================================================================
    # 2. Compute mean and covariance of the training images
    # =========================================================================
    my_average_file = Path(opt.precompute_root) / ('Average_{}x{}'.format(opt.img_size, opt.img_size)+'.npy')
    my_cov_file = Path(opt.precompute_root) / ('Cov_{}x{}'.format(opt.img_size, opt.img_size)+'.npy')

    Path(opt.precompute_root).mkdir(parents=True, exist_ok=True)
    if not(my_average_file.is_file()) or not(my_cov_file.is_file()) or opt.precompute:
        print('Computing covariance and mean (overwrite previous files)')
        Mean_had, Cov_had = Stat_had(trainloader, opt.precompute_root)
    else:
        print('Loading covariance and mean')
        Mean_had = np.load(my_average_file)
        Cov_had  = np.load(my_cov_file)

    #%% =======================================================================
    # 3. Define a Neural Network
    # =========================================================================
    model = compNet(opt.img_size, opt.CR, Mean_had, Cov_had, opt.net_arch)

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
    # We  loop over our data iterator, feed the inputs to the
    # network compute the gradient by backpropagation and optimize.
    model, train_info = train_model(model, criterion, \
            optimizer, scheduler, dataloaders, device, opt.model_root, num_epochs=opt.num_epochs,\
            disp=True, do_checkpoint=opt.checkpoint_interval)

    #%% =======================================================================
    # 6. Saving the model
    # =========================================================================
    #- network's architecture
    net_type = ['c0mp', 'comp','pinv', 'free']
    Path(opt.model_root).mkdir(parents=True, exist_ok=True)

    #- training parameters
    suffix = '_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
           opt.img_size, opt.CR, opt.num_epochs, opt.lr, opt.step_size,\
           opt.gamma, opt.batch_size, opt.reg)
    title = opt.model_root / ('NET_'+net_type[opt.net_arch]+suffix)
    print(title)
    save_net(title, model)

   #- save training history
    params = Train_par(opt.batch_size, opt.lr, opt.img_size,reg=opt.reg);
    params.set_loss(train_info);
    train_path = opt.model_root / ('TRAIN_'+net_type[opt.net_arch]+suffix+'.pkl')
    with open(train_path, 'wb') as param_file:
        pickle.dump(params,param_file)