from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import argparse
from PIL import Image
import sys
from pathlib import Path
import scipy.io as sio
import h5py as h5
import scipy.linalg as lin



sys.path.append('/scratch/tdsi_julentin/')
#sys.path.append('C:/Users/Admin/Documents/INSA/5GE/tdsi_projet/Code/spyrit-master/spyrit')
import function.learning.model_Radon_DCAN as radon
from function.learning.nets import *
#from function.misc.disp import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Acquisition
    parser.add_argument("--img_size",   type=int,   default=64,   help="Height / width dimension")
    parser.add_argument("--N0",         type=float, default=0 ,   help="Mean maximum total number of photons")
    parser.add_argument("--sig",        type=float, default=0.15, help="Standard deviation of maximum total number of photons")
    parser.add_argument("--pixel_size", type=int, default=64, help="Number of pixels of the sensor")
    parser.add_argument("--nbAngles", type=int, default=20, help="Number of measured angles")

    # Network and training
    parser.add_argument("--data_root",  type=str,   default="../../data/", help="Path to SLT-10 dataset")
    parser.add_argument("--net_arch",   type=int,   default=2,   help="Network architecture (variants for the FCL)")
    parser.add_argument("--precompute_root", type=str, default="../../models/radon/", help="Path to precomputed data")
    parser.add_argument("--precompute",type=bool, default=False, help="Tells if the precomputed data is available")
    parser.add_argument("--denoise",   type=int, default=0, help="Tells if we use the denoising architecture")
    parser.add_argument("--model_root",type=str, default='../../models/radon/', help="Path to model saving files")
    parser.add_argument("--expe_root", type=str, default="", help="Path to precomputed data")
    # Optimisation
    parser.add_argument("--num_epochs", type=int,   default=50,   help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=1000, help="Size of each training batch")
    parser.add_argument("--reg",        type=float, default=1e-7, help="Regularisation Parameter")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--step_size",  type=int,   default=10,   help="Scheduler Step Size")
    parser.add_argument("--gamma",      type=float, default=0.5,  help="Scheduler Decrease Rate")
    parser.add_argument("--checkpoint_model", type=str, default="", help="Optional path to checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=0, help="Interval between saving model checkpoints"
    )
    opt = parser.parse_args()
    #opt.batch_size = 50
    print(opt)

    ########################################################################
    # 0. Setting up parameters for training
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # The device of the machine, number of workers...
    # 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ########################################################################
    # 1. Loading and normalizing STL10
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
        torchvision.datasets.STL10(root=opt.data_root, split='test',download=False, transform=transform)
    testloader = \
        torch.utils.data.DataLoader(testset, batch_size=opt.batch_size,shuffle=False)
    
    dataloaders = {'train': trainloader, 'val': testloader}
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 2. Statistics of the training images
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    my_average_file = Path(opt.precompute_root) / ('Mean_Q{}D{}'.format(opt.img_size, opt.pixel_size)
                                                   + '.pt')
    my_cov_file = Path(opt.precompute_root) / ('Cov_Q{}D{}'.format(opt.img_size, opt.pixel_size)
                                               + '.pt')
    
    Path(opt.precompute_root).mkdir(parents=True, exist_ok=True)
    if not(my_average_file.is_file()) or not(my_cov_file.is_file()) or opt.precompute:
        print('Computing covariance and mean (overwrite previous files)')

        radon_matrix_path = '../../models/radon/Q{}_D{}'.format(opt.img_size, opt.pixel_size)
        H = sio.loadmat(radon_matrix_path);
        A = H.get("A")
        A = torch.from_numpy(A)

        Mean_radon, Cov_radon = radon.Stat_radon(device, trainloader, opt.precompute_root, A, opt.img_size, opt.pixel_size)
    else:
        print('Loading covariance and mean')
        Mean_radon = torch.load(my_average_file, map_location=device)
        Cov_radon  = torch.load(my_cov_file, map_location=device)

    if opt.expe_root:
        my_transform_file = Path(opt.expe_root) / ('transform_{}x{}'.format(opt.img_size, opt.img_size)+'.mat')
        H = sio.loadmat(my_transform_file);
        H = (1/opt.img_size)*H["H"]
        
        my_average_file = Path(opt.expe_root) / ('Average_{}x{}'.format(opt.img_size, opt.img_size)+'.mat')
        my_cov_file = Path(opt.expe_root) / ('Cov_{}x{}'.format(opt.img_size, opt.img_size)+'.mat')
        Mean_had_1 = sio.loadmat(my_average_file)
        Cov_had_1  = sio.loadmat(my_cov_file)
         
    else:
        print('Loading forward matrix')
        radon_matrix_path = '../../models/radon/Q{}_D{}.mat'.format(opt.img_size,opt.pixel_size)
        #H = sio.loadmat(radon_matrix_path);
        H = h5.File(radon_matrix_path, 'r')
        A = H.get("A")
        A = np.array(A)
        A = np.transpose(A)
        A = torch.from_numpy(A)
        A = radon.radonSpecifyAngles(A, radon.generateAngles(opt.nbAngles))

        print('Computing pseudo inverse of the forward matrix')
        if device == "cuda:0":
            pinvA = lin.pinv(A.cpu().numpy())
        else:
            pinvA = lin.pinv(A.numpy())
        pinvA = torch.from_numpy(pinvA)


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 3. Define a Neural Network
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    if opt.N0==0:
        model = radon.compNet(opt.img_size, opt.pixel_size, opt.nbAngles, A, pinvA,  Mean_radon, Cov_radon, opt.net_arch)
    else:
        if opt.denoise:
            model = radon.DenoiCompNet(opt.img_size, opt.CR, Mean_radon, Cov_radon, opt.net_arch, opt.N0, opt.sig, H);
        else:
            model = radon.noiCompNet(opt.img_size, opt.CR, Mean_radon, Cov_radon, opt.net_arch, opt.N0, opt.sig, H)

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
    criterion = radon.Weight_Decay_Loss(loss);
    optimizer = optim.Adam(model.parameters(), lr=opt.lr);
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 5. Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    # We  loop over our data iterator, feed the inputs to the
    # network compute the gradient by backpropagation and optimize. 
    model, train_info = train_model(model, criterion, \
            optimizer, scheduler, dataloaders, device, opt.model_root, num_epochs=opt.num_epochs,\
            disp=True, do_checkpoint=opt.checkpoint_interval)
    
    #%%#####################################################################
    # 6. Saving the model so that it can later be utilized
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #- network's architecture
    net_type = ['c0mp', 'comp','pinv', 'free']
    Path(opt.model_root).mkdir(parents=True, exist_ok=True)
    recon_type = "";

    if opt.N0==0:
        train_type = ''
    else :
        train_type = '_N0_{:g}_sig_{:g}'.format(opt.N0,opt.sig)
        if opt.denoise:
            recon_type+="_Denoi";
        
    #- training parameters
    suffix = '_Q_{}_D_{}_T_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
           opt.img_size, opt.pixel_size, opt.nbAngles, opt.num_epochs, opt.lr, opt.step_size,\
           opt.gamma, opt.batch_size, opt.reg)
    title = opt.model_root + 'NET_'+net_type[opt.net_arch]+train_type+recon_type+suffix    
    print(title)
    save_net(title, model)
    
   #- save training history
    params = Train_par(opt.batch_size, opt.lr, opt.img_size,reg=opt.reg);
    params.set_loss(train_info);
    train_path = opt.model_root+'TRAIN_'+net_type[opt.net_arch]+train_type+recon_type+suffix+'.pkl'
    with open(train_path, 'wb') as param_file:
        pickle.dump(params,param_file)
