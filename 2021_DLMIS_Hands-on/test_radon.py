from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import argparse
from PIL import Image
import sys
from pathlib import Path
import scipy.io as sio
import h5py
import argparse


sys.path.append('C:/Users/Admin/Documents/INSA/5GE/tdsi_projet/Code/spyrit-master/spyrit')
import function.learning.model_Radon_DCAN as radon

from function.learning.nets import *
from function.learning.nets import *
#from hadamard_optim import *
from function.misc.disp import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Acquisition
    parser.add_argument("--img_size",   type=int,   default=64,   help="Height / width dimension")
    parser.add_argument("--N0",         type=float, default=0 ,   help="Mean maximum total number of photons")
    parser.add_argument("--sig",        type=float, default=0.15, help="Standard deviation of maximum total number of photons")
    parser.add_argument("--pixel_size", type=int, default=64, help="Number of pixels of the sensor")
    parser.add_argument("--nbAngles", type=int, default=181, help="Number of measured angles")

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

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    transform = transforms.Compose(
        [transforms.functional.to_grayscale,
         transforms.Resize((opt.img_size, opt.img_size)),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

    trainset = \
        torchvision.datasets.STL10(root=opt.data_root, split='train+unlabeled', download=False, transform=transform)
    trainloader = \
        torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=False)

    testset = \
        torchvision.datasets.STL10(root=opt.data_root, split='test', download=False, transform=transform)
    testloader = \
        torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False)

    dataloaders = {'train': trainloader, 'val': testloader}

    #%% #######################################################################
    # A. Load Mean, Cov and matrices
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    my_average_file = Path(opt.precompute_root) / ('Mean_{}x{}'.format(opt.img_size, opt.pixel_size) + '.pt')
    my_cov_file = Path(opt.precompute_root) / ('Cov_{}x{}'.format(opt.img_size, opt.pixel_size) + '.pt')

    Mean_radon = torch.load(my_average_file, map_location=device)
    Cov_radon  = torch.load(my_cov_file, map_location=device)

    radon_matrix_path = '../../models/radon/Q{}_D{}.mat'.format(opt.img_size, opt.pixel_size)
    H = sio.loadmat(radon_matrix_path)
    A = H.get("A")
    A = np.array(A)
    B = torch.from_numpy(A)
    B = B.type(torch.FloatTensor)
  # B = radon.radonSpecifyAngles(B, radon.generateAngles(opt.nbAngles))
    A = torch2numpy(B)

    '''
    pinvA = np.linalg.pinv(A)
    pinvB = torch.from_numpy(pinvA)
    '''

    radon_matrix_path = '../../models/radon/pinv/pinv_Q{}_D{}.mat'.format(opt.img_size,opt.pixel_size)
    H = h5py.File(radon_matrix_path);
    pinvA = H.get("A_pinv")
    pinvA = np.array(pinvA)
    pinvB = torch.from_numpy(pinvA)
    pinvB = torch.transpose(pinvB, 0, 1)
    pinvB = pinvB.type(torch.FloatTensor)
    pinvA = torch2numpy(pinvB)


    #%% #######################################################################
    # A. Load Model
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    net_type = ['c0mp', 'comp','pinv', 'free']

    suffix = '_Q_{}_D_{}_T_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format( \
        opt.img_size, opt.pixel_size, opt.nbAngles, opt.num_epochs, opt.lr, opt.step_size, \
        opt.gamma, opt.batch_size, opt.reg)
    title = opt.model_root + 'NET_'+ net_type[opt.net_arch] + suffix

    net_arch = 2;
    model = radon.compNet(opt.img_size, opt.pixel_size, opt.nbAngles, B, pinvB, variant=opt.net_arch)
    model = model.to(device)
    load_net(title, model)

    #%% #######################################################################
    # B. Run tests
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    test_amt = 2
    inputs, labels = next(iter(dataloaders['val']))
    inputs = inputs.to(device)

    #
    fig, axs = plt.subplots(test_amt, 4)
    fig.suptitle('', fontsize=16)

    for index in range(test_amt):
        i_test = np.random.randint(0, inputs.shape[0])
        gt = inputs[i_test, 0, :, :]
        gt = gt.type(torch.FloatTensor)
        gt_flat = gt.view([1, opt.pixel_size * opt.img_size])
        img_radon = torch.mv(B, gt_flat[0,:])
        inv = torch.mv(pinvB, img_radon)

        ax = axs[index, 0]
        ax.set_title("Ground-truth")
        im = ax.imshow(torch2numpy(gt))
        fig.colorbar(im, ax=ax)
        #
        ax = axs[index, 1]
        ax.set_title("Sinograme")
        img_radon = img_radon.view(opt.nbAngles, opt.pixel_size)
        img_nup = torch2numpy(img_radon)
        im = ax.imshow(np.transpose(img_radon))
        fig.colorbar(im, ax=ax)

        ax = axs[index, 2]
        rec = model.evaluate_fcl(inputs)
        ax.set_title("Radon pinvNet ")
        im = ax.imshow(rec[i_test, 0, :, :])
        fig.colorbar(im, ax=ax)

        ax = axs[index, 3]
        rec2 = inv.view(opt.img_size, opt.img_size)
        rec_np = torch2numpy(rec2)
        ax.set_title("Corrected image")
        im = ax.imshow(rec2)
        fig.colorbar(im, ax=ax)

    plt.show()