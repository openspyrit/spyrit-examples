import torch
import numpy as np
from torchvision import datasets, transforms
from pathlib import Path
import spyrit.misc.walsh_hadamard as wh

from spyrit.misc.statistics import Cov2Var
from spyrit.misc.disp import *
from spyrit.misc.metrics import psnr_

from spyrit.core.Acquisition import * 
from spyrit.core.Forward_Operator import *
from spyrit.core.Preprocess import *
from spyrit.core.Data_Consistency import *
from spyrit.core.neural_network import *
from spyrit.core.reconstruction import *
from spyrit.core.training import *

from spyrit.misc.sampling import Permutation_Matrix
from spyrit.misc.disp import imagesc, add_colorbar, noaxis

#NB (15-Sep-22): to debug needs to run
import collections
collections.Callable = collections.abc.Callable

#%% User-defined parameters
img_size = 64 # image size
M = img_size**2//4

N0 = 2    # Image intensity (in photons)
bs = 128 # Batch size

i_im = 72     # image index

data_root = Path('../../data/')
stat_root = Path('../../stat/stl10')
model_root = Path('../../model_v2/dc-net_unet_stl10/reprod')

average_file = stat_root / ('Average_{}x{}'.format(img_size,img_size)+'.npy')
cov_file = stat_root / ('Cov_{}x{}'.format(img_size,img_size)+'.npy')    

seed_list = [0,1,2,3,4,5,6,7,8,9]
noise_list = [0,1,2,3,4,5,6,7,8,9]    # noise sample index

N0_train = [50,10,2]
ph = [50, 10, 2] # nb of photons

#%% A batch of STL-10 test images
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Torch device: {device}')

transform = transforms.Compose(
    [transforms.functional.to_grayscale,
     transforms.Resize((img_size, img_size)),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

testset = \
    datasets.STL10(root=data_root, split='test',download=False, transform=transform)
testloader =  torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)

#%% Plot an image
inputs, _ = next(iter(testloader))
b,c,h,w = inputs.shape

x = inputs.view(b*c,w*h)
x_0 = torch.zeros_like(x)

img_true = x[i_im,:].numpy().reshape((h,w))
imagesc(img_true)

#%% Acquisition matrix, permutation, and statistics
# Init DC-Net
Mean = np.load(average_file)
Cov  = np.load(cov_file)
H =  wh.walsh2_matrix(img_size)

# 2. high-energy order
Ord = Cov2Var(Cov)

# permutation
Perm = Permutation_Matrix(Ord)
Hperm = Perm@H
Pmat = Hperm[:M,:]
Cov_perm = Perm @ Cov @ Perm.T

#%% Init
Forw = Forward_operator_Split_ft_had(Pmat, Perm, img_size, img_size)
Acq = Acquisition_Poisson_approx_Gauss(N0, Forw)
Prep = Preprocess_Split_diag_poisson(N0, M, img_size**2)

# 
DC = Generalized_Orthogonal_Tikhonov(Cov_perm, M, img_size**2)
Denoi = Unet()
model = DC2_Net(Acq, Prep, DC, Denoi)

# Make sure to load a model before running 'meas = Acq(x)', otherwise 
# meas is bugged !!! Still don't know why...
mod = f'dc-net_unet_stl10_rect_N0_50_N_64_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07_seed_0'
load_net(model_root / mod, model, device)
model.eval()

#%% Plot
for noise in noise_list:
    for seed in seed_list:
        f, axs = plt.subplots(len(ph), len(N0_train), figsize=(10,7),  dpi= 100)
    
        img = inputs[i_im, 0, :, :].cpu().detach().numpy()
        rec = []
        
        for ph_i, ph_v in enumerate(ph):
            
            #-- Meas
            torch.manual_seed(noise)    # for reproducibility
            #torch.seed()               # for random measurements
            
            Acq.alpha = ph_v
            meas = Acq(x)
            
            #-- Recon
            rec = []
            for N0_t in N0_train:
                
                model_name = f'dc-net_unet_stl10_rect_N0_{N0_t}_N_64_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07_seed_{seed}'
                
                load_net(model_root / model_name, model, device)
                model.eval() # Mandantory when batchNorm is used in Denoi, facultatory otherwise
        
                Prep.alpha = ph_v # 'Prep.alpha = ph_v' is equivalent to 'model.PreP.alpha = ph_v'
        
                rec_gpu = model.reconstruct(meas)
                rec_cpu = rec_gpu[i_im,0,:,:].cpu().detach().numpy()
                rec.append(rec_cpu)
            
            #-- plot
            for N0_i, N0_v in enumerate(N0_train):   
                #- Plot   
                im=axs[ph_i, N0_i].imshow(rec[N0_i], cmap='gray')
                axs[ph_i, N0_i].set_title(f"{N0_v} ph: ${psnr_(img,rec[N0_i]):.2f}$ dB")
                add_colorbar(im)
                
        # remove all axes
        for ax in iter(axs.flatten()):
            noaxis(ax)
            
        # row labels
        rows = ['{} photons'.format(row) for row in ph]
        for ax, row in zip(axs[:,0], rows):
            ax.set_ylabel(row,  size='large')#, rotation=0,)
            ax.get_yaxis().set_visible(True)
            ax.axis('on')
            #
            #ax.xaxis.set_visible(False)
            plt.setp(ax.spines.values(), visible=False)  # make spines (the box) invisible
            ax.tick_params(left=False, labelleft=False)  # remove ticks and labels for the left axis
            ax.patch.set_visible(False) #remove background patch (only needed for non-white background)
        
        f.subplots_adjust(wspace=0, hspace=0.2)
        #plt.suptitle(f"Measurement with ${N0}$ photons $\pm {sig}$")
        plt.savefig(f"dcnet_robustness_seed_{seed}_noise_{noise}.png", bbox_inches=0)
        plt.close()