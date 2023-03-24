import torch
import numpy as np
from torchvision import datasets, transforms
from pathlib import Path
import spyrit.misc.walsh_hadamard as wh

# from spyrit.misc.statistics import stat_walsh_stl10
#from spyrit.learning.model_Had_DCAN import Permutation_Matrix
from spyrit.misc.statistics import Cov2Var
from spyrit.misc.disp import *
from spyrit.misc.metrics import psnr_

from spyrit.core.Acquisition import * 
from spyrit.core.Forward_Operator import *
from spyrit.core.Preprocess import *
from spyrit.core.Data_Consistency import *
from spyrit.core.neural_network import *
from spyrit.core.reconstruction import *
from spyrit.core.training import load_net

from spyrit.misc.sampling import Permutation_Matrix

from spyrit.misc.disp import imagesc, add_colorbar, noaxis

#%% User-defined parameters
img_size = 64 # image size
M = img_size**2//4
bs = 256 # Batch size

i_im = 72     # image index

data_root = Path('../../data/')
stat_root = Path('../../stat/stl10')
model_root = Path('../../model_v2/dc-net_unet_stl10/reprod')

average_file = stat_root / ('Average_{}x{}'.format(img_size,img_size)+'.npy')
cov_file = stat_root / ('Cov_{}x{}'.format(img_size,img_size)+'.npy')    

seed_list = range(10)
noise_list = range(25)    # noise sample index

N0_train = 50
N0_test  = N0_train # nb of photons

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
Acq = Acquisition_Poisson_GaussApprox_sameNoise(N0_test, Forw) # same noise sample across batch
Prep = Preprocess_Split_diag_poisson(N0_test, M, img_size**2)

# 
DC = Generalized_Orthogonal_Tikhonov(Cov_perm, M, img_size**2)
Denoi = Unet()
model = DC2_Net(Acq, Prep, DC, Denoi)

# Make sure to load a model before running 'meas = Acq(x)', otherwise 
# meas is bugged !!! Still don't know why...
mod = f'dc-net_unet_stl10_rect_N0_{N0_train}_N_64_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07_seed_0'
load_net(model_root / mod, model, device)
model.eval()

Prep.N0 = N0_test # or equivalently model.PreP.N0 = N0_test
Acq.alpha = N0_test

#%% Reconstruct all
rec = np.zeros((len(seed_list),len(noise_list),bs,64,64))

img = inputs[:, 0, :, :].cpu().detach().numpy()

for seed in seed_list:
    
    #-- Recon        
    model_name = f'dc-net_unet_stl10_rect_N0_{N0_train}_N_64_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07_seed_{seed}'
    
    load_net(model_root / model_name, model, device)
    model.eval() # Mandantory when batchNorm is used in Denoi, facultatory otherwise
    
    for noise in noise_list:
        
        print(f'\r Noise sample: {noise}', end="")
        
        #-- Meas
        torch.manual_seed(noise)    # for reproducibility
        meas = Acq(x)  
    
        rec_gpu = model.reconstruct(meas)
        rec_cpu = rec_gpu[:,0,:,:].cpu().detach().numpy()
        rec[seed,noise,:,:,:] = rec_cpu
        
    print('\n')

# save
if False:
    with open(f'recon_train_{N0_train}_test_{N0_test}.npy', 'wb') as f:
        np.save(f,rec)
        
    np.save('image.npy',img)

#%% Copute standard deviation
error = np.square(rec - img)

mse_seed = np.mean(error,axis=0)
mse_noise = np.mean(error,axis=1)
mse_image = np.mean(error,axis=2)

std_seed = np.std(rec,axis=0)
std_noise = np.std(rec,axis=1)
std_image = np.std(rec,axis=2)

#%% Plot
plt.rcParams['font.size'] = '20'

i_image_list = [1,2,72] # 1 for donkey; 2 for dog; 72 for baboon;     
i_noise = 0
i_train = 0

for i_image in i_image_list:
    
    # plot
    f, axs = plt.subplots(2, 1, figsize=(6,10))
    
    im = axs[0].imshow(std_noise[i_train,i_image,:,:], cmap='gray')
    axs[0].set_title(f"Noise std (model #{i_train})")
    add_colorbar(im)
    
    im = axs[1].imshow(std_seed[i_noise,i_image,:,:], cmap='gray')
    axs[1].set_title(f"Model std (noise #{i_noise})")
    add_colorbar(im)
    
    # remove all axes
    for ax in iter(axs.flatten()):
        noaxis(ax)
    
    # save
    f.subplots_adjust(wspace=0, hspace=0.2)
    plt.savefig(f"std_image_{i_image}_N0_train_{N0_train}_N0_test_{N0_test}.png", bbox_inches=0)
    plt.close()
    
#%% Plot
i_image = 72

i_train_list = range(3)

for i_train in i_train_list:
    
    # plot
    f, axs = plt.subplots(2, 1, figsize=(6,10))
    
    im = axs[0].imshow(std_image[i_train, i_noise,:,:], cmap='gray')
    axs[0].set_title(f"Image std (Meas #{i_noise})")
    add_colorbar(im)
    
    im = axs[1].imshow(std_noise[i_train, i_image,:,:], cmap='gray')
    axs[1].set_title(f"Meas std (image #{i_image})")
    add_colorbar(im)
    
    
    # remove all axes
    for ax in iter(axs.flatten()):
        noaxis(ax)
    
    # save
    f.subplots_adjust(wspace=0, hspace=0.2)
    plt.savefig(f"std_train_{i_train}_N0_train_{N0_train}_N0_test_{N0_test}.png", bbox_inches=0)
    plt.close()