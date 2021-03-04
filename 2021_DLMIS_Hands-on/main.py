from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from net_sDCAN import *
from nets import *
from disp import *
import imageio
import cv2
from hadamard_optim import *
import fht
import matplotlib.pyplot as plt
import scipy.io as sio
import PIL
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

spyritfolder = 'D:/Creatis/Programmes/Python/spyrit'
sys.path.insert(1, spyritfolder);
from function.learning.model_Had_DCAN import compNet, Stat_had, Weight_Decay_Loss
from function.learning.nets import *
from function.misc.disp import *

#%%#######################################################################
# 3. Trying second Neural network 
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 

def evaluate_meas_data(inputs, model):
    with torch.no_grad():
        x = model.conv1(inputs)
        x = x.view([x.shape[0],1, model.M])
    return x

def evaluate_fcl(inputs, model):
    with torch.no_grad():
        x = model.conv1(inputs)
        x = x.view([x.shape[0],1, model.M])
        x = model.fc1(x)
        x = x.view([x.shape[0], 1, model.n, model.n])  
    return x

def evaluate(inputs, model):   
    with torch.no_grad():
        x = model(inputs)       
    return x

def batch_psnr(torch_batch, output_batch):
    list_psnr = [];
    for i in range(torch_batch.shape[0]):
        img = torch_batch[i, 0, :, :];
        img_out = output_batch[i,0,:,:];
        img = img.cpu().detach().numpy();
        img_out = img_out.cpu().detach().numpy();
        list_psnr.append(psnr(img, img_out));
    return list_psnr;

def dataset_psnr(dataloader, model):
    psnr = [];
    psnr_fc = [];   
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            net_output  = model(inputs);
            net_output2 = evaluate_fcl(inputs, model);
    
        psnr += batch_psnr(inputs, net_output);
        psnr_fc += batch_psnr(inputs, net_output2);               
    psnr = np.array(psnr);
    psnr_fc = np.array(psnr_fc);
    return psnr, psnr_fc

def dataset_meas(dataloader, model):
    meas = [];
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            net_output = evaluate_meas_data(inputs, model);
            raw = net_output[:, 0, :];
            raw = raw.cpu().detach().numpy();
        meas.extend(raw);          
    return meas

def print_mean_std(x, tag=''):  
    print("{}psnr = {} +/- {}".format(tag,np.mean(x),np.std(x)))
    
def count_trainable_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_param(model):
    return sum(p.numel() for p in model.parameters())
    

#%% #######################################################################
# 0. User-defined
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The device of the machine, number of workers...
# 

img_size = 64
M = 333
num_epochs = 60
lr = 1e-3 
step_size = 10
gamma = 0.5
batch_size =256
reg = 3e-7

model_root = '../models/sDCAN/'
data_root = '../data/'

# to save plot using type 1 font
matplotlib.rcParams['pdf.fonttype'] = 42
#%% #######################################################################
# 1. Load data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The device of the machine, number of workers...
# 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)

transform = transforms.Compose(
    [transforms.functional.to_grayscale,
     transforms.Resize((img_size, img_size)),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])


trainset = \
    torchvision.datasets.STL10(root=data_root, split='train+unlabeled',download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)

testset = \
    torchvision.datasets.STL10(root=data_root, split='test',download=False, transform=transform)
testloader =  torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)

dataloaders = {'train':trainloader, 'val':testloader}

#%%#######################################################################
# 2. Load saved Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Take the DCAN network from the corresponding file
#
print('Loading Cov and Mean')    
my_file = "Cov_{}x{}.npy".format(img_size, img_size);
my_fullfile = Path(data_root + "stl10_binary/stat_train/" + my_file);
Cov_had = np.load(Path(data_root)/'stl10_binary'/'stat_train'/my_file)
Mean_had = np.load(Path(data_root)/'stl10_binary'/'stat_train'/'Average_{}x{}.npy'.format(img_size, img_size))


#%% Compute PSNR for all methods
#
suffix = '{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
         img_size, M, num_epochs, lr, step_size, gamma, batch_size, reg)

psnr_net_prob = [];
psnr_net_pinv = [];
psnr_net_free = [];
psnr_prob = [];
psnr_pinv = [];
psnr_free = [];

#-- Proposed -----------------------------------------------------------------#
model = compNet(img_size,M, Mean_had, Cov_had)
model = model.to(device)
title = model_root+'NET_sdcan'+ suffix +'.pth'
load_net(title, model)
#    
psnr_net_prob, psnr_prob = dataset_psnr(dataloaders['val'], model)
print_mean_std(psnr_net_prob,'ProbNet: ')
print_mean_std(psnr_prob,'Prob: ')

print('Number of trainable parameters: {}'.format(count_trainable_param(model)))
print('Total number of parameters: {}'.format(count_param(model)))

#-- Pseudo Inverse -----------------------------------------------------------#
model = compNet(img_size,M,Mean_had,Cov_had)
model = model.to(device)
title = model_root+'NET_sdcanH'+ suffix +'.pth'
load_net(title, model)
#
psnr_net_pinv, psnr_pinv = dataset_psnr(dataloaders['val'], model)
print_mean_std(psnr_net_pinv,'PInvNet: ')
print_mean_std(psnr_pinv,'PInv: ')

print('Number of trainable parameters: {}'.format(count_trainable_param(model)))
print('Total number of parameters: {}'.format(count_param(model)))

#-- Fully learnt -------------------------------------------------------------#
model = compNet(img_size,M,Mean_had,Cov_had)
model = model.to(device)
title = model_root+'NET_sdcannone'+ suffix +'.pth'
load_net(title, model)
#    
psnr_net_free, psnr_free = dataset_psnr(dataloaders['val'], model)
print_mean_std(psnr_net_free,'FreeNet: ')
print_mean_std(psnr_free,'Free: ')

print('Number of trainable parameters: {}'.format(count_trainable_param(model)))
print('Total number of parameters: {}'.format(count_param(model)))

#%% Load measured data
meas = dataset_meas(dataloaders['val'], model)
measTrain = dataset_meas(dataloaders['train'], model)
meas = np.array(meas)
measTrain = np.array(measTrain)

#%% PSNR box plot
fig, ax = plt.subplots()
ax.set_title('PSNR for different methods')
ax.boxplot([psnr_pinv, psnr_prob, psnr_net_pinv, psnr_net_prob],1)
ax.set_xticklabels(['pinv', 'prob', 'pinvNET', 'probNET'])
plt.show()
plt.savefig("boxplot.pdf")


#%% PSNR box plot
plt.rcParams.update({'font.size': 16})
plt.figure()
sns.set_style("whitegrid")
ax = sns.boxplot(data=pd.DataFrame([psnr_pinv, psnr_prob, psnr_net_pinv, psnr_net_prob, psnr_net_prob]).T)
ax.set_xticklabels(['pinv', 'comp', 'pinvNET', 'compNET', 'freeNET']);
ax.set_ylabel('PSNR')
plt.savefig("boxplot_sns.pdf")


#%%
n1 = 2; #2,12 or 2,7
n2 = 7;

sns.jointplot(meas[:,n1], meas[:,n2], kind='reg', ratio=3)#, xlim=[-2,2], ylim=[-5, 5])
plt.xlabel('Hadamard coefficient #{}'.format(n1))
plt.ylabel('Hadamard coefficient #{}'.format(n2))
plt.show()
plt.savefig("scatter_test.pdf")


#%% PLOT LOSS OF BOTH TEST AND TRAINING
train_path = model_root+'TRAIN_sdcan'+suffix
train_net_prob = read_param(train_path)
train_path = model_root+'TRAIN_sdcanH'+suffix
train_net_pinv = read_param(train_path)
train_path = model_root+'TRAIN_sdcannone'+suffix
train_net_free = read_param(train_path)

fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 6))
ax = axs[0]
#ax.set_yscale('log')
ax.set_title('Training set')
ax.set_xlabel('Time (epochs)')
ax.set_ylabel('Loss (MSE)')
ax.plot(train_net_prob.train_loss,'*r-')
ax.plot(train_net_pinv.train_loss, '.g-')
ax.plot(train_net_free.train_loss, 'm-')
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

ax = axs[1]
ax.set_yscale('log')
ax.set_title('Test set')
ax.set_xlabel('Time (epochs)')
ax.plot(train_net_prob.val_loss,'*r--')
ax.plot(train_net_pinv.val_loss,'.g--')
ax.plot(train_net_free.val_loss,'m--')
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

fig.savefig('loss.pdf', dpi=fig.dpi, bbox_inches='tight')

#%% PLOT LOSS OF TEST ONLY
plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(figsize=(10,6))

#ax.set_yscale('log')
#ax.set_title('Test set')
ax.set_xlabel('Time (epochs)')
ax.set_ylabel('Loss (MSE)')
ax.plot(train_net_pinv.val_loss,'g--', linewidth=4)
ax.plot(train_net_prob.val_loss,'r-.', linewidth=4)
ax.plot(train_net_free.val_loss,'m', linewidth=4)
ax.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
plt.grid(True)
ax.legend(('pinvNET', 'compNET', 'freeNET'),  loc='upper right')
fig.savefig('loss_test.pdf', dpi=fig.dpi, bbox_inches='tight')# pad_inches=0.1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
#%% test on images
img_size = 64;
x_crop = 150;
y_crop = 100

img_green = './microscopy/melanoma_cells_green'
img_red   = './microscopy/melanoma_cells_red'
img_pat   = './microscopy/melanoma_cells_pat'

imgs = np.zeros((img_size,img_size,2))

img = imageio.imread(img_green+'.png')
img = img[x_crop:x_crop+img_size, y_crop:y_crop+img_size]
img = 2*(img/np.amax(img))-1;
imgs[:,:,0] = img

img = imageio.imread(img_red+'.png');
img = img[x_crop:x_crop+img_size, y_crop:y_crop+img_size]
img = 2*(img/np.amax(img))-1;
imgs[:,:,1] = img

# Convert and send to GPU
imm = np.transpose(imgs, (2,0,1))
immm = np.reshape(imm,(2,1,64,64))
torch_img = torch.from_numpy(immm);
torch_img = torch_img.to(device);
torch_img = torch_img.float();


#-- Pseudo Inverse -----------------------------------------------------------#
model = Stat_Completion_Network_H(Cov_had, Mean_had, M)
model = model.to(device)
title = model_root+'NET_sdcanH'+ suffix +'.pth'
load_net(title, model)
torch_recon = evaluate(torch_img, model)
torch_recon2 = evaluate_fcl(torch_img, model);

im_pinv_red = torch2numpy(torch_recon2[0,0,:,:,])
im_pinv_green = torch2numpy(torch_recon2[1,0,:,:,])

#-- compNET  -----------------------------------------------------------------#
model = Stat_Completion_Network(Cov_had, Mean_had, M)
model = model.to(device)
title = model_root+'NET_sdcan'+ suffix +'.pth'
load_net(title, model)
torch_recon = evaluate(torch_img, model)
torch_recon2 = evaluate_fcl(torch_img, model);

#
im_comp_red = torch2numpy(torch_recon[0,0,:,:,])
im_comp_green = torch2numpy(torch_recon[1,0,:,:,])
im_error0 = imgs[:,:,0]-im_comp_red
im_error1 = imgs[:,:,1]-im_comp_green

im_blin_red = torch2numpy(torch_recon2[0,0,:,:,])
im_blin_green = torch2numpy(torch_recon2[1,0,:,:,])

#-- freeNET -------------------------------------------------------------#
model = Stat_Completion_Network_none(Cov_had, Mean_had, M)
model = model.to(device)
title = model_root+'NET_sdcannone'+ suffix +'.pth'
load_net(title, model)
torch_recon = evaluate(torch_img, model)
#    
im_free_red = torch2numpy(torch_recon[0,0,:,:,])
im_free_green = torch2numpy(torch_recon[1,0,:,:,])

#%% Save as for Matlab TVAL recon
sio.savemat(img_green  + '.mat', {'X':imgs[:,:,0]})
sio.savemat(img_red + '.mat', {'X':imgs[:,:,1]})

# save patterns
P = torch2numpy(model.conv1.weight.data.view(333,64**2))
sio.savemat(img_pat  + '.mat', {'P':P})

# run main_isbi_completion.m to generate melanoma_cells_*_tval.mat

# load matlab tval recon
im_tval_green = sio.loadmat(img_green  + '_tval.mat')['X_green_tval']
im_tval_red   = sio.loadmat(img_red  + '_tval.mat')['X_red_tval']


#%%
fig, axs = plt.subplots(nrows=2, ncols=4, sharey=True, figsize=(12, 6))
#sns.set_style('ticks')
imx = imgs[:,:,0]
ax = axs[0,0]
cs = ax.imshow(imx, vmin = -1, vmax = 1)
ax.set_title('Ground-truth')

ax = axs[0,1]
cs = ax.imshow(im_tval_green, vmin = -1, vmax = 1)
ax.set_title('tval')

ax = axs[0,2]
cs = ax.imshow(im_free_red, vmin = -1, vmax = 1)
ax.set_title('freeNET')

ax = axs[0,3]
cs = ax.imshow(im_comp_red, vmin = -1, vmax = 1)
ax.set_title('compNET')
#fig.colorbar(cs, ax=axs[0,3])

imx = imgs[:,:,1]
ax = axs[1,0]
cs = ax.imshow(imx, vmin = -1, vmax = 1)

ax = axs[1,1]
cs = ax.imshow(im_tval_red, vmin = -1, vmax = 1)

ax = axs[1,2]
cs = ax.imshow(im_free_green, vmin = -1, vmax = 1)

ax = axs[1,3]
cs = ax.imshow(im_comp_green, vmin = -1, vmax = 1)
#fig.colorbar(cs, ax=axs[1,1])

#%%
im_gt =  np.stack((imgs[:,:,0], imgs[:,:,1],-np.ones((64,64))), axis=2)
im_comp = np.stack((im_comp_red, im_comp_green, -np.ones((64,64))), axis=2)
im_free = np.stack((im_free_red, im_free_green, -np.ones((64,64))), axis=2)
im_pinv = np.stack((im_pinv_red, im_pinv_green, -np.ones((64,64))), axis=2)
im_blin = np.stack((im_blin_red, im_blin_green, -np.ones((64,64))), axis=2)
im_tval = np.stack((im_tval_green, im_tval_red, -np.ones((64,64))), axis=2)

im_gt = 0.5*(im_gt+1);
im_comp = 0.5*(im_comp+1);
im_blin = 0.5*(im_blin+1);
im_free = 0.5*(im_free+1);
im_tval = 0.5*(im_tval+1);
im_pinv = 0.5*(im_pinv+1);


plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(7, 6))
#sns.set_style('ticks')
ax = axs[0,0]
cs = ax.imshow(im_gt, vmin = -1, vmax = 1)
ax.set_title('(a) Ground-Truth')
ax.axis('off')

ax = axs[1,0]
cs = ax.imshow(im_tval, vmin = -1, vmax = 1)
ax.set_title('(c) Total Variation')
ax.axis('off')

ax = axs[0,1]
cs = ax.imshow(im_pinv, vmin = -1, vmax = 1)
ax.set_title('(b) Pseudo Inverse')
ax.axis('off')

ax = axs[1,1]
cs = ax.imshow(im_comp, vmin = -1, vmax = 1)
ax.set_title('(d) compNET')
ax.axis('off')

fig.savefig('microscopy.pdf', dpi=fig.dpi, bbox_inches='tight')# pad_inches=0.1)

print('psnr = ${}$'.format(psnr(imgs[:,:,0],im_pinv_red)))
print('psnr = ${}$'.format(psnr(imgs[:,:,0],im_tval_green)))
print('psnr = ${}$'.format(psnr(imgs[:,:,0],im_comp_red)))

print('psnr = ${}$'.format(psnr(imgs[:,:,1],im_pinv_green)))
print('psnr = ${}$'.format(psnr(imgs[:,:,1],im_tval_red)))
print('psnr = ${}$'.format(psnr(imgs[:,:,1],im_comp_green)))


#%% test on images
img_size = 64;

img = imageio.imread('./microscopy/melanoma_cells_green_64x64.png');
#img = cv2.resize(img, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
img = 2*(img/np.amax(img))-1;

torch_img = torch.from_numpy(img);
torch_img = torch.reshape(torch_img, (1,1,img_size, img_size));
torch_img = torch_img.to(device);
torch_img = torch_img.float();
torch_recon = evaluate(torch_img, model)
img_recon = torch2numpy(torch_recon).reshape((img_size,img_size))


fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 6))
sns.set_style('ticks')
ax = axs[0]
cs = ax.imshow(img, vmin = -1, vmax = 1)
ax.set_title('Ground-truth')
#fig.colorbar(cs, ax=axs[0])

ax = axs[1]
cs = ax.imshow(img_recon, vmin = -1, vmax = 1)
ax.set_title('Recon')
fig.colorbar(cs, ax=axs[1])

ax = axs[2]
cs = ax.imshow(img-img_recon)
ax.set_title('Recon')
fig.colorbar(cs, ax=axs[2])
