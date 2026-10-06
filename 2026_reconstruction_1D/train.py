#%%
import numpy as np
from spyrit.misc.statistics import data_loaders_stl10
import matplotlib.pyplot as plt
import torchvision

from spyrit.core.prep import Rerange, Unsplit, Identity, UnsplitRescale, UnsplitRescaleEstim, RescaleEstim
from spyrit.core.noise import PoissonGaussian
from spyrit.core.recon import OrderedDict, TikhoNet, PinvNet
from spyrit.core.nnet import Unet
from spyrit.core.meas import LinearSplit, Linear
import torch.nn as nn
from tools import hadamard

from pathlib import Path

import json
import ast

import imageio.v3 as iio

from PIL import Image
import torch
from spyrit.misc.walsh_hadamard import walsh_matrix, walsh_matrix_2d
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from spyrit.misc.disp import imagesc


# %% PARAMETERS

alpha = 100 # photon intensity
K = 128 # if you want to downsample
N = 128

# Noise parameters - as in experimentation
g =  0.115   # 0.1123

mu = np.load("CalibrationData/dark/mu_dark_image_binned_x3_1.0_12.04.npy")
var = np.load("CalibrationData/dark/var_dark_image_binned_x3_1.0_12.04.npy")

# for unet do everything in photons: 
gamma = 1
mudark = mu.mean() / g  # 254.9
std_mudark = mu.std()/ g 

sigdark = np.sqrt(var.mean())/ g 
std_sigdark = np.sqrt(var.std())/ g 

mu_dark_image = torch.normal( mean=mudark, std=std_mudark, size=(N,1)).to(device).expand(-1, K*2) 
sig_dark_image = torch.normal( mean=sigdark, std=std_sigdark, size=(N,1)).to(device).expand(-1, K*2)   # don't change size of first dimension, expand (repeat) P times

# load Sigma
image_cov = np.load('data/Cov_1_128x128.npy')
Sigma = torch.from_numpy(image_cov).to(device)
#%% Obtain STL10 images

h = 128
batch_size = 50
data_root = './data'

# Dataloader for STL-10 dataset
mode_run = True
if mode_run:
    dataloaders = data_loaders_stl10(
        data_root,
        img_size=h,
        batch_size=batch_size,
        seed=7,
        shuffle=True,
        download=True,
        normalize=False,
    )

#%% Send to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# system
H  = walsh_matrix(N)
H = torch.from_numpy(H).to(device)

# meas_op = LinearSplit(H[0:K, :], noise_model = PoissonGaussian(alpha=alpha, mu=mu_dark_image, g=gamma, sigma=sig_dark_image), device = device)
meas_op = LinearSplit(H[0:K, :], noise_model = PoissonGaussian(alpha=alpha), device = device)
prep_op = UnsplitRescale(alpha)

denoiser = torch.nn.Sequential(OrderedDict({"denoi": Unet()}))

# choose what type of net - remember to update save file name!!! realistically should be automatic

#model = PinvNet(meas_op, prep=Unsplit(), denoi = denoiser, store_H_pinv = True, device = device)
#model = TikhoNet(meas_op, prep = prep_op , sigma=Sigma, device = device) 
model = TikhoNet(meas_op, prep = prep_op , sigma=Sigma,  denoi = denoiser, device = device) 

#%% CHECK - ie run without denoiser to check input 
# model.eval()
# model = model.to(device)

# images, labels = next(iter(dataloaders['train']))
# x = images.to(device)


# with torch.no_grad():
#     #prep
#     y = model.acquire(x)
#     x_hat = model.reconstruct(y/gamma)
#     x_tilde = (x_hat).detach().cpu().numpy().squeeze() 
# del model 

# plt.imshow(x_tilde[5,:,:])
# plt.colorbar()
#%%

from spyrit.core.train import Weight_Decay_Loss

# Parameters
lr = 1e-3
step_size = 10
gamma = 0.5

loss = torch.nn.MSELoss()
criterion = Weight_Decay_Loss(loss)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
# %%
from spyrit.core.train import train_model
from datetime import datetime

# Parameters
model_root = Path("./model")  # path to model saving files
num_epochs = 20  # number of training epochs (num_epochs = 30)
checkpoint_interval = 0  # interval between saving model checkpoints
tb_freq = (
    50  # interval between logging to Tensorboard (iterations through the dataloader)
)

# Path for Tensorboard experiment tracking logs
name_run = "stl10_hadam_positive"
now = datetime.now().strftime("%Y-%m-%d_%H-%M")
tb_path = f"runs/runs_{name_run}_nonoise_m{meas_op.M}/{now}"

# Train the network
model, train_info = train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    device,
    model_root,
    num_epochs=num_epochs,
    disp=True,
    do_checkpoint=checkpoint_interval,
    tb_path=tb_path,
    tb_freq=tb_freq,
)



from spyrit.core.train import save_net

title = f"TIKOgit status_UNET_{h}x{h}_K=128_alpha{alpha}_20epochs"

Path(model_root).mkdir(parents=True, exist_ok=True)
model_path = model_root / (title + ".pth")
train_path = model_root / (title + ".pkl")

if checkpoint_interval:
    Path(model_path).mkdir(parents=True, exist_ok=True)

save_net(model_path, model.denoi)
# save_net(model_root/(title+"_cnn.pth"), pinv_net.denoi.denoi)

# Save training history
import pickle


from spyrit.core.train import Train_par

reg = 1e-7  # Default value
params = Train_par(batch_size, lr, h, reg=reg)
params.set_loss(train_info)

train_path = model_root / (title + ".pkl")

with open(train_path, "wb") as param_file:
    pickle.dump(params, param_file)
torch.cuda.empty_cache()
# %%
