# -*- coding: utf-8 -*-

from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np
import torch
from spyrit.misc.statistics import Cov2Var
from spyrit.core.noise import NoNoise, Poisson
from spyrit.core.prep import SplitPoisson
from spyrit.core.meas import HadamSplit
from spyrit.core.nnet import Unet
from spyrit.core.recon import TikhonovMeasurementPriorDiag, PinvNet, DCNet
from spyrit.misc.statistics import data_loaders_stl10
from spyrit.misc.disp import imagesc
from spyrit.misc.sampling import Permutation_Matrix, meas2img2
from spyrit.core.train import count_param, count_trainable_param, count_memory


alpha = 10.0 #ph/pixel max
H = 64
M = H**2 // 8 # subsampled by factor 8
B = 10

# init reconstrcution networks
cov_file = '../../stat/ILSVRC2012_v10102019/Cov_8_64x64.npy'
#cov_file = '../../../stat/stl10/Cov_64x64.npy'
Cov = np.load(cov_file)

Ord = Cov2Var(Cov)
meas = HadamSplit(M, H, Ord)

noise = Poisson(meas, alpha)
#noise  = NoNoise(meas)    # noiseless
prep  = SplitPoisson(alpha, meas)
pinet = PinvNet(noise, prep)
denoi = Unet()
dcnet = DCNet(noise, prep, Cov, denoi)

# A batch of images
dataloaders = data_loaders_stl10('../../data', img_size=H, batch_size=100)  
x, _ = next(iter(dataloaders['val']))

# use GPU, if available
#device = "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
dcnet = dcnet.to(device)

# send model to GPU, if available
count_trainable_param(dcnet)
count_param(dcnet)
count_memory(dcnet)
print(torch.cuda.memory_summary())

# send batch of images to GPU, if available
x = x.to(device)
print(torch.cuda.memory_summary())

# wamup
y = dcnet(x)

with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory = True) as prof:
    with record_function("model_inference"):
        y = dcnet(x)   # another reconstruction, from the ground-truth image

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))