#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import spyrit.misc.walsh_hadamard as wh

from matplotlib import pyplot as plt

from spyrit.learning.model_Had_DCAN import *
from spyrit.misc.disp import torch2numpy
from spyrit.misc.statistics import Cov2Var
from spyrit.learning.nets import *

from spas import read_metadata, reconstruction_hadamard
from spas import ReconstructionParameters, setup_reconstruction, load_noise, reconstruct
from spas.noise import noiseClass
from spas.visualization import *
#from siemens_star_analysis import *

plt.rcParams['text.usetex'] = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Torch device: {device}')



# In[ ]:

f = np.load('./data/zoom_x1_starsector/zoom_x1_starsector_spectraldata.npz')
spectral_data = f['spectral_data']

metadata, acquisition_metadata, spectrometer_parameters, dmd_parameters =     read_metadata('./data/zoom_x1_starsector/zoom_x1_starsector_metadata.json')
wavelengths = np.asarray(acquisition_metadata.wavelengths)

print(f'Spectral data dimensions: {spectral_data.shape}')
print(f'Wavelength range: {wavelengths[0]} - {wavelengths[-1]} nm')

print('\nAcquired data description:')
print(f'Light source: {metadata.light_source}')
print(f'Object: {metadata.object}')
print(f'Filter: {metadata.filter}')
print(f'Patterns: {acquisition_metadata.pattern_amount}')
print(f'Integration time: {spectrometer_parameters.integration_time_ms} ms')


print(metadata)

plt.figure()
plt.plot(wavelengths,spectral_data[0,:])
plt.title('First pattern spectrum')

#%% Full recon with all channels and all hadamard coefficients
H = wh.walsh2_matrix(64)
recon = reconstruction_hadamard(acquisition_metadata.patterns, 'walsh', H, spectral_data)
plt.imshow(np.sum(recon, axis=2), cmap='gray')
plt.colorbar()
plt.title('total intensity image')

#%% Load noise experimental parameters
data = np.load('./fit_model2.npz')
mu = data['mu']
sigma = data['sigma']
coeff = data['k']
noise = noiseClass(mu, sigma, coeff)

#%% Preprocess
def unsplit(raw):
    had = raw[0::2] - raw[1::2]
    return had

def reorder(raw, ind, N): # some pattern indices
    had = np.zeros(N*N)
    had[ind] = raw
    return had

def reorder2(meas, ind, N): # all pattern indices
    y = np.pad(meas, (0, N**2-len(meas)))
    P = permutation_from_ind(ind+1)
    had = np.dot(P,y).reshape((N, N))
    return had
    
def preprocess(raw, ind, N):    # some pattern indices
    had1 = unsplit(raw)
    had  = reorder(had1, ind[0::2]//2, N)
    return had

def preprocess2(raw, ind, N): # all pattern indices
    had1 = unsplit(raw)
    had  = reorder2(had1, ind[0::2]//2, N)
    return had

#%% Spectral binning
lambda_min = 530
lambda_max = 530.1

F_bin_GT, wavelengths_bin_recon, bin_width, noise_bin = spectral_binning(
    spectral_data.T, 
    wavelengths, 
    lambda_min, 
    lambda_max, 
    1, 
    noise)


plt.figure()
plt.plot(F_bin_GT.squeeze())
plt.title(f'Raw measurements at $\lambda$={lambda_min} nm')
plt.xlabel('Raw pattern indices')
plt.ylabel('Intensity (in counts)')

#%% Preprocessed measurements
had = preprocess(F_bin_GT.squeeze(), 
                 np.array(acquisition_metadata.patterns),
                 64)

plt.figure()
plt.plot(had)
plt.title(f'Prep measurements at $\lambda$={lambda_min} nm')
plt.xlabel('Walsh-Hadamard pattern indices')
plt.ylabel('Intensity (in counts)')

print(f'Mean hadmard coefficients: {had[1:].mean():.2f}')

#%% Recon from iwalsh
had_img = np.reshape(had,(64,64))
rec_had = wh.iwalsh2(had_img)
#
plt.figure()
plt.subplot(121)
#plt.imshow(had_img, cmap='gray')
plt.imshow(np.log10(np.abs(had_img)), cmap='gray')
plt.title("Hadamard coefficients")
plt.colorbar()
plt.subplot(122)
plt.imshow(rec_had, cmap='gray')
plt.colorbar()
plt.title("Reconstruction")

plt.figure()
plt.imshow(rec_had, cmap='gray')
plt.colorbar()
plt.title("Reconstruction iwalsh")

#%% Recon from full Hadamard matrix
rec_had_2 = np.matmul(had,H)/H.shape[0]
rec_had_2 = np.reshape(rec_had_2, had_img.shape)
#
plt.figure()
plt.subplot(121)
#plt.imshow(had_img, cmap='gray')
plt.imshow(np.log10(np.abs(had_img)), cmap='gray')
plt.title("Hadamard coefficients")
plt.colorbar()
plt.subplot(122)
plt.imshow(rec_had_2, cmap='gray')
plt.colorbar()
plt.title("Reconstruction")

plt.figure()
plt.imshow(rec_had_2, cmap='gray')
plt.colorbar()
plt.title("Reconstruction matmul")

#%% Direct recon
recon_GT = reconstruction_hadamard(acquisition_metadata.patterns, 
                                   'walsh', H, F_bin_GT.T)

plt.figure()
plt.title(f'Direct recon in range [{lambda_min}-{lambda_max}] nm, M = {64**2}')
plt.imshow(recon_GT, cmap='gray')#, vmin=-8, vmax=15)
plt.colorbar()

print(recon_GT.max())
print(recon_GT.min())

N0_GT = recon_GT.max()


#%%
def subsample(spectral_data, CR):
    
    # If only one wavelength is considered
    if spectral_data.ndim == 1:
        torch_img = np.zeros((2*CR))
        
        pos = spectral_data[0::2][:CR]
        neg = spectral_data[1::2][:CR]
        
        torch_img[0::2] = pos
        torch_img[1::2] = neg
    
    # If spectral_data contains all wavelengths
    if spectral_data.ndim == 2:
        
        torch_img = np.zeros((2*CR, spectral_data.shape[1]))
        
        pos = spectral_data[0::2][:CR,:]
        neg = spectral_data[1::2][:CR,:]
        
        torch_img[0::2,:] = pos
        torch_img[1::2,:] = neg
    
    return torch_img

#%% Spectral binning (and checking a few things)
CR = 2048

imgs = subsample(spectral_data, CR).T
F_bin, wavelengths_bin_recon, bin_width, noise_bin = spectral_binning(
    imgs, wavelengths, lambda_min, lambda_max, 1, noise)

lambda_ind, = np.where((wavelengths > lambda_min) & (wavelengths < lambda_max))
lambda_ind.shape
spectral_data
mu.min()
sigma.max()

math.sqrt(1723)*15


#%% Recon from iwalsh
had = preprocess(F_bin_GT.squeeze()[:2*CR], 
                 np.array(acquisition_metadata.patterns)[:2*CR],
                 64)

had_img = np.reshape(had,(64,64))
rec_had = wh.iwalsh2(had_img)
#
plt.figure()
plt.imshow(np.log10(np.abs(had_img)), cmap='gray')
plt.title(f"Hadamard coefficients, M = {CR}")
plt.colorbar()

plt.figure()
plt.imshow(rec_had, cmap='gray')
plt.colorbar()
plt.title(f"Reconstruction iwalsh, M = {CR}")


#%% Direct recon
recon_fbin = reconstruction_hadamard(acquisition_metadata.patterns[:2*CR], 
                                     'walsh', H, F_bin_GT.T[:2*CR])

plt.figure()
plt.imshow(recon_fbin, cmap='gray')
plt.colorbar()
plt.title(f'Direct recon in range [{lambda_min}-{lambda_max}] nm, M = {CR}')


#%% Load reconstrcution network
img_size = 64
net_arch = 0 # Network variant

# Intensity distribution
N0 = 10000
sig = 0.5

#- Training parameters
num_epochs = 30
lr = 1e-3 
step_size = 10
gamma = 0.5
batch_size = 512
reg = 1e-7

suffix = '_N0_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(
           img_size, CR, num_epochs, lr, step_size,
           gamma, batch_size, reg)

H_network = H / img_size
#Mean = np.load('./stats/Average_64x64.npy')/img_size
#Cov  = np.load('./stats/Cov_64x64.npy')/img_size**2
Mean = np.load('../../spas/data/Average_64x64.npy')/img_size
Cov  = np.load('../../spas/data/Cov_64x64.npy')/img_size**2

model = DenoiCompNet(img_size, CR, Mean, Cov, net_arch, N0, sig, H_network, Cov2Var(Cov))
network_path = './models/NET_c0mp_N0_10000.0_sig_0.5_Denoi_N_64_M_2048_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07'
load_net(network_path, model, device)
model.to(device)

torch_img = torch.from_numpy(F_bin)
torch_img = torch_img.float()
torch_img = torch.reshape(torch_img, (1, 1, 2*CR)) # batches, channels, patterns
torch_img = torch_img.to(device)


#%%
def meas2img_from_ind(meas, ind, N):
    """Return image from measurement vector

    Args:
        meas (ndarray): Measurement image.
        ind (ndarray): Pattern indices
        N (int) : Image size is N x N

    Returns:
        Img (ndarray): Measurement image
    """
    y = np.pad(meas, (0, N**2-len(meas)))
    Perm = permutation_from_ind(ind+1)
    Img = (Perm @ y).reshape((N, N))
    return Img

def stat_completion_ind(meas, M, ind, Cov, Mean):
    
    nx, ny = Mean.shape
    
    # reorder measurements and statistics
    y     = Perm @ (meas.ravel())
    mu    = Perm @ (Mean.ravel())
    #Sigma = Cov[]
    
    # extract blocks corresponding to acquired and missing measurements 
    mu1 = mu[:M]            # mean of acquired measurements
    mu2 = mu[M:]            # mean of missing measurements
    Sigma1  = Sigma[:M,:M]  # covariance of acquired measurements
    Sigma21 = Sigma[M:,:M]  # covariance of missing measurements

    m = y[:M];
    
    # Bayesian denoised completion
    #y1 = mu1 + W1 @ ( m - mu1)
    y1 = m
    #y2 = mu2 + Sigma21 @ np.linalg.lstsq(Sigma1,(y1 - mu1), rcond=None)[0]
    y2 = Sigma21 @ np.linalg.lstsq(Sigma1, y1, rcond=None)[0]
    #y2 = Sigma21 @ np.linalg.inv(Sigma1) @ y1
    y[:M] = y1
    y[M:] = y2
    
    # reorder output
    y = Perm.T @ y
    y = np.reshape(y,(nx, ny))
    
    return y

def stat_completion_matrices(meas, M, Perm, Cov, Mean):
    
    nx, ny = Mean.shape
    
    # reorder measurements and statistics
    y     = Perm @ (meas.ravel())
    mu    = Perm @ (Mean.ravel())
    Sigma = Perm @ Cov @ Perm.T
    
    # extract blocks corresponding to acquired and missing measurements 
    mu1 = mu[:M]            # mean of acquired measurements
    mu2 = mu[M:]            # mean of missing measurements
    Sigma1  = Sigma[:M,:M]  # covariance of acquired measurements
    Sigma21 = Sigma[M:,:M]  # covariance of missing measurements

    m = y[:M];
    
    # Bayesian denoised completion
    #y1 = mu1 + W1 @ ( m - mu1)
    y1 = m
    #y2 = mu2 + Sigma21 @ np.linalg.lstsq(Sigma1,(y1 - mu1), rcond=None)[0]
    y2 = Sigma21 @ np.linalg.lstsq(Sigma1, y1, rcond=None)[0]
    #y2 = Sigma21 @ np.linalg.inv(Sigma1) @ y1
    y[:M] = y1
    y[M:] = y2
    
    # reorder output
    y = Perm.T @ y
    y = np.reshape(y,(nx, ny))
    
    return y


def stat_completion(Im, CR, P, Cov, Mean):
    """Reconstruct (not TESTED)

    Args:
        Cov (np.ndarray): Covariance matrix.
        Mean (np.ndarray): Mean matrix.
        Im (np.ndarray): Data matrix.

    Returns:
        meas (np.ndarray): Measurement vector
    """
    img_size, ny = Im.shape;

    Sigma = np.dot(P,np.dot(Cov,np.transpose(P)))
    mu = np.dot(P, np.reshape(Mean, (img_size**2,1)))
    mu1 = mu[:CR];

    Sigma1 = Sigma[:CR,:CR]
    Sigma21 = Sigma[CR:,:CR]
    
    W_p = np.zeros((img_size**2,CR))
    W_p[:CR,:] = np.eye(CR);
    W_p[CR:,:] = np.dot(Sigma21, np.linalg.inv(Sigma1));
    
    W = np.dot(np.transpose(P),W_p)
    b = np.dot(np.transpose(P),mu)

    f = np.reshape(Im, (img_size**2,1))
    y = np.dot(P, f)
    
    f_star = b + np.dot(W,(y[:CR]-mu1))
    img_comp = np.reshape(f_star,(img_size, img_size));

    return img_comp;



#%% Recon from iwalsh
CR = 2048
had = unsplit(F_bin_GT.squeeze()[:2*CR])
ind = np.array(acquisition_metadata.patterns)[::2]//2
had_img = meas2img_from_ind(had, ind, 64) #np.array(acquisition_metadata.patterns)[:2*CR],64)

rec_had = wh.iwalsh2(had_img)
#
plt.figure()
plt.imshow(np.log10(np.abs(had_img)), cmap='gray')
plt.title(f"Hadamard coefficients, M = {CR}")
plt.colorbar()

plt.figure()
plt.imshow(rec_had, cmap='gray')
plt.colorbar()
plt.title(f"Reconstruction iwalsh, M = {CR}")

#%% Recon from iwalsh using completion
CR = 2048
had = unsplit(F_bin_GT.squeeze()[:2*CR])
ind = np.array(acquisition_metadata.patterns)[::2]//2
P = permutation_from_ind(ind+1)
had_img   = meas2img_from_ind(had, ind, 64) #np.array(acquisition_metadata.patterns)[:2*CR],64)
had_img_c = stat_completion_matrices(had_img, CR, P.T, Cov, Mean)
#had_img_c = stat_completion(had_img, CR, P.T, Cov, Mean)

rec_had_c = wh.iwalsh2(had_img_c)
#
plt.figure()
plt.imshow(np.log10(np.abs(had_img_c - had_img)), cmap='gray')
plt.title(f"Hadamard coefficients, M = {CR}")
plt.colorbar()

plt.figure()
plt.imshow(np.log10(np.abs(had_img_c)), cmap='gray')
plt.title(f"Hadamard coefficients, M = {CR}")
plt.colorbar()

plt.figure()
plt.imshow(rec_had_c, cmap='gray')
plt.colorbar()
plt.title(f"Reconstruction Completion+iwalsh, M = {CR}")

#%%
CR = 2048
had = unsplit(F_bin_GT.squeeze()[:2*CR])
P = Permutation_Matrix(Cov2Var(Cov))
#ind = np.array(acquisition_metadata.patterns)[::2]//2
#P = permutation_from_ind(ind+1)
had_img   = meas2img_from_ind(had, ind, 64) #np.array(acquisition_metadata.patterns)[:2*CR],64)
had_img_c = stat_completion_matrices(had_img, CR, P.T, Cov, Mean)
#had_img_c = stat_completion(had_img, CR, P.T, Cov, Mean)

rec_had_c = wh.iwalsh2(had_img_c)
#
plt.figure()
plt.imshow(np.log10(np.abs(had_img_c - had_img)), cmap='gray')
plt.title(f"Hadamard coefficients, M = {CR}")
plt.colorbar()

plt.figure()
plt.imshow(np.log10(np.abs(had_img_c)), cmap='gray')
plt.title(f"Hadamard coefficients, M = {CR}")
plt.colorbar()

plt.figure()
plt.imshow(rec_had_c, cmap='gray')
plt.colorbar()
plt.title(f"Reconstruction Completion+iwalsh, M = {CR}")


#%% Network recon in [530-730] nm using half of the hadamard coefficients

result = model.forward_reconstruct_pinv(
    torch_img, 1, 1, img_size, img_size,
)

result = (result+1)*model.N0/2
result = result.cpu().detach().numpy().squeeze()

plt.figure()
plt.suptitle('Comparing pinv from direct and network recon')
plt.subplot(121)
plt.imshow(result, cmap='gray')
plt.colorbar()
plt.title('Network')
plt.subplot(122)
plt.imshow(result - recon_fbin.squeeze(), cmap='gray')
plt.colorbar()
plt.tight_layout()
plt.title('Error = Net - Direct')

#%% PINV recon
torch_had = torch_img[0,0,0::2]-torch_img[0,0,1::2]
offset = torch_had[1:].sum()

result = model.forward_reconstruct_pinv(
    torch_img, 1, 1, img_size, img_size, 
)
result = (result+1) * model.N0/2
N0_pinv = result.max()
result = result.cpu().detach().numpy().squeeze()


plt.figure()
plt.imshow(result, cmap='gray')
plt.colorbar()
plt.title(f'PINV recon, M = {CR}')

#%% MMSE recon
model.N0 = N0_GT
 
result = model.forward_reconstruct_comp(
    torch_img, 1, 1, img_size, img_size, 
)

result = (result+1) * model.N0/2
result = result.cpu().detach().numpy().squeeze()

plt.figure()
plt.imshow(result, cmap='gray')
plt.colorbar()
plt.title(f'COMP recon, M = {CR}')

#%% MMSE recon
model.N0 = N0_GT
 
result = model.forward_reconstruct_mmse(
    torch_img, 1, 1, img_size, img_size, 
)

#result = (result+1) * model.N0/2
result = result.cpu().detach().numpy().squeeze()

plt.figure()
plt.imshow(result, cmap='gray')
plt.colorbar()
plt.title(f'MMSE recon, N0 = {model.N0:.1f} (from GT), M = {CR}')

#%% #%% Network recon (no expe = N0 is estimated from GT)
model.N0 = N0_GT
result = model.forward_reconstruct(
    torch_img, 1, 1, img_size, img_size,
    )

result = (result+1) * model.N0/2
result = result.cpu().detach().numpy().squeeze()

plt.figure()
plt.imshow(result, cmap='gray')
plt.colorbar()
plt.title(f'Net recon, N0 = {model.N0:.1f} (from GT), var theo, M = {CR}')

#%% #%% Network recon (no expe = N0 is estimated from PINV)
model.N0 = N0_pinv
result = model.forward_reconstruct(
    torch_img, 1, 1, img_size, img_size,
    )

result = (result+1) * model.N0/2
result = result.cpu().detach().numpy().squeeze()

plt.figure()
plt.imshow(result, cmap='gray')
plt.colorbar()
plt.title(f'Net recon, N0 = {model.N0:.1f} (from pinv), var theo, M = {CR}')

#%% Network recon (expe = N0 is estimated)
result = (model.forward_reconstruct_expe(
    torch_img, 1, 1, img_size, img_size, 
    torch.from_numpy(noise_bin.mu).float().to(device),
    torch.from_numpy(noise_bin.sigma).float().to(device),
    torch.from_numpy(noise_bin.K).float().to(device),
    ))

_,N0 = model.forward_preprocess_expe(torch_img, 1, 1, img_size, img_size)

N0 = N0.cpu().detach().numpy().squeeze()
result = (result+1) * model.N0/2
result = result.cpu().detach().numpy().squeeze()

plt.figure()
plt.imshow(result, cmap='gray')
plt.colorbar()
plt.title(f'Net recon expe, N0 = {N0:.1f} (from pinv), var expe, M = {CR}')

