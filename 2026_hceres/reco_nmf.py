# Loading
import numpy as np
import torch
import tensorly as tl
import matplotlib.pyplot as plt
import json
import ast
from nmf_codes import spa, MU_SinglePixel
tl.set_backend('pytorch')

data = np.load("./data/2025-11-10_test_HCERES/obj_Cat_bicolor_thin_overlap_source_white_LED_Walsh_im_64x64_ti_9ms_zoom_x1/obj_Cat_bicolor_thin_overlap_source_white_LED_Walsh_im_64x64_ti_9ms_zoom_x1_spectraldata.npz", allow_pickle=True)
Ymeas = data["spectral_data"] # the only valid key --> why

# Metadata --> not a correct naming, contains in particular the pattern indices !!!
file = open("./data/2025-11-10_test_HCERES/obj_Cat_bicolor_thin_overlap_source_white_LED_Walsh_im_64x64_ti_9ms_zoom_x1/obj_Cat_bicolor_thin_overlap_source_white_LED_Walsh_im_64x64_ti_9ms_zoom_x1_metadata.json", "r")
json_metadata = json.load(file)[4]
file.close()

# replace "np.int32(" with an empty string and ")" with an empty string
tmp = json_metadata["patterns"]
tmp = tmp.replace("np.int32(", "").replace(")", "")
patterns = ast.literal_eval(tmp)  # the list (of list of) of pattern indices (evaluation because stored as text...)
wavelengths = ast.literal_eval(json_metadata["wavelengths"])

#%% Permutation of measurements

from spyrit.misc import sampling as samp
subsampling_factor = 1
img_size = 64

acq_size = img_size // subsampling_factor
# Patterns are acquired pos/neg, and the indices are given for pos, neg each time.
# The order of the (+1,-1) patterns is given by patterns, but the indices are multiplied by two consequently.
Ord_acq = (-np.array(patterns)[::2] // 2).reshape((acq_size, acq_size))

# Measurement and noise operators
Ord_rec = torch.ones(img_size, img_size)

# Define the two permutation matrices used to reorder the measurements
# measurement order -> natural order -> reconstruction order
Perm_rec = samp.Permutation_Matrix(Ord_rec)
Perm_acq = samp.Permutation_Matrix(Ord_acq).T
# each element of 'measurements' has shape (measurements, wavelengths)
Ymeas = samp.reorder(Ymeas, Perm_acq, Perm_rec)

print("Shape of measurements tensor (wavelengths x measurements): ", Ymeas.shape)
# Plotting the measurements summed over wavelengths
Ymeas_sum_neg = np.sum(Ymeas[1::2,:], axis=1)
plt.imshow(Ymeas_sum_neg.reshape((acq_size, acq_size)), cmap='gray')
plt.title("Sum of measurements over wavelengths (negative patterns)")
plt.colorbar()
plt.axis('off')
plt.show()


#%% Pre-processing of the measurements

# Make a wavelength x measurements tensor
Y = torch.tensor(Ymeas, dtype=torch.float32).T
del Ymeas  # free memory

# Unbiais by removing the dark current, estimated as the min value of marginals of Y (better?)
#dc = torch.min(torch.sum(Y, axis=0))/Y.shape[0]
dc = torch.sum(Y[:,1])/Y.shape[0] # average of dc over all wavelengths of zero pattern
print(f"Estimated dark current to remove: {dc}")
Y = Y - dc
#print(Y[:5,:5])  # show a small part of Y
# clip negative values for KL later
Y = torch.clip(Y, 0, torch.max(Y))
# Normalization to [0,1]
Y = Y / torch.max(Y)

# Showing the measurements after dark current removal
plt.imshow(Y.cpu().numpy(), aspect='auto', cmap='gray')
plt.title("Measurements after dark current removal")
plt.xlabel("Pattern index")
plt.ylabel("Wavelength index")
plt.colorbar()
plt.show()

#%% Classic recon (Jérémy)

# Reconstruction with pseudo-inverse
from spyrit.core.meas import HadamSplit2d
import spyrit.misc.sampling as samp

n=64
acq_size = n
meas_op = HadamSplit2d(n, n**2 * 2, Ord_rec)
A = HadamSplit2d(n).A  # noiseless operator

# Remove row of zero
# Remove 16 first bands that are null
nbremove = 16
Anz = torch.cat([A[0:1,:], A[2:,:]], dim=0)
Ynz = torch.cat([Y[nbremove:,0:1], Y[nbremove:,2:]], dim=1)
wavelengths_nz = wavelengths[nbremove:]
print(f"Measurements shape after removing zero rows and first 16 bands: {Ynz.shape}")

# eight bands binning
bin_size = 8
Ynz_binned = tl.zeros((Ynz.shape[0]//bin_size, Ynz.shape[1]))
wavelengths_binned = []
for i in range(0, Ynz.shape[0], bin_size):
    if i+bin_size <= Ynz.shape[0]:
        Ynz_binned[i//bin_size, :] = tl.mean(Ynz[i:i+bin_size, :], axis=0)
        wavelengths_binned.append(np.mean(wavelengths_nz[i:i+bin_size]))
    else:
        Ynz_binned[i//bin_size, :] = tl.mean(Ynz[i:,:], axis=0)
        wavelengths_binned.append(np.mean(wavelengths_nz[i:]))
Ynz = Ynz_binned
wavelengths_nz = wavelengths_binned
print(f"Binned measurements shape: {Ynz.shape}")

# Pseudo-inverse reconstruction
print("------- Performing pseudo-inverse reconstruction --------")
X_rec = torch.linalg.lstsq(Anz, Ynz.T).solution.T
print(X_rec.shape)  # should be (num_wavelengths, n^2)

# show hypercube at some wavelengthsplt.figure(figsize=(12,6))
for i, wl_idx in enumerate([1, 10, 150, 250]):
    plt.subplot(2,2,i+1)
    # pivot by 180 degrees
    plt.imshow(np.rot90(X_rec[wl_idx,:].reshape(64,64), 0), cmap='gray')
    plt.title(f"Reconstructed image at wl index {wl_idx} ({wavelengths[wl_idx]:.1f} nm)")
    plt.colorbar()
    plt.axis('off')
    
# show some recons spectra
plot_idx = [0, 1000, 2000, 3000]
plt.figure(figsize=(12,6))
for i, px_idx in enumerate(plot_idx):
    plt.subplot(2,2,i+1)
    plt.plot(X_rec[:,px_idx].cpu().numpy())
    plt.title(f"Reconstructed spectrum at pixel index {px_idx}")
    plt.xlabel("Wavelength index")
    plt.ylabel("Intensity")
plt.show()

#%% NMF-based reconstruction
print("------- Starting NMF-based reconstruction --------")

# Init pinv+spa
Kset, W0, A0 = spa(X_rec, 2)

# Reconstruction with NMF
W_est, A_est, crit = MU_SinglePixel(Ynz, Anz, tl.abs(A0), tl.abs(W0), lmbd=[1e-4, 0], maxA=1, niter=100, n_iter_inner=20, eps=1e-8, verbose=True, print_it=20)  # regularization just for implicit scaling
# Normalization of W_est and A_est
#sum_W_est = torch.sum(W_est, dim=0, keepdim=True)
#W_est = W_est/sum_W_est
#A_est = A_est*sum_W_est.T
print("Computation done")

# PLot the initial and estimated spectra
plt.figure(figsize=(12,4))
plt.plot(wavelengths_nz, W0[:,0].cpu().numpy(), 'b')
plt.plot(wavelengths_nz, W_est[:,0].cpu().numpy(), 'k')
plt.legend(['Init W0', 'Estimated W'])
plt.plot(wavelengths_nz, W0[:,1].cpu().numpy(), 'b')
plt.plot(wavelengths_nz, W_est[:,1].cpu().numpy(), 'k')
plt.legend(['Init W0', 'Estimated W'])
plt.show()

# and their abundance maps, both init and NMF
plt.figure(figsize=(12,4))
plt.subplot(2,2,1)
plt.imshow(A_est[0,:].reshape(n,n), cmap='gray')
plt.title("Estimated abundance map for material 1")
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(A_est[1,:].reshape(n,n), cmap='gray')
plt.title("Estimated abundance map for material 2")
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('off')
plt.subplot(2,2,3)  
plt.imshow(A0[0,:].reshape(n,n), cmap='gray')
plt.title("Init abundance map for material 1")
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('off')
plt.subplot(2,2,4)
plt.imshow(A0[1,:].reshape(n,n), cmap='gray')
plt.title("Init abundance map for material 2")
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('off')
plt.show()