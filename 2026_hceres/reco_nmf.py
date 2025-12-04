# Loading
import numpy as np
import torch
import tensorly as tl
import matplotlib.pyplot as plt
import json
import ast
from nmf_codes import MU_SinglePixel_fast, snpa
from spyrit.core.meas import HadamSplit2d
import spyrit.misc.sampling as samp
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


# Permutation of measurements
from spyrit.misc import sampling as samp
subsampling_factor = 1
img_size = 64

acq_size = img_size // subsampling_factor
Ord_acq = (-np.array(patterns)[::2] // 2).reshape((acq_size, acq_size))

# Measurement and noise operators
Ord_rec = torch.ones(img_size, img_size)

# %%
# Define the two permutation matrices used to reorder the measurements
# measurement order -> natural order -> reconstruction order
Perm_rec = samp.Permutation_Matrix(Ord_rec)
Perm_acq = samp.Permutation_Matrix(Ord_acq).T
# each element of 'measurements' has shape (measurements, wavelengths)
Ymeas = samp.reorder(Ymeas, Perm_acq, Perm_rec)

print("Shape of measurements tensor (wavelengths x measurements): ", Ymeas.shape)
# Plotting the measurements summed over wavelengths
Ymeas_sum_pos = np.sum(Ymeas[::2,:], axis=1)
Ymeas_sum_neg = np.sum(Ymeas[1::2,:], axis=1)
plt.imshow(Ymeas_sum_neg.reshape((acq_size, acq_size)), cmap='gray')
plt.title("Sum of measurements over wavelengths (negative patterns)")
plt.colorbar()
plt.axis('off')
plt.show()

# Post-processing of the measurements
# Make a wavelength x measurements tensor
Y = torch.tensor(Ymeas, dtype=torch.float32).T
#del Ymeas#, Ymeas_perm  # free memory

# Unbiais by removing the dark current, estimated as the min value of marginals of Y (better?)
#dc = tl.min(tl.sum(Y, axis=0))/Y.shape[0]
dc = tl.sum(Y[:,1])/Y.shape[0] # average of dc over all wavelengths
print(f"Estimated dark current to remove: {dc}")
Y = Y - dc
#print(Y[:5,:5])  # show a small part of Y
# clip negative values
Y = tl.clip(Y, 0, tl.max(Y))
# Normalization to [0,1]
Y = Y / tl.max(Y)

# Showing the measurements after dark current removal
plt.imshow(Y.cpu().numpy(), aspect='auto', cmap='gray')
plt.title("Measurements after dark current removal")
plt.xlabel("Pattern index")
plt.ylabel("Wavelength index")
plt.colorbar()
plt.show()

## Classic recon


n=64

# PAtterns are acquired in order Acq, compared to order nat
# Patterns are stored in order Rec in spyrit
acq_size = n
meas_op = HadamSplit2d(n)
A = meas_op.A  # noiseless operator
Anz = torch.cat([A[0:1,:], A[2:,:]], dim=0)
print(Anz.shape)

# Remove row of zero
# Remove 16 first bands that are null
nbremove = 16
Ynz = torch.cat([Y[nbremove:,0:1], Y[nbremove:,2:]], dim=1)
wavelengths_nz = wavelengths[nbremove:]
print(f"Measurements shape after removing zero rows and first 16 bands: {Ynz.shape}")
# five bands binning
bin_size = 8
Ynz_binned = tl.zeros((Ynz.shape[0]//bin_size, Ynz.shape[1]))
wavelengths_binned = []
for i in range(0, Ynz.shape[0], bin_size):
    if i+bin_size <= Ynz.shape[0]:
        Ynz_binned[i//bin_size,:] = tl.mean(Ynz[i:i+bin_size,:], axis=0)
        wavelengths_binned.append(np.mean(wavelengths_nz[i:i+bin_size]))
    else:
        Ynz_binned[i//bin_size,:] = tl.mean(Ynz[i:,:], axis=0)
        wavelengths_binned.append(np.mean(wavelengths_nz[i:]))
Ynz = Ynz_binned
wavelengths_nz = wavelengths_binned
print(f"Binned measurements shape: {Ynz.shape}")

# useful for algorithm
#sumH = torch.ones((Ynz.shape[0], 2*n**2-1))@Anz  # (H.T @ 1).T
#print(sumH.shape)

# define custom foward and adjoint forward functions
def forward(x):
    # x@Anz.T or Anz@x
    # x is of shape (B, N**2)  # batch first
    # reshape to (B, N, N)
    temp = x.reshape((x.shape[0], n, n))  # Whyyyyyyy >????
    temp = meas_op.forward(temp).T  # shape (M, B)
    # Removing the zero row of A changes the forward A@X
    return torch.cat([temp[0:1,:], temp[2:,:]], dim=0)
def adjoint(y):
    # y is of shape (B, M)
    # need to add a zero column in y, after the first column
    temp = torch.cat([y[:, 0:1], torch.zeros((y.shape[0], 1)), y[:, 1:]], dim=1)  # add zero column
    temp = meas_op.adjoint(temp)  # shape (B, M)
    return temp
def adjoint_faster(y):
    # sad :(
    return y@Anz
#sumH = torch.ones()

# Pseudo-inverse reconstruction
X_rec = torch.linalg.lstsq(Anz, Ynz.T).solution.T
#X_rec = hals_nnls(Anz.T@Ynz.T, Anz.T@Anz, n_iter_max=10, epsilon=1e-8).T  # too slow

print(Ynz.shape)  # should be (num_wavelengths, n^2)
print(X_rec.shape)  # should be (num_wavelengths, n^2)

# show hypercube at some wavelengthsplt.figure(figsize=(12,6))
for i, wl_idx in enumerate([1, 10, 150, 250]):
    plt.subplot(2,2,i+1)
    # pivot by 180 degrees
    plt.imshow(X_rec[wl_idx,:].reshape(64,64), cmap='gray')
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

## NMF-based reconstruction
print("------- Starting NMF-based reconstruction --------")

# Running the NMF-based unmixing on the reconstructed hypercube
# Init pinv+spa
rank = 3
#Kset, W0, A0 = spa(X_rec, rank)
Kset, W0, A0 = snpa(X_rec, rank, verbose=True)

# Reconstruction with NMF
lmbd = 1e-3#[1e-4, 0]
#W_est, A_est, crit = MU_SinglePixel(Ynz, Anz, tl.abs(A0), tl.abs(W0), lmbd=lmbd, maxA=1, niter=10, n_iter_inner=20, eps=1e-8, verbose=True, print_it=1)  # regularization just for implicit scaling
W_est, A_est, crit = MU_SinglePixel_fast(Ynz, forward, adjoint_faster, tl.abs(A0), tl.abs(W0), lmbd=lmbd, maxA=1, niter=100, n_iter_inner=20, eps=1e-8, verbose=True, print_it=20)  # regularization just for implicit scaling
# Normalization of W_est and A_est
#sum_W_est = torch.sum(W_est, dim=0, keepdim=True)
#W_est = W_est/sum_W_est
#A_est = A_est*sum_W_est.T
print("Computation done")

# Plot the sum of the two maps, to see if they cover the whole image
plt.imshow(torch.sum(A_est, axis=0).reshape(n,n), cmap='gray')
plt.title("Sum of the two estimated abundance maps")
plt.colorbar()
plt.axis('off')
plt.show()

# Superimpose the four estimated abundance maps in an RBG image with four different colors
abundance_map_rgb = torch.zeros((n, n, 3))
# First component in red
abundance_map_rgb[:, :, 0] += A_est[0,:].reshape(n,n)
# Second component in green
abundance_map_rgb[:, :, 1] += A_est[1,:].reshape(n,n)
# Third component in blue
abundance_map_rgb[:, :, 2] += A_est[2,:].reshape(n,n)
if rank > 3:
    # Fourth component in yellow (red+green)
    abundance_map_rgb[:, :, 0] += A_est[3,:].reshape(n,n)
    abundance_map_rgb[:, :, 1] += A_est[3,:].reshape(n,n)
    if rank > 4:
        # use magenta for the fifth component (red+blue)
        abundance_map_rgb[:,:,0] += A_est[4,:].reshape(n,n)
        abundance_map_rgb[:,:,2] += A_est[4,:].reshape(n,n)
        if rank > 5:
            # use cyan for the sixth component (green+blue)
            abundance_map_rgb[:,:,1] += A_est[5,:].reshape(n,n)
            abundance_map_rgb[:,:,2] += A_est[5,:].reshape(n,n)
# Clip values to [0,1]
abundance_map_rgb = torch.clamp(abundance_map_rgb, 0, 1)
plt.imshow(abundance_map_rgb.cpu().numpy())
plt.title("Superimposed estimated abundance maps")
plt.axis('off')
# add color legends
plt.legend(['Component 1 (Red)', 'Component 2 (Green)', 'Component 3 (Blue)', 'Component 4 (Yellow)', 'Component 5 (Magenta)', 'Component 6 (Cyan)'][:rank], loc='upper right')
plt.show()