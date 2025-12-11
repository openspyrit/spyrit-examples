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
# replace "np.int32(" with an empty string and ")" with an empty string
tmp = json_metadata["patterns"]
tmp = tmp.replace("np.int32(", "").replace(")", "")
patterns = ast.literal_eval(tmp)  # the list (of list of) of pattern indices (evaluation because stored as text)
wavelengths = ast.literal_eval(json_metadata["wavelengths"])

# Permutation of measurements, that are acquired in an experiment-specific order
img_size = 64
acq_size = img_size
Ord_acq = (-np.array(patterns)[::2] // 2).reshape((acq_size, acq_size))
Ord_rec = torch.ones(img_size, img_size)
Perm_rec = samp.Permutation_Matrix(Ord_rec)
Perm_acq = samp.Permutation_Matrix(Ord_acq).T
Ymeas = samp.reorder(Ymeas, Perm_acq, Perm_rec)

# Post-processing of the measurements
Y = torch.tensor(Ymeas, dtype=torch.float32).T
del Ymeas

# Unbiais by removing the dark current, estimated as the min value of marginals of Y (better?)
dc = tl.sum(Y[:,1])/Y.shape[0] # average of dc over all wavelengths
print(f"Estimated dark current to remove: {dc}")
Y = Y - dc
Y = tl.clip(Y, 0, tl.max(Y))
# Normalization to [0,1]
Y = Y / tl.max(Y)

## Showing the measurements after dark current removal
#plt.imshow(Y.cpu().numpy(), aspect='auto', cmap='gray')
#plt.title("Measurements after dark current removal")
#plt.xlabel("Pattern index")
#plt.ylabel("Wavelength index")
#plt.colorbar()
#plt.show()

## Classic recon

# Reconstruction with pseudo-inverse
acq_size = img_size
meas_op = HadamSplit2d(img_size)
A = meas_op.A  # noiseless operator
Anz = torch.cat([A[0:1,:], A[2:,:]], dim=0)
print(Anz.shape)

# Remove row of zero and remove 16 first bands that are null
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

# define custom foward and adjoint forward functions
def forward(x):
    # x@Anz.T or Anz@x
    # x is of shape (B, N**2)  # batch first
    # reshape to (B, N, N)
    temp = x.reshape((x.shape[0], img_size, img_size))  # Whyyyyyyy >????
    temp = meas_op.forward(temp).T  # shape (M, B)
    # Removing the zero row of A changes the forward A@X
    return torch.cat([temp[0:1,:], temp[2:,:]], dim=0)
def adjoint(y):
    # Also implemented as a contraction in spyrit
    return y@Anz


# Pseudo-inverse reconstruction, the NNLS is quite slow here
X_rec = torch.linalg.lstsq(Anz, Ynz.T).solution.T

# Show cat reconstructed with all wavelengths
plt.figure(figsize=(8,6))
plt.imshow(np.rot90(torch.sum(X_rec, dim=0).reshape(64,64), 2), cmap='gray')
plt.title("Reconstructed image at all wavelengths")
plt.colorbar()
plt.axis('off')

plt.show()

## NMF-based reconstruction
print("------- Starting NMF-based reconstruction --------")

# Running the NMF-based unmixing on the reconstructed hypercube
# Init pinv+spa
rank = 3
Kset, W0, A0 = snpa(X_rec, rank, verbose=True)

# Reconstruction with NMF
lmbd = 1e-4
W_est, A_est, crit = MU_SinglePixel_fast(Ynz, forward, adjoint, tl.abs(A0), tl.abs(W0), lmbd=lmbd, maxA=1, niter=120, n_iter_inner=20, eps=1e-8, verbose=True, print_it=40)  # regularization just for implicit scaling
print("Computation done")


Anorms = [torch.max(A_est[i,:]) for i in range(rank)]
A0norms = [torch.max(A0[i,:]) for i in range(rank)]

# show hypercube at some wavelengths and some spectra
plt.figure(figsize=(10,10))
for i in range(rank):
    plt.subplot(rank,3,3*i+1)
    plt.imshow(np.rot90(A_est[i,:].reshape(64,64), 2), cmap='gray')
    plt.title(f"Abundance map AMU comp. {i+1}")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.subplot(rank,3,3*i+2)
    plt.plot(wavelengths_nz, A0norms[i]*W0[:,i].cpu().numpy())
    plt.plot(wavelengths_nz, Anorms[i]*W_est[:,i].cpu().numpy())
    plt.title(f"Spectrum comp. {i+1}")
    plt.xlabel("Wavelength index")
    plt.ylabel("Intensity") 
    plt.legend(["pinv+SNPA", "AMU"])
    plt.subplot(rank,3,3*i+3)
    plt.imshow(np.rot90(A0[i,:].reshape(64,64), 2), cmap='gray')
    plt.title(f"Abundance map init comp. {i+1}")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.tight_layout()
plt.show()


## Superimpose the four estimated abundance maps in an RBG image with four different colors
#abundance_map_rgb = torch.zeros((n, n, 3))
## First component in red
#abundance_map_rgb[:, :, 0] += A_est[0,:].reshape(n,n)
## Second component in green
#abundance_map_rgb[:, :, 1] += A_est[1,:].reshape(n,n)
## Third component in blue
#abundance_map_rgb[:, :, 2] += A_est[2,:].reshape(n,n)
#if rank > 3:
    ## Fourth component in yellow (red+green)
    #abundance_map_rgb[:, :, 0] += A_est[3,:].reshape(n,n)
    #abundance_map_rgb[:, :, 1] += A_est[3,:].reshape(n,n)
    #if rank > 4:
        ## use magenta for the fifth component (red+blue)
        #abundance_map_rgb[:,:,0] += A_est[4,:].reshape(n,n)
        #abundance_map_rgb[:,:,2] += A_est[4,:].reshape(n,n)
        #if rank > 5:
            ## use cyan for the sixth component (green+blue)
            #abundance_map_rgb[:,:,1] += A_est[5,:].reshape(n,n)
            #abundance_map_rgb[:,:,2] += A_est[5,:].reshape(n,n)
## Clip values to [0,1]
#abundance_map_rgb = torch.clamp(abundance_map_rgb, 0, 1)
#plt.imshow(abundance_map_rgb.cpu().numpy())
#plt.title("Superimposed estimated abundance maps")
#plt.axis('off')
## add color legends
#plt.legend(['Component 1 (Red)', 'Component 2 (Green)', 'Component 3 (Blue)', 'Component 4 (Yellow)', 'Component 5 (Magenta)', 'Component 6 (Cyan)'][:rank], loc='upper right')
#plt.show()