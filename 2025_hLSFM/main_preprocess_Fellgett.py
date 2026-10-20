
#%% Set up
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('./fonction')

from fonction.load_data import  load_pattern, load_pattern_pos_neg, load_data_pos_neg, load_data, plot_imshow, save_arrays

# Data paths   
data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/' # data folder for Can vs Had sample 
raw_dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/'     # location of raw data measurements (downloaded from PILOT)

prepped_data = '/Preprocess/'               # where prepped data will be saved to
prepped_matrices= '/Reconstruction/Mat_rc/' # where prepped motifs will be saved to
save_tag = True



#%% Prep Hadamard patterns 

Run = 'RUN0001'

H_pos, H_neg = load_pattern_pos_neg(raw_dir, Run, 4)

norm = H_pos[0,16:500].mean()
H_pos = np.flip(H_pos,1).copy() # copy() required to remove negatives strides
H_neg = np.flip(H_neg,1).copy() 
H_pos /= norm
H_neg /= norm

print(f'Hadamard pattern normalization factor: {norm}')

fig, axs = plt.subplots(3, 1)
plot_imshow(axs[0], H_pos, 'Positive measurement patterns')
plot_imshow(axs[1], H_neg, 'Negative measurement patterns')
plot_imshow(axs[2], H_pos + H_neg, 'Sum')

Nl, Nh = H_pos.shape

if save_tag:

    out = Path(data_folder+prepped_matrices)
    out.mkdir(parents=True, exist_ok=True)

    save_arrays(out, {
        f'hadamard_matrix_{Nl}_{Nh}_pos.npy': H_pos,
        f'hadamard_matrix_{Nl}_{Nh}_neg.npy': H_neg,
        f'hadamard_matrix_{Nl}_{Nh}.npy': H_pos - H_neg,
    })

    plt.savefig(out / f'hadamard_matrix_{Nl}_{Nh}.pdf',
                bbox_inches='tight', dpi=600)



 #%% Prep Hadamard data: mRFp + DsRed sample

Run = 'RUN0004'
n_channel = 2

stack_pos, stack_neg = load_data_pos_neg(raw_dir, Run, 56, 2104, 4, n_channel)

if save_tag:
    out = Path(data_folder+prepped_data)
    out.mkdir(parents=True, exist_ok=True)

    Nl, Nh, Nc = stack_pos.shape

    save_arrays(out, {
        f'{Run}_Had_{Nl}_{Nh}_{Nc}_pos.npy': stack_pos,
        f'{Run}_Had_{Nl}_{Nh}_{Nc}_neg.npy': stack_neg,
    })

    print('-- Preprocessed measurements saved')


#%% Prep Canonical (pushbroom) patterns
Run_can = 'RUN0002'
Run_dark = 'RUN0001'

H_can = load_pattern(raw_dir, Run_can, 4)
_, H_neg = load_pattern_pos_neg(raw_dir, Run_dark, 4)
H_dark = H_neg[0, :].reshape((1, -1)) # use negative hadamard as dark


H_can  = np.flip(H_can,1).copy() # copy() required to remove negatives strides
H_dark = np.flip(H_dark,1).copy() 

H = H_can - H_dark

norm = H[0,16:500].max()
H /= norm
H_can /= norm
H_dark /= norm

print(f'Canonical pattern normalization factor: {norm}')

fig, axs = plt.subplots(3, 1)
plot_imshow(axs[0], H_can, 'Canonical patterns')

axs[1].set_title('Dark patterns')
axs[1].plot(H_dark.squeeze())

plot_imshow(axs[2], H, 'Difference')

Nl, Nh = H.shape

if save_tag:
    out = Path(data_folder+prepped_matrices)
    out.mkdir(parents=True, exist_ok=True)

    save_arrays(out, {
        f'canonical_matrix_{Nl}_{Nh}_pos.npy': H_can,
        f'dark_matrix_{Nl}_{Nh}_neg.npy': H_dark,
        f'canonical_diff_matrix_{Nl}_{Nh}.npy': H,
    })

    plt.savefig(out / f'canonical_matrix_{Nl}_{Nh}.pdf',
                bbox_inches='tight', dpi=600)

    # SVD
    _, S_can, _ = np.linalg.svd(H_can)
    _, S, _ = np.linalg.svd(H)

    fig = plt.figure()
    plt.plot(S_can)
    plt.plot(S)
    plt.yscale('log')
    plt.legend(['Raw', 'Diff'])
    plt.title('Singular values of measurement matrix')

    plt.savefig(out / f'canonical_svd_{Nl}_{Nh}.pdf',
                bbox_inches='tight', dpi=600)


 #%% Prep Canonical data: mRFp + DsRed sample 

Run_can = 'RUN0006'
Run_dark = 'RUN0003'
n_channel = 2

stack_can = load_data(raw_dir, Run_can, 56, 2104, 4, n_channel)
stack_dark = load_data(raw_dir, Run_dark, 56, 2104, 4, n_channel)

if save_tag:
    out = Path(data_folder+prepped_data)
    out.mkdir(parents=True, exist_ok=True)
    Nl, Nh, Nc = stack_can.shape

    save_arrays(out, {
        f'{Run_can}_Can_{Nl}_{Nh}_{Nc}_can.npy': stack_can,
        f'{Run_dark}_Can_{Nl}_{Nh}_{Nc}_dark.npy': stack_dark,
    })

    print('-- Preprocessed measurements saved')