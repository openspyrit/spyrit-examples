# # -*- coding: utf-8 -*-
# """
# Created on Wed Oct  4 15:38:33 2023

# @author: ducros
# """
# #%%
# import numpy as np
# #from PIL import Image
# import sys
# sys.path.append('./fonction')
# from fonction.load_data import load_pattern_pos_neg, load_data_pos_neg
# from pathlib import Path


# #%% Hadamard patterns  /!\ DATA NOT IN THE WAREHOUSE YET (hard drive only)
# import matplotlib.pyplot as plt
# from spyrit.misc.disp import add_colorbar

# save_tag = True
# data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/'
# Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/'
# Run = 'RUN0001' 
# save_folder = '/Reconstruction/Mat_rc/'

# # Binning is chosen such that:
# # 56 - 2104 = 2048 rows, hence 512 rows after x4 binning
# # 20 = 128 spectral channels 
# H_pos, H_neg = load_pattern_pos_neg(Dir,Run,4)

# norm = H_pos[0,16:500].mean()
# H_pos = np.flip(H_pos,1).copy() # copy() required to remove negatives strides
# H_neg = np.flip(H_neg,1).copy() # copy() required to remove negatives strides
# H_pos /= norm
# H_neg /= norm

# print(f'Hadamard pattern normalization factor: {norm}')

# f, axs = plt.subplots(3, 1)
# axs[0].set_title('Positive measurement patterns')
# im = axs[0].imshow(H_pos, cmap='gray') 
# add_colorbar(im)
# axs[0].get_xaxis().set_visible(False)

# axs[1].set_title('Negative measurement patterns')
# im = axs[1].imshow(H_neg, cmap='gray') 
# add_colorbar(im)
# axs[1].get_xaxis().set_visible(False)

# axs[2].set_title('Sum')
# im = axs[2].imshow(H_pos + H_neg, cmap='gray') 
# add_colorbar(im)
# axs[2].get_xaxis().set_visible(False)

# # Save
# Nl, Nh, = H_pos.shape
# if save_tag:

#     Path(data_folder+save_folder).mkdir(parents=True, exist_ok=True)
#     # data
#     filename = f'hadamard_matrix_{Nl}_{Nh}_pos.npy'
#     np.save(Path(data_folder+save_folder) / filename, H_pos)
#     filename = f'hadamard_matrix_{Nl}_{Nh}_neg.npy'
#     np.save(Path(data_folder+save_folder) / filename, H_neg)
#     filename = f'hadamard_matrix_{Nl}_{Nh}.npy'
#     np.save(Path(data_folder+save_folder) / filename, H_pos - H_neg)
#     # figure
#     filename = f'hadamard_matrix_{Nl}_{Nh}.pdf'
#     plt.savefig(Path(data_folder+save_folder)/filename, bbox_inches='tight', dpi=600)
    

# # Check the patterns
# # filename = f'motifs_Hadamard_{Nl}_{Nh}.npy'
# # H_2 = np.load(Path(data_folder+save_folder) / filename)
# # H_2 /= H_2[0,16:500].mean()
# # H_2 = np.flip(H_2,1).copy() # copy() required to remove negatives strides

# # H = H_pos - H_neg
# # H = np.flip(H,1)
# # H /= H[0,16:500].mean()
# # H = np.flip(H,1)
# # print(f'error: {np.linalg.norm(H-H_2)/np.linalg.norm(H_2)}')
    
# #%% mRFp + DsRed sample /!\ DATA NOT IN THE WAREHOUSE YET (hard drive only)

# Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/'
# Run = 'RUN0004' 
# save_folder = '/Preprocess/'
# n_channel = 2


# # Binning is chosen such that:
# # 56 - 2104 = 2048 rows, hence 512 rows after x4 binning
# # 20 = 128 spectral channels 
# stack_pos, stack_neg = load_data_pos_neg(Dir, Run, 56, 2104, 4, n_channel)

# # Save
# if save_tag:

#     Path(data_folder+save_folder).mkdir(parents=True, exist_ok=True)
    
#     Nl, Nh, Nc = stack_pos.shape
#     filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_pos.npy'
#     np.save(Path(data_folder+save_folder) / filename, stack_pos)
    
#     filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_neg.npy'
#     np.save(Path(data_folder+save_folder) / filename, stack_neg)
    
#     print('-- Preprocessed measurements saved')
    
# #%% Check the prep data (pos neg should math with seb's prep)
# # data_folder = './data/2023_02_28_mRFP_DsRed_3D/'
# # Run = 'RUN0006' 
# # Nl, Nh, Nc = 512, 128, 128

# # filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}.npy'
# # prep = np.load(Path(data_folder+save_folder) / filename)

# # filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_pos.npy'
# # prep_pos = np.load(Path(data_folder+save_folder) / filename)

# # filename = f'{Run}_Had_{Nl}_{Nh}_{Nc}_neg.npy'
# # prep_neg =  np.load(Path(data_folder+save_folder) / filename)

# # print(f'error: {np.linalg.norm(prep_pos-prep_neg-prep)/np.linalg.norm(prep)}')


# #%% Canonical (pushbroom) patterns  /!\ DATA NOT IN THE WAREHOUSE YET
# import matplotlib.pyplot as plt
# from spyrit.misc.disp import add_colorbar
# from load_data import load_pattern

# save_tag = True
# data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/'
# Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/'
# save_folder = '/Reconstruction/Mat_rc/'
# Run_can = 'RUN0002'     # As no dark pattern was measured we reuse the neg of
# Run_dark = 'RUN0001'    # the fist Hadamard pattern

# # Binning is chosen such that:
# # 56 - 2104 = 2048 rows, hence 512 rows after x4 binning
# # 20 = 128 spectral channels 
# H_can  = load_pattern(Dir, Run_can,  4) # Seb used 750,1000, not the default

# # Dark as first negative of Hadamard patterns
# _, H_neg = load_pattern_pos_neg(Dir,Run_dark,4)
# H_dark = H_neg[0,:].reshape((1,-1))

# H_can  = np.flip(H_can,1).copy() # copy() required to remove negatives strides
# H_dark = np.flip(H_dark,1).copy() # copy() required to remove negatives strides

# H = H_can - H_dark

# norm = H[0,16:500].max()
# H /= norm
# H_can /= norm
# H_dark /= norm

# print(f'Canonical pattern normalization factor: {norm}')

# f, axs = plt.subplots(3, 1)
# axs[0].set_title('Canonical patterns')
# im = axs[0].imshow(H_can, cmap='gray') 
# add_colorbar(im)
# axs[0].get_xaxis().set_visible(False)

# axs[1].set_title('Dark patterns')
# axs[1].plot(H_dark.squeeze())
# #axs[1].get_xaxis().set_visible(False)

# axs[2].set_title('Difference')
# im = axs[2].imshow(H, cmap='gray') 
# add_colorbar(im)
# axs[2].get_xaxis().set_visible(False)

# # Save
# Nl, Nh, = H.shape
# if save_tag:

#     Path(data_folder+save_folder).mkdir(parents=True, exist_ok=True)
#     # data
#     filename = f'canonical_matrix_{Nl}_{Nh}_pos.npy'
#     np.save(Path(data_folder+save_folder) / filename, H_can)
#     filename = f'dark_matrix_{Nl}_{Nh}_neg.npy'
#     np.save(Path(data_folder+save_folder) / filename, H_dark)
#     filename = f'canonical_diff_matrix_{Nl}_{Nh}.npy'
#     np.save(Path(data_folder+save_folder) / filename, H)
#     # figure
#     filename = f'canonical_matrix_{Nl}_{Nh}.pdf'
#     plt.savefig(Path(data_folder+save_folder)/filename, bbox_inches='tight', dpi=600)
    
# # svd
# U, S_can, Vh = np.linalg.svd(H_can) #full_matrices=True
# U, S, Vh = np.linalg.svd(H) #full_matrices=True

# if save_tag:
#     fig = plt.figure()
#     plt.plot(S_can)
#     plt.plot(S)
#     plt.yscale('log')
#     plt.legend(['Raw', 'Diff'])
#     plt.title('Sigular values of measurement matrix')
#     #
#     filename = f'canonical_svd_{Nl}_{Nh}.pdf'
#     plt.savefig(Path(data_folder+save_folder)/filename, bbox_inches='tight', dpi=600)

# #%% mRFp + DsRed sample /!\ DATA NOT IN THE WAREHOUSE YET
# from load_data import load_data

# Dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/'
# Run_can = 'RUN0006'     #
# Run_dark = 'RUN0003'    #
# save_folder = '/Preprocess/'
# save_tag = True
# n_channel = 2

# # Binning is chosen such that:
# # 56 - 2104 = 2048 rows, hence 512 rows after x4 binning
# # 20 = 128 spectral channels 
# stack_can  = load_data(Dir, Run_can, 56, 2104, 4, n_channel)
# stack_dark = load_data(Dir, Run_dark, 56, 2104, 4, n_channel)

# # Save
# if save_tag:

#     Path(data_folder+save_folder).mkdir(parents=True, exist_ok=True)
    
#     Nl, Nh, Nc = stack_can.shape

#     filename = f'{Run_can}_Can_{Nl}_{Nh}_{Nc}_can.npy'
#     np.save(Path(data_folder+save_folder) / filename, stack_can)
#     filename = f'{Run_dark}_Can_{Nl}_{Nh}_{Nc}_dark.npy'
#     np.save(Path(data_folder+save_folder) / filename, stack_dark)     
    
#     print('-- Preprocessed measurements saved')

# -*- coding: utf-8 -*-
#%% Set up
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('./fonction')

from fonction.load_data import  load_pattern, load_pattern_pos_neg, load_data_pos_neg, load_data
from spyrit.misc.disp import add_colorbar


def save_arrays(base_path, arrays: dict):
    for name, arr in arrays.items():
        np.save(base_path / name, arr)


def plot_imshow(ax, data, title):
    ax.set_title(title)
    im = ax.imshow(data, cmap='gray')
    add_colorbar(im)
    ax.get_xaxis().set_visible(False)



# path to data: where to find raw data and where to save prep  
data_folder = './data/2023_03_07_mRFP_DsRed_can_vs_had/'
raw_dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/'

prepped_data = '/Preprocess/' # where to save prepped data
prepped_matrices= '/Reconstruction/Mat_rc/'   # where to save prepped patterns
save_tag = True



#%% Hadamard patterns 

Run = 'RUN0001'

H_pos, H_neg = load_pattern_pos_neg(raw_dir, Run, 4)

norm = H_pos[0,16:500].mean()
H_pos = np.flip(H_pos,1).copy() # copy() required to remove negatives strides
H_neg = np.flip(H_neg,1).copy() # copy() required to remove negatives strides
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



 #%% Hadamard data: mRFp + DsRed sample

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


#%% Canonical (pushbroom) patterns
Run_can = 'RUN0002'
Run_dark = 'RUN0001'

H_can = load_pattern(raw_dir, Run_can, 4)
_, H_neg = load_pattern_pos_neg(raw_dir, Run_dark, 4)
H_dark = H_neg[0, :].reshape((1, -1))


H_can  = np.flip(H_can,1).copy() # copy() required to remove negatives strides
H_dark = np.flip(H_dark,1).copy() # copy() required to remove negatives strides

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


 #%% Canonical data: mRFp + DsRed sample 

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