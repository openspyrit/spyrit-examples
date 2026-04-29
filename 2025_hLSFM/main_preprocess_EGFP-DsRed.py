
#%% Setup
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('./fonction')

from fonction.load_data import load_pattern_pos_neg, load_data_pos_neg
from spyrit.misc.disp import add_colorbar


def save_arrays(base_path, arrays: dict):
    for name, arr in arrays.items():
        np.save(base_path / name, arr)


def plot_imshow(ax, data, title):
    ax.set_title(title)
    im = ax.imshow(data, cmap='gray')
    add_colorbar(im, 'bottom')
    ax.get_xaxis().set_visible(False)


# where data is / should go
data_folder = './data/2023_03_13_2023_03_14_eGFP_DsRed_3D/'
other_data_folder = './data/2023_02_28_mRFP_DsRed_3D/' #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
raw_dir = other_data_folder + 'Raw_data_chSPSIM_and_SPIM/data_2023_02_28/'



prepped_matrices = 'Reconstruction/Mat_rc/'
prepped_data = 'Preprocess'
save_tag = True


#%% Acquisition patterns
Run = 'RUN0002'

H_pos, H_neg = load_pattern_pos_neg(raw_dir, Run, 4)

# H_pos = np.flip(H_pos,1).copy() # copy() required to remove negatives strides
# H_neg = np.flip(H_neg,1).copy() # copy() required to remove negatives strides

H_pos = H_pos.copy() # copy() required to remove negatives strides
H_neg = H_neg.copy() # copy() required to remove negatives strides


norm = H_pos[0,16:500].mean()
H_pos /= norm
H_neg /= norm
print(f'Hadamard pattern normalization factor: {norm}')

# plot
fig, axs = plt.subplots(3, 1)
plot_imshow(axs[0], H_pos, 'Positive measurement patterns')
plot_imshow(axs[1], H_neg, 'Negative measurement patterns')
plot_imshow(axs[2], H_pos + H_neg, 'Sum')

Nl, Nh = H_pos.shape

# save
if save_tag:
    save_path = Path(data_folder + prepped_matrices)
    save_path.mkdir(parents=True, exist_ok=True)
    save_arrays(save_path, {
        f'motifs_Hadamard_{Nl}_{Nh}_pos.npy': H_pos,
        f'motifs_Hadamard_{Nl}_{Nh}_neg.npy': H_neg,
        f'motifs_Hadamard_{Nl}_{Nh}.npy': H_pos - H_neg,
    })

    plt.savefig(save_path / f'motifs_Hadamard_{Nl}_{Nh}.png',
                bbox_inches='tight', dpi=600)
    

#%% Raw data  eGFP + DsRed sample \
raw_dir = data_folder + 'Raw_data_chSPSIM_and_SPIM/'

save_path = Path(data_folder) / prepped_data
save_path.mkdir(parents=True, exist_ok=True)

T_list = range(1,27)    # slice indices

for t in T_list:
    
    print(f'-- Slice: {t}')
    
    if t<6:
        date = '2023_03_13'
        Run = f'RUN{t+1:04}'
    else:
        date = '2023_03_14'
        Run = f'RUN{t-5:04}'
    
    
    # Binning is chosen such that:
    # 56 - 2104 = 2048 rows, hence 512 rows after x4 binning
    # 20 = 128 spectral channels 
    
    Dir = raw_dir + 'data_' + date + '/'
    stack_pos, stack_neg = load_data_pos_neg(Dir,Run,56,2104,4,20)
    
    # Save
    Nl, Nh, Nc = stack_pos.shape
    filename = f'T{t}_{Run}_{date}_Had_{Nl}_{Nh}_{Nc}_pos.npy'
    np.save(save_path / filename, stack_pos)
    
    filename = f'T{t}_{Run}_{date}_Had_{Nl}_{Nh}_{Nc}_neg.npy'
    np.save(save_path / filename, stack_neg)
    
    print('-- Preprocessed measurements saved')
    
