

# -*- coding: utf-8 -*-
"""
Refactored (no redundant save/plot code, core ops unchanged)
"""

#%% Setup
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append('./fonction')

from fonction.load_data import load_pattern_pos_neg, load_data_pos_neg
from spyrit.misc.disp import add_colorbar

# maybe put in fonction l
def save_arrays(base_path, arrays: dict):
    for name, arr in arrays.items():
        np.save(base_path / name, arr)


def plot_imshow(ax, data, title):
    ax.set_title(title)
    im = ax.imshow(data, cmap='gray')
    add_colorbar(im)
    ax.get_xaxis().set_visible(False)


#%% Acquisition patterns
save_tag = True

data_folder = './data/2023_02_28_mRFP_DsRed_3D/'
raw_data = data_folder + 'Raw_data_chSPSIM_and_SPIM/data_2023_02_28/'
Run = 'RUN0002'

prepped_matrices = 'Reconstruction/Mat_rc/'
prepped_data = 'Preprocess'

matrices_path = Path(data_folder) / prepped_matrices
matrices_path.mkdir(parents=True, exist_ok=True)

data_path = Path(data_folder + prepped_data)
data_path.mkdir(parents=True, exist_ok=True)

H_pos, H_neg = load_pattern_pos_neg(raw_data, Run, 4)

norm = H_pos[0,16:500].mean()
# H_pos = np.flip(H_pos,1).copy()
# H_neg = np.flip(H_neg,1).copy()
H_pos /= norm
H_neg /= norm

# Plot
fig, axs = plt.subplots(3, 1)
plot_imshow(axs[0], H_pos, 'Positive measurement patterns')
plot_imshow(axs[1], H_neg, 'Negative measurement patterns')
plot_imshow(axs[2], H_pos + H_neg, 'Sum')

# Save
Nl, Nh = H_pos.shape
if save_tag:
    save_arrays(matrices_path, {
        f'measurement_matrix_{Nl}_{Nh}_pos.npy': H_pos,
        f'measurement_matrix_{Nl}_{Nh}_neg.npy': H_neg,
        f'measurement_matrix_{Nl}_{Nh}.npy': H_pos - H_neg,
    })

    plt.savefig(matrices_path / f'measurement_matrix_{Nl}_{Nh}.png',
                bbox_inches='tight', dpi=600)


# =======================
# Check the patterns (using other data??) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ??
# =======================
egfp_matrices = 'data/2023_03_13_2023_03_14_eGFP_DsRed_3D/Reconstruction/Mat_rc'
filename = f'motifs_Hadamard_{Nl}_{Nh}.npy'
H_2 = np.load(Path(egfp_matrices)/ filename) 

H_2 /= H_2[0,16:500].mean()
H_2 = np.flip(H_2,1).copy()

H = H_pos - H_neg
H = np.flip(H,1)
H /= H[0,16:500].mean()
H = np.flip(H,1)

print(f'error: {np.linalg.norm(H-H_2)/np.linalg.norm(H_2)}')


#%% data: mRFp + DsRed sample 

T_list = range(4, 25)

for t in T_list:

    if t == 8:
        print(f"-- Skipping corrupted slice: {t}")
        continue

    print(f'-- Slice: {t}')
    Run = f'RUN{t:04}'

    stack_pos, stack_neg = load_data_pos_neg(raw_data, Run, 56, 2104, 4, 20)

    Nl, Nh, Nc = stack_pos.shape

    save_arrays(data_path, {
        f'{Run}_Had_{Nl}_{Nh}_{Nc}_pos.npy': stack_pos,
        f'{Run}_Had_{Nl}_{Nh}_{Nc}_neg.npy': stack_neg,
    })

    print('-- Preprocessed measurements saved')


# =======================
# Check preprocessed data
# =======================

data_folder = './data/2023_02_28_mRFP_DsRed_3D/'
Run = 'RUN0006'
Nl, Nh, Nc = 512, 128, 128


prep = np.load(data_path / f'{Run}_Had_{Nl}_{Nh}_{Nc}.npy')
prep_pos = np.load(data_path / f'{Run}_Had_{Nl}_{Nh}_{Nc}_pos.npy')
prep_neg = np.load(data_path / f'{Run}_Had_{Nl}_{Nh}_{Nc}_neg.npy')

print(f'error: {np.linalg.norm(prep_pos - prep_neg - prep)/np.linalg.norm(prep)}')