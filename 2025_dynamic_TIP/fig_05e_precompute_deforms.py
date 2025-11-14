"""
Script used to generate and precompute deformation fields used in the experiments of 
Fig. 5e in [ref]. This script is long to run (several hours) and should be run in parallel. 
Results are provided at https://tomoradio-warehouse.creatis.insa-lyon.fr/#collection/66796d3cbaa5a90007058946/folder/6908b71804d23f6e964b142e
"""

# Import bib
import torch
import numpy as np

from pathlib import Path

from spyrit.misc.disp import torch2numpy
from spyrit.core.warp import ElasticDeformation

from scipy.interpolate import griddata
from tqdm import tqdm
import argparse


# Parser
parser = argparse.ArgumentParser(description="Process some parameters.")
parser.add_argument("--n_deforms", type=int, default=500, help="Number of deformations generated")
parser.add_argument("--index_min", type=int, default=0, help="First index of the deformation generated (useful for parallel computing)")
parser.add_argument("--results_root", type=str, default="/deformations", help="Root directory for saving results")

args = parser.parse_args()


# LOAD IMAGE DATA
img_size = 88  # full image side's size in pixels
meas_size = 64  # measurement pattern side's size in pixels (Hadamard matrix)
img_shape = (img_size, img_size)
meas_shape = (meas_size, meas_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dtype = torch.float32

results_root = Path(args.results_root)
Path(results_root).mkdir(parents=True, exist_ok=True)


# MOTION PARAMS
smoothness = 5
n_interpolation = 3

n_deformations = args.n_deforms  # 500 = 8000 / 16 
index_min = args.index_min

print(index_min)

amp_array = np.zeros(n_deformations)


# SIMULATE DEF FIELDS
torch.manual_seed(index_min)   # ensures that the randomness is different when computing in parallel 

with torch.no_grad():
    for index in range(index_min, index_min + n_deformations):
        amp_array[index - index_min] = np.random.uniform(50, 500)

        def_field = ElasticDeformation(amp_array[index - index_min], smoothness, img_shape, 2 * (meas_size**2), n_interpolation, dtype=dtype)

        ## FORWARD MAPPING WITH GRIDDATA
        scale_factor = (torch.tensor(img_shape) - 1)
        def_field_scaled = (def_field.field + 1) / 2 * scale_factor
        def_field_np = torch2numpy(def_field_scaled)

        n_frames, height, width, _ = def_field.field.shape
        interval_1, interval_2 = np.linspace(0, width - 1, width), np.linspace(0, height - 1, height)
        x1, x2 = np.meshgrid(interval_1, interval_2, indexing='xy')
        grid = np.moveaxis(np.array((x1, x2)), 0, -1).reshape((img_size ** 2, 2))

        field_inv_np = np.zeros((n_frames,  img_shape[0], img_shape[1], 2), dtype=np.float32)
 
        n_img_per_batch = 1  # gain de temps pas terrible, même perte de temps en fait?
        n_batchs = n_frames // n_img_per_batch
        bcasted_grid = np.broadcast_to(grid, (n_img_per_batch, img_size**2, 2)).reshape((n_img_per_batch * img_size**2, 2))
        for frame in tqdm(range(n_batchs)):
            frame_beg, frame_end = frame * n_img_per_batch, (frame + 1) * n_img_per_batch
            field_inv_np[frame_beg:frame_end, :, :, 0] = griddata(def_field_np[frame_beg:frame_end, :, :, :].reshape((n_img_per_batch * img_size ** 2, 2)), bcasted_grid[:, 0], bcasted_grid, method='linear', fill_value=0).reshape((n_img_per_batch, *img_shape))  # need looping
            field_inv_np[frame_beg:frame_end, :, :, 1] = griddata(def_field_np[frame_beg:frame_end, :, :, :].reshape((n_img_per_batch * img_size ** 2, 2)), bcasted_grid[:, 1], bcasted_grid, method='linear', fill_value=0).reshape((n_img_per_batch, *img_shape))  # need looping

        ## SAVE DEFORM FIELD
        # use advantage of compression data
        comp_factor = 10
        field_inv_c = np.floor(field_inv_np * comp_factor).astype(np.int16)
        field_c = np.floor(def_field_np * comp_factor).astype(np.int16) 

        name_file = results_root / Path('def_field_index_%d_comp_%d.npz' % (index, comp_factor))
        np.savez_compressed(name_file, direct=field_c, inverse=field_inv_c)

name_file = results_root / Path('amplitudes_index_%d_%d.npy' % (index_min, index_min + n_deformations))
np.save(name_file, amp_array)




# %%
