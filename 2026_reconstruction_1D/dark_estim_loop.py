#%% imports
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tools import binArray

#%% params

# bin dark images to match convention of whatever data they'll be used for
bin_fact = 3  # eg here bin spatial dim (y) by 3 : 384 -> 128
G_values = [18.06, 12.04, 6.02, 0.0]   # camera gains (dB)
ti_values = [100.0, 10.0, 1.0]         # integration times (ms)

# True : re-read the raw acquisitions and re-bin them (slow)
# False: reuse the saved binned stack if it already exists
REBUILD_STACKS = False

# where raw dark acquisitions are stored
data_folder = Path(r"C:/Users/ceidigh/Documents/2026-05-05_calib_bruit/dark")

# where processed data and figures are saved
out_folder = Path("CalibrationData")
out_folder.mkdir(exist_ok=True)

# dimensions - could of course extract to avoid hard coding blah blah blah
N, L = 384, 608
n_images = 1000


#%% functions
def raw_folder(ti, G):
    return (
        data_folder
        / f"ti_{ti}ms"
        / (f"obj_{G}dB_source_No source_" f"Lc_600nm_Gr_2_Walsh_im_1x1_" f"ti_{ti}ms_zoom_x1")
        / "raw_data"
    )


def load_dark_stack(ti, G):
    """Return the binned stack of dark images (n_images, N/bin_fact, L) for one (ti, G)."""
    stack_file = out_folder / f"dark_images_binned_x{bin_fact}_{ti}_{G}.npy"

    if stack_file.exists() and not REBUILD_STACKS:
        return np.load(stack_file)

    folder = raw_folder(ti, G)
    dark_images = np.zeros((n_images, N, L))
    n_loaded = 0

    for k in range(n_images):
        file_path = folder / f"spectral_NR_0_Gr_2_Lc_600nm_NA_{k}_NS_0.npz"

        if not file_path.exists():
            print(f"    WARNING: Missing file {file_path}")
            continue

        dark_images[n_loaded] = np.load(file_path)["arr_0"].astype(np.float64)
        n_loaded += 1

    # drop the unused (all-zero) slots so missing files don't bias the statistics
    dark_images = dark_images[:n_loaded]

    # bin dark images to match convention of whatever data they'll be used for
    dark_images = binArray(dark_images, 1, bin_fact, bin_fact, func=np.sum)

    np.save(stack_file, dark_images)
    return dark_images


def compute_and_save_stats(dark_images, ti, G):
    """Per-pixel mean, variance and std over the frames; saved to disk."""
    mu_dark_image = dark_images.mean(axis=0)
    var_dark_image = dark_images.var(axis=0, ddof=1)
    sigma_dark_image = dark_images.std(axis=0, ddof=1)

    np.save(out_folder / f"mu_dark_image_binned_x{bin_fact}_{ti}_{G}.npy", mu_dark_image)
    np.save(out_folder / f"var_dark_image_binned_x{bin_fact}_{ti}_{G}.npy", var_dark_image)
    np.save(out_folder / f"sigma_dark_image_binned_x{bin_fact}_{ti}_{G}.npy", sigma_dark_image)

    return mu_dark_image, var_dark_image, sigma_dark_image


def plot_gain(G, stats):
    """One figure per gain: rows = integration times, columns = mean / variance / sigma."""
    n_rows = len(ti_values)
    fig, axs = plt.subplots(
        n_rows, 3, figsize=(18, 3.4 * n_rows), constrained_layout=True
    )
    fig.suptitle(f"Dark noise: camera gain = {G} dB", fontsize=16)

    for row, ti in enumerate(ti_values):
        mu_img, var_img, sigma_img = stats[ti]

        panels = [
            (mu_img, "Dark Mean Image", r"\hat{\bar{\mu}}_{DARK}"),
            (var_img, "Dark Variance Image", r"\hat{\bar{\sigma}}^2_{DARK}"),
            (sigma_img, "Sigma Dark Image", r"\hat{\bar{\sigma}}_{DARK}"),
        ]

        for col, (img, name, symbol) in enumerate(panels):
            ax = axs[row, col]
            im = ax.imshow(img)

            ax.set_xlabel(r"$\Lambda$")
            if col == 0:
                ax.set_ylabel(rf"$t_i$ = {ti} ms" + "\n" + r"$N_y$")
            else:
                ax.set_ylabel(r"$N_y$")

            ax.set_title(
                name + "\n" + rf"${symbol}$ = {img.mean():.2f} $\pm$ {img.std():.2f}"
            )

            fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.1, shrink=0.9)

    fig.savefig(out_folder / f"dark_noise_binned_x{bin_fact}_G_{G}.png", dpi=150)
    return fig


#%% RUN: loop over gains and integration times
for G in G_values:
    stats = {}

    for ti in ti_values:
        print(f"G = {G} dB, ti = {ti} ms")
        dark_images = load_dark_stack(ti, G)
        stats[ti] = compute_and_save_stats(dark_images, ti, G)
        del dark_images  # free memory before loading the next stack

    plot_gain(G, stats)
    plt.show()

# %%