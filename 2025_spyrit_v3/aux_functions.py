"""
Auxiliary functions for the Optics Express 2024 paper.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as ssim


# DISPLAY FUNCTIONS
def imagesc_mod(
    img,
    title="",
    figsize=(5, 5),
    colormap=plt.cm.gray,
    title_fontsize=16,
    dpi=100,
    minmax=None,
    showscale=False,
    **kwargs,
):
    """
    Plot images with a custom colormap and a custom color for 'nan' values.
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    # clean the axes
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    # set min max for the colormap
    if minmax is None:
        minmax = (img[~img.isnan()].min(), img[~img.isnan()].max())
    # define the color for 'nan' values
    colormap.set_bad(color="grey")
    plt.imshow(img, cmap=colormap, vmin=minmax[0], vmax=minmax[1], **kwargs)
    plt.title(title, fontsize=title_fontsize)
    if showscale:
        # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        plt.colorbar(orientation="vertical")
    plt.show()


def split_meas2img(measurements, meas_operator):
    r"""
    Generates a 2D image from split measurements acquired from a LinearSplit or
    HadamSplit operator.

    /!\ The measurements must be in the alternating positive / negative format.

    Using spyrit 2.3.2
    """
    M = meas_operator.M
    N = meas_operator.N
    h, w = meas_operator.meas_shape
    # using 'nan' so that we can show them with a custom color (see imagesc_mod)
    img_pos = torch.full((N,), torch.tensor(float("nan")))  # even rows
    img_neg = torch.full((N,), torch.tensor(float("nan")))  # odd rows

    # split the measurements in pos/neg, then apply meas2img to each
    meas = measurements.view(2 * M)
    meas_pos = meas[0::2]
    meas_neg = meas[1::2]

    # fill img_pos and img_neg with the measurements
    if hasattr(meas_operator, 'indices'):
        indices = meas_operator.indices[:M]
    else:
        indices = np.arange(0, h*w, dtype=int)
        
    img_pos[indices] = meas_pos
    img_neg[indices] = meas_neg

    # concatenate and reshape the images
    img = torch.cat((img_pos.reshape(h, w), img_neg.reshape(h, w)), dim=0)

    return img


def center_measurements(measurements):
    r"""
    Centers the measurements so that the max value is the opposite of the min
    value. This is useful for visualization purposes.
    """
    max_val = measurements[~measurements.isnan()].max()
    min_val = measurements[~measurements.isnan()].min()
    return measurements - (max_val + min_val) / 2


def compute_nrmse(x, x_gt, dim=[2, 3]):
    # Compute relative error across pixels
    if isinstance(x, np.ndarray):
        nrmse_val = nrmse(x, x_gt)
    else:
        nrmse_val = torch.linalg.norm(x - x_gt, dim=dim) / torch.linalg.norm(
            x_gt, dim=dim
        )
    return nrmse_val


def compute_ssim(x, x_gt):
    if not isinstance(x, np.ndarray):
        x = x.cpu().detach().numpy().squeeze()
        x_gt = x_gt.cpu().detach().numpy().squeeze()
    ssim_val = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        ssim_val[i] = ssim(x[i], x_gt[i], data_range=x[i].max() - x[i].min())
    return torch.tensor(ssim_val)


def compute_metric_batch(images, targets, metric="nrmse", operation="sum"):
    """
    Compute mean and variance of a metric
    """
    if metric == "nrmse":
        metric_batch = compute_nrmse(images, targets)
    elif metric == "ssim":
        metric_batch = compute_ssim(images, targets)
    else:
        raise ValueError(f"Metric {metric} not supported")

    if operation == "sum":
        # Sum over all images in the batch
        metric_batch_sum = torch.sum(metric_batch)

        # Sum of squares to compute variance
        metric_batch_sq = torch.sum(metric_batch**2)
        return metric_batch_sum, metric_batch_sq
    elif operation == "mean":
        metric_batch = torch.mean(metric_batch)
        return metric_batch
    else:
        raise ValueError(f"Operation {operation} not supported")


def eval_model_metrics_batch_cum(
    model, dataloader, device, metrics=["nrmse", "ssim"], num_batchs=None
):
    """
    Compute metrics meand and variance for a dataset, accumulating across batches
    """
    model.eval()
    results = {}
    n = 0
    for i, (inputs, _) in enumerate(dataloader):
        if num_batchs is not None and i >= num_batchs:
            break
        inputs = inputs.to(device)
        outputs = model(inputs)
        for metric in metrics:
            # Accumulate sum and sum of squares across batches
            results_batch_sum, results_batch_sq = compute_metric_batch(
                outputs, inputs, metric
            )
            results[metric] = (
                results.get(metric, 0) + results_batch_sum.cpu().detach().numpy().item()
            )
            results[metric + "_var"] = (
                results.get(metric + "_var", 0)
                + results_batch_sq.cpu().detach().numpy().item()
            )

        n = n + inputs.shape[0]
    for metric in metrics:
        # Compute mean and variance
        results[metric] = results[metric] / n
        results[metric + "_var"] = results[metric + "_var"] / n - results[metric] ** 2
        results[metric + "_std_dev"] = results[metric + "_var"] ** 0.5
    return results
