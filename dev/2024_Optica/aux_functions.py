"""
Auxiliary functions for the Optica 2024 paper.
"""

import torch
import numpy as np

from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as ssim



def compute_nrmse(x, x_gt, dim=[2,3]):
    # Compute relative error across pixels
    if isinstance(x, np.ndarray):
        nrmse_val = nrmse(x, x_gt)
    else:
        nrmse_val = torch.linalg.norm(x - x_gt, dim=dim)/ torch.linalg.norm(x_gt, dim=dim)
    return nrmse_val


def compute_ssim(x, x_gt):
    if not isinstance(x, np.ndarray):
        x = x.cpu().detach().numpy().squeeze()
        x_gt = x_gt.cpu().detach().numpy().squeeze()
    ssim_val = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        ssim_val[i] = ssim(x[i], x_gt[i], data_range=x[i].max() - x[i].min())
    return torch.tensor(ssim_val)


def compute_metric_batch(images, targets, metric='nrmse', operation='sum'):
    """
    Compute mean and variance of a metric
    """
    if metric == 'nrmse':
        metric_batch = compute_nrmse(images, targets)
    elif metric == 'ssim':
        metric_batch = compute_ssim(images, targets)
    else:
        raise ValueError(f'Metric {metric} not supported')
    
    if operation == 'sum':
        # Sum over all images in the batch
        metric_batch_sum = torch.sum(metric_batch)
        
        # Sum of squares to compute variance
        metric_batch_sq = torch.sum(metric_batch**2)
        return metric_batch_sum, metric_batch_sq
    elif operation == 'mean':
        metric_batch = torch.mean(metric_batch)
        return metric_batch
    else:
        raise ValueError(f'Operation {operation} not supported')


def eval_model_metrics_batch_cum(model, dataloader, device, metrics = ['nrmse', 'ssim'], num_batchs = None):
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
            results_batch_sum, results_batch_sq = compute_metric_batch(outputs, inputs, metric)
            results[metric] = results.get(metric, 0)  + results_batch_sum.cpu().detach().numpy().item()
            results[metric + '_var'] = results.get(metric + '_var', 0) + results_batch_sq.cpu().detach().numpy().item()

        n = n + inputs.shape[0]
    for metric in metrics:
        # Compute mean and variance
        results[metric] = results[metric] / n
        results[metric + '_var'] = results[metric + '_var'] / n - results[metric]**2
        results[metric + '_std_dev'] = results[metric + '_var']**0.5
    return results   
