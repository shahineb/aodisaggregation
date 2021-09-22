import torch
import numpy as np
from scipy.stats import pearsonr


def compute_scores(prediction_3d, groundtruth_3d, targets_2d, aggregate_fn):
    """Computes prediction scores

    Args:
        prediction_3d (torch.Tensor): (time, lat, lon, lev)
        groundtruth_3d (torch.Tensor): (time, lat, lon, lev)
        targets_2d (torch.Tensor): (time, lat, lon)
        aggregate_fn (callable): callable used to aggregate (time, lat, lon, lev, -1) -> (time, lat, lo)

    Returns:
        type: Description of returned object.

    """
    scores_2d = compute_2d_aggregate_metrics(prediction_3d, targets_2d, aggregate_fn)
    scores_3d = compute_3d_isotropic_metrics(prediction_3d, groundtruth_3d)
    output = {'2d': scores_2d, '3d': scores_3d}
    return output


def compute_2d_aggregate_metrics(prediction_3d, targets_2d, aggregate_fn):
    """Computes prediction scores between aggregation of 3D+t prediction and
    2D+t aggregate targets used for training

    Args:
        prediction_3d (torch.Tensor): (time, lat, lon, lev)
        targets_2d (torch.Tensor): (time, lat, lon)
        aggregate_fn (callable): callable used to aggregate (time, lat, lon, lev, -1) -> (time, lat, lon)

    Returns:
        type: dict[float]

    """
    # Get prediction into shape that the aggregation function expects
    n_col = prediction_3d.size(0) * prediction_3d.size(1) * prediction_3d.size(2)
    prediction_3d = prediction_3d.reshape(n_col, -1)
    aggregate_prediction_2d = aggregate_fn(prediction_3d.unsqueeze(-1)).squeeze().flatten()

    difference = aggregate_prediction_2d.sub(targets_2d.flatten())
    rmse = torch.square(difference).mean().sqrt()
    mae = torch.abs(difference).mean()
    corr = spearman_correlation(aggregate_prediction_2d.flatten(), targets_2d.flatten())
    output = {'rmse': rmse.item(), 'mae': mae.item(), 'corr': corr}
    return output


def compute_3d_isotropic_metrics(prediction_3d, groundtruth_3d):
    """Computes prediction scores between 3D+t prediction and 3D+t unobserved groundtruth.
    Metrics are averaged isotropically accross all dimensions.

    Args:
        prediction_3d (torch.Tensor): (time, lat, lon, lev)
        groundtruth_3d (torch.Tensor): (time, lat, lon, lev)

    Returns:
        type: dict[float]

    """
    difference = prediction_3d.sub(groundtruth_3d)
    rmse = torch.square(difference).mean().sqrt()
    mae = torch.abs(difference).mean()
    corr = spearman_correlation(prediction_3d.flatten(), groundtruth_3d.flatten())
    output = {'rmse': rmse.item(), 'mae': mae.item(), 'corr': corr}
    return output


def compute_3d_vertical_metrics(prediction_3d, groundtruth_3d):
    """Computes prediction scores between 3D+t prediction and 3D+t unobserved groundtruth.
    Metrics are computed for vertical profiles and then averaged over time, lat, lon.

    Args:
        prediction_3d (torch.Tensor): (time, lat, lon, lev)
        groundtruth_3d (torch.Tensor): (time, lat, lon, lev)

    Returns:
        type: dict[float]

    """
    # Reshape tensors as (n_columns, lev)
    n_col = prediction_3d.size(0) * prediction_3d.size(1) * prediction_3d.size(2)
    prediction_3d = prediction_3d.reshape(n_col, -1)
    groundtruth_3d = groundtruth_3d.reshape(n_col, -1)
    difference = prediction_3d.sub(groundtruth_3d)

    # Compute metrics columnwise and average out
    rmse = torch.square(difference).mean(dim=-1).sqrt().mean()
    mae = torch.abs(difference).mean(dim=-1).mean()
    corr = np.mean([spearman_correlation(pred, gt) for (pred, gt) in zip(prediction_3d, groundtruth_3d)])

    # Return metrics dictionnary
    output = {'rmse': rmse.item(), 'mae': mae.item(), 'corr': corr}
    return output


def spearman_correlation(x, y):
    """Computes Spearman Correlation between x and y

    Args:
        x (torch.Tensor)
        y (torch.Tensor)

    Returns:
        type: torch.Tensor

    """
    x_std = (x - x.mean()) / x.std()
    y_std = (y - y.mean()) / y.std()
    corr = float(pearsonr(x_std.numpy(), y_std.numpy())[0])
    return corr
