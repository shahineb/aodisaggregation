import torch
import numpy as np
from scipy.stats import pearsonr


def compute_scores(prediction_3d, groundtruth_3d, targets_2d, aggregate_fn):
    """Computes prediction scores

    Args:
        prediction_3d (torch.Tensor): (time * lat * lon, lev)
        groundtruth_3d (torch.Tensor): (time * lat * lon, lev)
        targets_2d (torch.Tensor): (time * lat * lon)
        aggregate_fn (callable): callable used to aggregate (time * lat * lon, lev, -1) -> (time * lat * lon, -1)

    Returns:
        type: Description of returned object.

    """
    # Compute metrics over raw predictions
    scores_2d = compute_2d_aggregate_metrics(prediction_3d, targets_2d, aggregate_fn)
    scores_3d_isotropic = compute_3d_isotropic_metrics(prediction_3d, groundtruth_3d)
    scores_3d_vertical = compute_3d_vertical_metrics(prediction_3d, groundtruth_3d)

    # Compute metrics over standardized predictions
    sigma_2d = targets_2d.std()
    sigma_3d = groundtruth_3d.std()
    std_scores_2d = compute_2d_aggregate_metrics(prediction_3d.div(sigma_2d), targets_2d.div(sigma_2d), aggregate_fn)
    std_scores_3d_isotropic = compute_3d_isotropic_metrics(prediction_3d.div(sigma_3d), groundtruth_3d.div(sigma_3d))
    std_scores_3d_vertical = compute_3d_vertical_metrics(prediction_3d.div(sigma_3d), groundtruth_3d.div(sigma_3d))

    # Encapsulate scores into output dictionnary
    output = {'raw': {'2d': scores_2d,
                      '3d': {'isotropic': scores_3d_isotropic,
                             'vertical': scores_3d_vertical}
                      },

              'std': {'2d': std_scores_2d,
                      '3d': {'isotropic': std_scores_3d_isotropic,
                             'vertical': std_scores_3d_vertical}
                      }
              }
    return output


def compute_2d_aggregate_metrics(prediction_3d, targets_2d, aggregate_fn):
    """Computes prediction scores between aggregation of 3D+t prediction and
    2D+t aggregate targets used for training

    Args:
        prediction_3d (torch.Tensor): (time * lat * lon, lev)
        targets_2d (torch.Tensor): (time * lat * lon)
        aggregate_fn (callable): callable used to aggregate (time * lat * lon, lev, -1) -> (time * lat * lon, -1)

    Returns:
        type: dict[float]

    """
    # Aggregate prediction along height
    aggregate_prediction_2d = aggregate_fn(prediction_3d.unsqueeze(-1)).squeeze()

    # Compute raw distances metrics
    difference = aggregate_prediction_2d.sub(targets_2d)
    rmse = torch.square(difference).mean().sqrt()
    mae = torch.abs(difference).mean()

    # Compute normalized distances metrics
    q25, q75 = torch.quantile(targets_2d, q=torch.tensor([0.25, 0.75]))
    nrmse = rmse.div(q75 - q25)
    nmae = mae.div(q75 - q25)

    # Compute spearman correlation
    corr = spearman_correlation(aggregate_prediction_2d, targets_2d)

    # Encapsulate results in output dictionnary
    output = {'rmse': rmse.item(),
              'mae': mae.item(),
              'nrmse': nrmse.item(),
              'nmae': nmae.item(),
              'corr': corr}
    return output


def compute_3d_isotropic_metrics(prediction_3d, groundtruth_3d):
    """Computes prediction scores between 3D+t prediction and 3D+t unobserved groundtruth.
    Metrics are averaged isotropically accross all dimensions.

    Args:
        prediction_3d (torch.Tensor): (time * lat * lon, lev)
        groundtruth_3d (torch.Tensor): (time * lat * lon, lev)

    Returns:
        type: dict[float]

    """
    # Compute raw distances metrics
    difference = prediction_3d.sub(groundtruth_3d)
    rmse = torch.square(difference).mean().sqrt()
    mae = torch.abs(difference).mean()

    # Compute normalized distances metrics
    q25, q75 = torch.quantile(groundtruth_3d, q=torch.tensor([0.25, 0.75]))
    nrmse = rmse.div(q75 - q25)
    nmae = mae.div(q75 - q25)

    # Compute spearman correlation
    corr = spearman_correlation(prediction_3d.flatten(), groundtruth_3d.flatten())

    # Encapsulate results in output dictionnary
    output = {'rmse': rmse.item(),
              'mae': mae.item(),
              'nrmse': nrmse.item(),
              'nmae': nmae.item(),
              'corr': corr}
    return output


def compute_3d_vertical_metrics(prediction_3d, groundtruth_3d):
    """Computes prediction scores between 3D+t prediction and 3D+t unobserved groundtruth.
    Metrics are computed for vertical profiles and then averaged over time, lat, lon.

    Args:
        prediction_3d (torch.Tensor): (time * lat * lon, lev)
        groundtruth_3d (torch.Tensor): (time * lat * lon, lev)

    Returns:
        type: dict[float]

    """
    # Compute difference tensor
    difference = prediction_3d.sub(groundtruth_3d)

    # Compute metrics columnwise and average out
    rmse = torch.square(difference).mean(dim=-1).sqrt().mean()
    mae = torch.abs(difference).mean(dim=-1).mean()
    corr = np.mean([spearman_correlation(pred, gt) for (pred, gt) in zip(prediction_3d, groundtruth_3d)])

    # Compute normalized distances metrics
    q25, q75 = torch.quantile(groundtruth_3d, q=torch.tensor([0.25, 0.75]))
    nrmse = rmse.div(q75 - q25)
    nmae = mae.div(q75 - q25)

    # Encapsulate results in output dictionnary
    output = {'rmse': rmse.item(),
              'mae': mae.item(),
              'nrmse': nrmse.item(),
              'nmae': nmae.item(),
              'corr': float(corr)}
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
