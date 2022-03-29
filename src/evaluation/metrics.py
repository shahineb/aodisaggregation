import torch
import numpy as np
from scipy.stats import pearsonr


def compute_scores(prediction_3d_dist, bext_dist, groundtruth_3d, targets_2d, aggregate_fn, calibration_seed, n_test_samples, ideal):
    """Computes prediction scores

    Args:
        prediction_3d_dist (torch.distributions.Distribution): (time * lat * lon, lev)
        bext_dist (torch.distributions.Distribution): (n_samples, time * lat * lon, lev)
        groundtruth_3d (torch.Tensor): (time * lat * lon, lev)
        targets_2d (torch.Tensor): (time * lat * lon)
        aggregate_fn (callable): callable used to aggregate (time * lat * lon, lev, -1) -> (time * lat * lon, -1)
        calibration_seed (int): seed to use to choose profiles for calibration (here we want to exclude them)

    Returns:
        type: Description of returned object.

    """
    # Extract posterior mean prediction
    prediction_3d = prediction_3d_dist.mean.cpu()

    # Compute metrics over raw predictions
    scores_2d = compute_2d_aggregate_metrics(prediction_3d, targets_2d, aggregate_fn)
    scores_3d_isotropic = compute_3d_isotropic_metrics(prediction_3d, groundtruth_3d)
    scores_3d_vertical = compute_3d_vertical_metrics(prediction_3d, groundtruth_3d)
    scores_3d_probabilistic = compute_3d_probabilistic_metrics(bext_dist, groundtruth_3d, calibration_seed, n_test_samples, ideal)

    # Compute metrics over standardized predictions
    sigma_2d = targets_2d.std()
    sigma_3d = groundtruth_3d.std()
    std_scores_2d = compute_2d_aggregate_metrics(prediction_3d.div(sigma_2d), targets_2d.div(sigma_2d), aggregate_fn)
    std_scores_3d_isotropic = compute_3d_isotropic_metrics(prediction_3d.div(sigma_3d), groundtruth_3d.div(sigma_3d))
    std_scores_3d_vertical = compute_3d_vertical_metrics(prediction_3d.div(sigma_3d), groundtruth_3d.div(sigma_3d))

    # Encapsulate scores into output dictionnary
    output = {'raw': {'2d': scores_2d,
                      '3d': {'isotropic': scores_3d_isotropic,
                             'vertical': scores_3d_vertical,
                             'probabilistic': scores_3d_probabilistic}
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
    mean_bias = difference.mean()
    rmse = torch.square(difference).mean().sqrt()
    mae = torch.abs(difference).mean()

    # Compute normalized distances metrics
    q25, q75 = torch.quantile(targets_2d, q=torch.tensor([0.25, 0.75]))
    nrmse = rmse.div(q75 - q25)
    nmae = mae.div(q75 - q25)

    # Compute spearman correlation
    corr = spearman_correlation(aggregate_prediction_2d, targets_2d)

    # Compute quantiles bias
    bias50 = torch.quantile(aggregate_prediction_2d, q=0.50) - torch.quantile(targets_2d, q=0.50)
    bias90 = torch.quantile(aggregate_prediction_2d, q=0.90) - torch.quantile(targets_2d, q=0.90)

    # Encapsulate results in output dictionnary
    output = {'mb': mean_bias.item(),
              'rmse': rmse.item(),
              'mae': mae.item(),
              'nrmse': nrmse.item(),
              'nmae': nmae.item(),
              'corr': corr,
              'bias50': bias50.item(),
              'bias90': bias90.item()}
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
    mean_bias = difference.mean()
    rmse = torch.square(difference).mean().sqrt()
    mae = torch.abs(difference).mean()

    # Compute normalized distances metrics
    q25, q75 = torch.quantile(groundtruth_3d, q=torch.tensor([0.25, 0.75]))
    nrmse = rmse.div(q75 - q25)
    nmae = mae.div(q75 - q25)

    # Compute spearman correlation
    corr = spearman_correlation(prediction_3d.flatten(), groundtruth_3d.flatten())

    # Compute quantiles bias
    bias50 = torch.quantile(prediction_3d, q=0.50) - torch.quantile(groundtruth_3d, q=0.50)
    bias90 = torch.quantile(prediction_3d, q=0.90) - torch.quantile(groundtruth_3d, q=0.90)

    # Encapsulate results in output dictionnary
    output = {'mb': mean_bias.item(),
              'rmse': rmse.item(),
              'mae': mae.item(),
              'nrmse': nrmse.item(),
              'nmae': nmae.item(),
              'corr': corr,
              'bias50': bias50.item(),
              'bias90': bias90.item()}
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
    mean_bias = difference.mean(dim=-1).mean()
    rmse = torch.square(difference).mean(dim=-1).sqrt().mean()
    mae = torch.abs(difference).mean(dim=-1).mean()
    corr = np.mean([spearman_correlation(pred, gt) for (pred, gt) in zip(prediction_3d, groundtruth_3d)])
    bias50 = torch.mean(torch.quantile(prediction_3d, q=0.50, dim=-1) - torch.quantile(groundtruth_3d, q=0.50), dim=-1)
    bias90 = torch.mean(torch.quantile(prediction_3d, q=0.90, dim=-1) - torch.quantile(groundtruth_3d, q=0.90), dim=-1)

    # Compute normalized distances metrics
    q25, q75 = torch.quantile(groundtruth_3d, q=torch.tensor([0.25, 0.75]))
    nrmse = rmse.div(q75 - q25)
    nmae = mae.div(q75 - q25)

    # Encapsulate results in output dictionnary
    output = {'mb': mean_bias.item(),
              'rmse': rmse.item(),
              'mae': mae.item(),
              'nrmse': nrmse.item(),
              'nmae': nmae.item(),
              'corr': float(corr),
              'bias50': bias50.item(),
              'bias90': bias90.item()}
    return output


def compute_3d_probabilistic_metrics(bext_dist, groundtruth_3d, calibration_seed, n_test_samples, ideal):
    """Computes prediction scores between 3D+t prediction distribution and 3D+t unobserved groundtruth.
    Metrics are computed for vertical profiles across all dimensions.

    Args:
        bext_dist (torch.distributions.Distribution): (n_samples, time * lat * lon, lev)
        groundtruth_3d (torch.Tensor): (time * lat * lon, lev)
        calibration_seed (int): seed to use to choose profiles for calibration (here we want to exclude them)
        n_test_samples (int): number of columns to use to speed up probabilistic metrics computation
        ideal (bool): if True, computes for idealized profile experiment (more efficient)

    Returns:
        type: dict[float]

    """
    # Draw indices of profiles used for calibration
    torch.random.manual_seed(calibration_seed)
    rdm_idx = torch.randperm(len(groundtruth_3d))
    _, test_idx = rdm_idx[:200], rdm_idx[200:200 + n_test_samples]
    if ideal:
        sub_bext_loc = bext_dist.loc[test_idx]
        sub_bext_scale = bext_dist.scale[test_idx]
    else:
        sub_bext_loc = bext_dist.loc[:, test_idx]
        sub_bext_scale = bext_dist.scale[:, test_idx]
    groundtruth_3d = groundtruth_3d[test_idx]
    bext_dist = torch.distributions.LogNormal(sub_bext_loc, sub_bext_scale)

    # Compute average ELBO of groundtruth under predicted posterior distribution
    eps = torch.finfo(torch.float64).eps
    gt = groundtruth_3d.clip(min=eps)
    if ideal:
        # Actually reaches upper bound for ideal model, i.e. ll = elbo
        elbo = bext_dist.log_prob(gt).mean()
    else:
        log_prob = bext_dist.log_prob(gt.unsqueeze(0).tile((bext_dist.mean.size(0), 1, 1)))
        elbo = log_prob.mean(dim=0).mean()

    # Compute 95% calibration score
    if ideal:
        lb, ub = bext_dist.icdf(torch.tensor(0.025)), bext_dist.icdf(torch.tensor(0.975))
    else:
        bext = bext_dist.sample()
        lb, ub = torch.quantile(bext, q=torch.tensor([0.025, 0.975]), dim=0).cpu()
    mask = (groundtruth_3d >= lb) & (groundtruth_3d <= ub)
    calib95 = mask.float().mean()

    # Compute integral calibration index
    cr_sizes = np.arange(0.05, 1.0, 0.05)
    calibs = []
    for size in cr_sizes:
        q_lb = (1 - float(size)) / 2
        q_ub = 1 - q_lb
        if ideal:
            lb, ub = bext_dist.icdf(torch.tensor(q_lb)), bext_dist.icdf(torch.tensor(q_ub))
        else:
            lb, ub = torch.quantile(bext, q=torch.tensor([q_lb, q_ub]), dim=0).cpu()
        mask = (groundtruth_3d >= lb) & (groundtruth_3d <= ub)
        calibs.append(mask.float().mean().item())
    ICI = np.abs(np.asarray(calibs) - cr_sizes).mean()

    # Encapsulate results in output dictionnary
    output = {'elbo': elbo.item(),
              'calib95': calib95.item(),
              'ICI': ICI.item()}
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
