import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.evaluation import metrics
from src.evaluation import visualization


def dump_scores(cfg, prediction_3d_dist, bext_dist, groundtruth_3d, targets_2d, aggregate_fn, output_dir, ideal=False):
    scores = metrics.compute_scores(prediction_3d_dist=prediction_3d_dist,
                                    bext_dist=bext_dist,
                                    groundtruth_3d=groundtruth_3d,
                                    targets_2d=targets_2d,
                                    aggregate_fn=aggregate_fn,
                                    calibration_seed=cfg['evaluation']['calibration_seed'],
                                    n_test_samples=cfg['evaluation']['n_test_samples'],
                                    ideal=ideal)
    dump_path = os.path.join(output_dir, 'scores.metrics')
    with open(dump_path, 'w') as f:
        yaml.dump(scores, f)


def dump_plots(cfg, dataset, prediction_3d_dist, bext_dist, sigma, aggregate_fn, output_dir, ideal=False):
    # Reshape prediction as (time, lat, lon, lev) grids for visualization
    prediction_3d_grid = prediction_3d_dist.mean.view(len(dataset.time), len(dataset.lat), len(dataset.lon), -1).cpu()
    prediction_3d_grid_q025 = prediction_3d_dist.icdf(torch.tensor(0.025)).view(len(dataset.time), len(dataset.lat), len(dataset.lon), -1).cpu()
    prediction_3d_grid_q975 = prediction_3d_dist.icdf(torch.tensor(0.975)).view(len(dataset.time), len(dataset.lat), len(dataset.lon), -1).cpu()
    if ideal:
        bext_q025 = bext_dist.icdf(torch.tensor(0.025)).view(len(dataset.time), len(dataset.lat), len(dataset.lon), -1).cpu()
        bext_q975 = bext_dist.icdf(torch.tensor(0.975)).view(len(dataset.time), len(dataset.lat), len(dataset.lon), -1).cpu()
    else:
        bext = bext_dist.sample().view(-1, len(dataset.time), len(dataset.lat), len(dataset.lon), len(dataset.lev)).cpu()
        bext_q025, bext_q975 = torch.quantile(bext, q=torch.tensor([0.025, 0.975]), dim=0).cpu()

    # Compute aggregated prediction shaped as (time, lat, lon) grids for visualization
    n_columns = prediction_3d_grid.size(0) * prediction_3d_grid.size(1) * prediction_3d_grid.size(2)
    prediction_3d = prediction_3d_grid.reshape(n_columns, -1)
    aggregate_prediction_2d = aggregate_fn(prediction_3d.unsqueeze(-1)).squeeze()
    aggregate_prediction_2d = aggregate_prediction_2d.reshape(prediction_3d_grid.size(0),
                                                              prediction_3d_grid.size(1),
                                                              prediction_3d_grid.size(2))
    loc = torch.log(aggregate_prediction_2d.clip(min=torch.finfo(torch.float64).eps)) - sigma.square().div(2)
    aggregate_prediction_2d_dist = torch.distributions.LogNormal(loc, sigma)
    aggregate_prediction_2d_q025 = aggregate_prediction_2d_dist.icdf(torch.tensor(0.025))
    aggregate_prediction_2d_q975 = aggregate_prediction_2d_dist.icdf(torch.tensor(0.975))

    # First plot - aggregate 2D prediction
    dump_path = os.path.join(output_dir, 'aggregated_2d_prediction.png')
    _ = visualization.plot_aggregate_2d_predictions(dataset=dataset,
                                                    target_key=cfg['dataset']['target'],
                                                    aggregate_prediction_2d=aggregate_prediction_2d,
                                                    aggregate_prediction_2d_q025=aggregate_prediction_2d_q025,
                                                    aggregate_prediction_2d_q975=aggregate_prediction_2d_q975)
    plt.savefig(dump_path)
    plt.close()

    # Second plot - slices of covariates
    dump_path = os.path.join(output_dir, 'covariates_slices.png')
    _ = visualization.plot_3d_covariates_slices(dataset=dataset,
                                                lat_idx=cfg['evaluation']['slice_latitude_idx'],
                                                time_idx=cfg['evaluation']['slice_time_idx'],
                                                covariates_keys=cfg['evaluation']['slices_covariates'])
    plt.savefig(dump_path)
    plt.close()

    # Third plot - prediction slice
    dump_path = os.path.join(output_dir, '3d_prediction_slice.png')
    _ = visualization.plot_vertical_prediction_slice(dataset=dataset,
                                                     lat_idx=cfg['evaluation']['slice_latitude_idx'],
                                                     time_idx=cfg['evaluation']['slice_time_idx'],
                                                     groundtruth_key=cfg['dataset']['groundtruth'],
                                                     prediction_3d_grid=prediction_3d_grid,
                                                     prediction_3d_grid_q025=prediction_3d_grid_q025,
                                                     prediction_3d_grid_q975=prediction_3d_grid_q975,
                                                     bext_q025=bext_q025,
                                                     bext_q975=bext_q975)
    plt.savefig(dump_path)
    plt.close()

    # Fourth - prediction profiles
    np.random.seed(cfg['evaluation']['seed'])
    lats = np.random.permutation(range(len(dataset.lat)))[:cfg['evaluation']['n_profiles']]
    lons = np.random.permutation(range(len(dataset.lon)))[:cfg['evaluation']['n_profiles']]
    dump_path = os.path.join(output_dir, '3d_prediction_profiles.png')
    _ = visualization.plot_vertical_prediction_profiles(dataset=dataset,
                                                        time_idx=cfg['evaluation']['profiles_time_idx'],
                                                        latlon_indices=list(zip(lats, lons)),
                                                        groundtruth_key=cfg['dataset']['groundtruth'],
                                                        prediction_3d_grid=prediction_3d_grid,
                                                        prediction_3d_grid_q025=prediction_3d_grid_q025,
                                                        prediction_3d_grid_q975=prediction_3d_grid_q975,
                                                        bext_q025=bext_q025,
                                                        bext_q975=bext_q975)
    plt.savefig(dump_path)
    plt.close()


def dump_model_parameters(model, output_dir):
    dump_path = os.path.join(output_dir, 'named_params.pt')
    torch.save(dict(model.named_parameters()), dump_path)


def dump_state_dict(model, output_dir):
    dump_path = os.path.join(output_dir, 'state_dict.pt')
    torch.save(model.state_dict(), dump_path)
