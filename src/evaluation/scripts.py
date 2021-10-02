import os
import yaml
import torch
import matplotlib.pyplot as plt
from src.evaluation import metrics
from src.evaluation import visualization


def dump_scores(prediction_3d, groundtruth_3d, targets_2d, aggregate_fn, output_dir):
    scores = metrics.compute_scores(prediction_3d, groundtruth_3d, targets_2d, aggregate_fn)
    dump_path = os.path.join(output_dir, 'scores.metrics')
    with open(dump_path, 'w') as f:
        yaml.dump(scores, f)


def dump_plots(cfg, dataset, prediction_3d, aggregate_fn, output_dir):
    # Reshape prediction as (time, lat, lon, lev) grids for visualization
    prediction_3d_grid = prediction_3d.view(len(dataset.time), len(dataset.lat), len(dataset.lon), -1)

    # First plot - aggregate 2D prediction
    dump_path = os.path.join(output_dir, 'aggregated_2d_prediction.png')
    _ = visualization.plot_aggregate_2d_predictions(dataset=dataset,
                                                    target_key=cfg['dataset']['target'],
                                                    prediction_3d_grid=prediction_3d_grid,
                                                    aggregate_fn=aggregate_fn)
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
                                                     prediction_3d_grid=prediction_3d_grid)
    plt.savefig(dump_path)
    plt.close()


def dump_model_parameters(model, output_dir):
    dump_path = os.path.join(output_dir, 'named_params.pt')
    torch.save(dict(model.named_parameters()), dump_path)


def dump_state_dict(model, output_dir):
    dump_path = os.path.join(output_dir, 'state_dict.pt')
    torch.save(model.state_dict(), dump_path)
