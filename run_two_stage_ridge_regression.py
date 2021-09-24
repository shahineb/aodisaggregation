"""
Description : Runs two-stage ridge regression experiment

Usage: run_two_staged_ridge_experiment.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --plot                           Outputs scatter plots.
"""
import os
import yaml
import logging
from docopt import docopt
import torch
import matplotlib.pyplot as plt
import src.preprocessing as preproc
from src.models import TwoStageAggregateRidgeRegression
from src.evaluation import metrics, visualization


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    dataset, standard_dataset, x_by_column_std, x_std, y_grid_std, y_std, z_grid, z_std, gt_grid, h, h_std = make_datasets(cfg=cfg)

    # Instantiate model
    model = make_model(cfg=cfg, h=h_std)
    logging.info(f"{model}")

    # Fit model
    model.fit(x_by_column_std, y_std, z_std)
    logging.info("Fitted model")

    # Run prediction
    with torch.no_grad():
        prediction = model(x_std)
        prediction_3d_std = prediction.reshape(*gt_grid.shape)
        prediction_3d = z_grid.std() * (prediction_3d_std + z_grid.mean()) / h.std()

    # Define aggregation wrt non-standardized height for evaluation
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=h.unsqueeze(-1), dim=-2)
        return aggregated_grid

    # Dump scores in output dir
    dump_scores(prediction_3d=prediction_3d,
                groundtruth_3d=gt_grid,
                targets_2d=z_grid,
                aggregate_fn=trpz,
                output_dir=args['--o'])

    # Dump plots in output dir
    if args['--plot']:
        dump_plots(cfg=cfg,
                   dataset=dataset,
                   standard_dataset=standard_dataset,
                   prediction_3d=prediction_3d,
                   aggregate_fn=trpz,
                   output_dir=args['--o'])
        logging.info("Dumped plots")


def make_datasets(cfg):
    # Load dataset
    dataset = preproc.load_dataset(file_path=cfg['dataset']['path'])

    # Compute standardized versions
    standard_dataset = preproc.standardize(dataset)

    # Convert into pytorch tensors
    x_grid_std = preproc.make_3d_covariates_tensors(dataset=standard_dataset, variables_keys=cfg['dataset']['3d_covariates'])
    y_grid_std = preproc.make_2d_covariates_tensors(dataset=standard_dataset, variables_keys=cfg['dataset']['2d_covariates'])
    z_grid_std = preproc.make_2d_tensor(dataset=standard_dataset, variable_key=cfg['dataset']['target'])
    h_grid_std = preproc.make_3d_tensor(dataset=standard_dataset, variable_key='height')
    z_grid = preproc.make_2d_tensor(dataset=dataset, variable_key=cfg['dataset']['target'])
    gt_grid = preproc.make_3d_tensor(dataset=dataset, variable_key=cfg['dataset']['groundtruth'])
    h_grid = preproc.make_3d_tensor(dataset=dataset, variable_key='height')

    # Reshape tensors
    x_by_column_std = x_grid_std.reshape(-1, x_grid_std.size(-2), x_grid_std.size(-1))
    x_std = x_by_column_std.reshape(-1, x_grid_std.size(-1))
    y_std = y_grid_std.reshape(-1, y_grid_std.size(-1))
    z_std = z_grid_std.flatten()
    h = h_grid.reshape(-1, x_grid_std.size(-2))
    h_std = h_grid_std.reshape(-1, x_grid_std.size(-2))
    return dataset, standard_dataset, x_by_column_std, x_std, y_grid_std, y_std, z_grid, z_std, gt_grid, h, h_std


def make_model(cfg, h):
    # Create aggregation operator
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=h.unsqueeze(-1), dim=-2)
        return aggregated_grid

    # Instantiate model
    model = TwoStageAggregateRidgeRegression(lbda_2d=cfg['model']['lbda_2d'],
                                             lbda_3d=cfg['model']['lbda_3d'],
                                             aggregate_fn=trpz,
                                             fit_intercept_2d=cfg['model']['fit_intercept_2d'],
                                             fit_intercept_3d=cfg['model']['fit_intercept_3d'])
    return model


def dump_scores(prediction_3d, groundtruth_3d, targets_2d, aggregate_fn, output_dir):
    scores = metrics.compute_scores(prediction_3d, groundtruth_3d, targets_2d, aggregate_fn)
    dump_path = os.path.join(output_dir, 'scores.metrics')
    with open(dump_path, 'w') as f:
        yaml.dump(scores, f)
    logging.info(f"Dumped scores at {dump_path}")


def dump_plots(cfg, dataset, standard_dataset, prediction_3d, aggregate_fn, output_dir):
    # First plot - aggregate 2D prediction
    dump_path = os.path.join(output_dir, 'aggregated_2d_prediction.png')
    _ = visualization.plot_aggregate_2d_predictions(dataset=dataset,
                                                    target_key=cfg['dataset']['target'],
                                                    prediction_3d=prediction_3d,
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
                                                     prediction_3d=prediction_3d)
    plt.savefig(dump_path)
    plt.close()


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load config file
    with open(args['--cfg'], "r") as f:
        cfg = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Arguments: {args}\n')
    logging.info(f'Configuration file: {cfg}\n')

    # Create output directory if doesn't exists
    os.makedirs(args['--o'], exist_ok=True)
    with open(os.path.join(args['--o'], 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # Run session
    main(args, cfg)
