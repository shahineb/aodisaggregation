"""
Description : Runs aggregate kernel ridge regression experiment

Usage: run_kernel_ridge_regression.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --plot                           Outputs plots.
"""
import os
import yaml
import logging
from docopt import docopt
import torch
from src.preprocessing import make_data
from src.kernels import RFFKernel
from src.models import AggregateKernelRidgeRegression
from src.evaluation import dump_scores, dump_plots, dump_model


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    data = make_data(cfg=cfg, include_2d=False)

    # Instantiate model
    model = make_model(cfg=cfg, data=data)
    logging.info(f"{model}")

    # Fit model
    model = fit(model=model, data=data)
    logging.info("Fitted model")

    # Run prediction
    prediction_3d = predict(model=model, data=data)

    # Run evaluation
    evaluate(prediction_3d=prediction_3d, data=data, model=model, cfg=cfg, plot=args['--plot'], output_dir=args['--o'])


def make_model(cfg, data):
    # Create aggregation operator over standardized heights
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h_std.unsqueeze(-1), dim=-2)
        return aggregated_grid

    # Initialize RFF kernel
    ard_num_dims = len(cfg['dataset']['3d_covariates']) + 4
    kernel = RFFKernel(nu=cfg['model']['nu'],
                       num_samples=cfg['model']['num_samples'],
                       ard_num_dims=ard_num_dims)
    kernel.lengthscale = cfg['model']['lengthscale'] * torch.ones(ard_num_dims)

    # Instantiate model
    model = AggregateKernelRidgeRegression(kernel=kernel,
                                           lbda=cfg['model']['lbda'],
                                           aggregate_fn=trpz)
    return model


def fit(model, data):
    model.fit(data.x_by_column_std, data.z_std)
    return model


def predict(model, data):
    with torch.no_grad():
        # Predict on standardized 3D covariates
        prediction = model(data.x_std)

        # Reshape as (time, lat, lon, lev) grid
        prediction_3d_std = prediction.reshape(*data.gt_grid.shape)

        # Unnormalize with mean and variance of observed aggregte targets – groundtruth 3D field is unobserved
        prediction_3d = data.z_grid.std() * (prediction_3d_std + data.z_grid.mean()) / data.h.std()
    return prediction_3d


def evaluate(prediction_3d, data, model, cfg, plot, output_dir):
    # Define aggregation wrt non-standardized height for evaluation
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h.unsqueeze(-1), dim=-2)
        return aggregated_grid

    # Dump scores in output dir
    dump_scores(prediction_3d=prediction_3d,
                groundtruth_3d=data.gt_grid,
                targets_2d=data.z_grid,
                aggregate_fn=trpz,
                output_dir=output_dir)

    # Dump plots in output dir
    if plot:
        dump_plots(cfg=cfg,
                   dataset=data.dataset,
                   prediction_3d=prediction_3d,
                   aggregate_fn=trpz,
                   output_dir=output_dir)
        logging.info("Dumped plots")

    # Dump model weights in output dir
    dump_model(model=model, output_dir=output_dir)
    logging.info("Dumped weights")


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
