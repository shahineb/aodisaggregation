"""
Description : Runs aggregate ridge regression experiment

Usage: run_ridge_regression.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --device=<device_index>          Device to use [default: cpu]
  --plot                           Outputs plots.
"""
import os
import yaml
import logging
from docopt import docopt
import torch
from src.preprocessing import make_data
from src.models import AggregateRidgeRegression
from src.evaluation import dump_scores, dump_plots, dump_state_dict


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    data = make_data(cfg=cfg, include_2d=False)

    # Move needed tensors only to device
    data = migrate_to_device(data=data)

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


def migrate_to_device(data):
    # These are the only tensors needed on device to run this experiment
    data = data._replace(x_by_column_std=data.x_by_column_std.to(device),
                         x_std=data.x_std.to(device),
                         z_std=data.z_std.to(device),
                         h_by_column_std=data.h_by_column_std.to(device))
    return data


def make_model(cfg, data):
    # Create aggregation operator over standardized heights
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column_std.unsqueeze(-1), dim=-2)
        return aggregated_grid

    # Instantiate model
    model = AggregateRidgeRegression(lbda=cfg['model']['lbda'],
                                     aggregate_fn=trpz,
                                     fit_intercept=cfg['model']['fit_intercept'])
    return model.to(device)


def fit(model, data):
    model.fit(data.x_by_column_std, data.z_std)
    return model


def predict(model, data):
    with torch.no_grad():
        # Predict on standardized 3D covariates
        prediction = model(data.x_std)

        # Reshape as (time * lat * lon, lev) grid
        prediction_3d_std = prediction.reshape(*data.h_by_column.shape)

        # Unnormalize with mean and variance of observed aggregte targets – because groundtruth 3D field is unobserved
        mean_z, sigma_z, sigma_h = data.z.mean().to(device), data.z.std().to(device), data.h_by_column.std().to(device)
        prediction_3d = sigma_z * (prediction_3d_std + mean_z) / sigma_h
    return prediction_3d


def evaluate(prediction_3d, data, model, cfg, plot, output_dir):
    # Define aggregation wrt non-standardized height for evaluation
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column.unsqueeze(-1), dim=-2)
        return aggregated_grid

    # Dump scores in output dir
    dump_scores(prediction_3d=prediction_3d.cpu(),
                groundtruth_3d=data.gt_by_column,
                targets_2d=data.z,
                aggregate_fn=trpz,
                output_dir=output_dir)

    # Dump plots in output dir
    if plot:
        dump_plots(cfg=cfg,
                   dataset=data.dataset,
                   prediction_3d=prediction_3d.cpu(),
                   aggregate_fn=trpz,
                   output_dir=output_dir)
        logging.info("Dumped plots")

    # Dump model weights in output dir
    dump_state_dict(model=model, output_dir=output_dir)
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

    # Setup global variable for device
    if torch.cuda.is_available() and args['--device'].isdigit():
        device = torch.device(f"cuda:{args['--device']}")
    else:
        device = torch.device('cpu')

    # Run session
    main(args, cfg)
