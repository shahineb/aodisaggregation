"""
Description : Runs idealized exponential aerosol vertical profile reconstruction

Usage: run_ideal_vertical_profile.py  [options] --cfg=<path_to_config> --o=<output_dir>

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
from src.evaluation import dump_scores, dump_plots


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    data = make_data(cfg=cfg, include_2d=False)

    # Move needed tensors only to device
    data = migrate_to_device(data=data, device=device)

    # Run prediction
    prediction_3d_dist = predict(data=data, cfg=cfg)

    # Run evaluation
    evaluate(prediction_3d_dist=prediction_3d_dist, data=data, cfg=cfg, plot=args['--plot'], output_dir=args['--o'])


def migrate_to_device(data, device):
    # These are the only tensors needed on device to run this experiment
    data = data._replace(x_std=data.x_std.to(device),
                         x_by_column_std=data.x_by_column_std.to(device),
                         z=data.z.to(device),
                         h_by_column_std=data.h_by_column_std.to(device))

    return data


def predict(data, cfg):
    # Predict idealized exponential height profile
    lbda = cfg['model']['lbda']
    prediction_3d = 0.000009 * torch.exp(-lbda * data.h_by_column_std)

    # Make distribution
    noise = cfg['evaluation']['noise']
    prediction_3d_dist = torch.distributions.Normal(prediction_3d, noise)
    return prediction_3d_dist


def evaluate(prediction_3d_dist, data, cfg, plot, output_dir):
    # Define aggregation wrt non-standardized height for evaluation
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column.unsqueeze(-1).cpu(), dim=-2)
        return aggregated_grid

    # Dump plots in output dir
    if plot:
        dump_plots(cfg=cfg,
                   dataset=data.dataset,
                   prediction_3d_dist=prediction_3d_dist,
                   aggregate_fn=trpz,
                   output_dir=output_dir)
        logging.info("Dumped plots")

    # Dump scores in output dir
    dump_scores(prediction_3d=prediction_3d_dist.mean.cpu(),
                groundtruth_3d=data.gt_by_column,
                targets_2d=data.z,
                aggregate_fn=trpz,
                output_dir=output_dir)
    logging.info("Dumped scores")


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
