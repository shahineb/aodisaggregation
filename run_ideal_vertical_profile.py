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
    prediction_3d_dist, bext_dist = predict(data=data, cfg=cfg)

    # Run evaluation
    evaluate(prediction_3d_dist=prediction_3d_dist, bext_dist=bext_dist, data=data, cfg=cfg, plot=args['--plot'], output_dir=args['--o'])


def migrate_to_device(data, device):
    # These are the only tensors needed on device to run this experiment
    data = data._replace(x_std=data.x_std.to(device),
                         x_by_column_std=data.x_by_column_std.to(device),
                         z=data.z.to(device),
                         z_smooth=data.z_smooth.to(device),
                         h_by_column_std=data.h_by_column_std.to(device))

    return data


def predict(data, cfg):
    # Predict idealized exponential height profile
    L = cfg['model']['L']
    h_stddev = data.h_by_column.std().to(device)
    prediction_3d = torch.exp(-data.h_by_column_std / L)

    # Rescale predictions by τ/∫φdh
    aggregate_prediction = h_stddev * L * (torch.exp(-data.h_by_column_std[:, -1] / L) - torch.exp(-data.h_by_column_std[:, 0] / L))
    correction = data.z_smooth / aggregate_prediction
    prediction_3d.mul_(correction.unsqueeze(-1))

    # Make latent vertical profile dummy distribution
    noise = cfg['evaluation']['noise']
    prediction_3d_dist = torch.distributions.Normal(prediction_3d, noise)

    # Make bext observation model distribution
    sigma_ext = torch.tensor(cfg['evaluation']['sigma_ext'])
    loc = torch.log(prediction_3d.clip(min=torch.finfo(torch.float64).eps)) - sigma_ext.square().div(2)
    bext_dist = torch.distributions.LogNormal(loc, sigma_ext)
    return prediction_3d_dist, bext_dist


def evaluate(prediction_3d_dist, bext_dist, data, cfg, plot, output_dir):
    # Define aggregation wrt non-standardized height for evaluation
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column.unsqueeze(-1).cpu(), dim=-2)
        return aggregated_grid

    # Dump plots in output dir
    if plot:
        dump_plots(cfg=cfg,
                   dataset=data.dataset,
                   prediction_3d_dist=prediction_3d_dist,
                   bext_dist=bext_dist,
                   sigma=torch.tensor(1.),
                   aggregate_fn=trpz,
                   ideal=True,
                   output_dir=output_dir)
        logging.info("Dumped plots")

    # Dump scores in output dir
    dump_scores(cfg=cfg,
                prediction_3d_dist=prediction_3d_dist,
                bext_dist=bext_dist,
                groundtruth_3d=data.gt_by_column,
                targets_2d=data.z,
                aggregate_fn=trpz,
                ideal=True,
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
