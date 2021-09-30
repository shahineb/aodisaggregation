"""
Description : Runs warped two-stage kernel ridge regression experiment

Usage: run_warped_two_stage_kernel_ridge_regression.py  [options] --cfg=<path_to_config> --o=<output_dir>

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
from progress.bar import Bar
import torch
from src.kernels import RFFKernel
from src.preprocessing import make_data
from src.models import WarpedTwoStageAggregateKernelRidgeRegression
from src.evaluation import dump_scores, dump_plots, dump_state_dict


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    data = make_data(cfg=cfg, include_2d=True)

    # Move needed tensors only to device
    data = migrate_to_device(data=data)

    # Instantiate model
    model = make_model(cfg=cfg, data=data)
    logging.info(f"{model}")

    # Fit model
    model = fit(model=model, data=data, cfg=cfg)
    logging.info("Fitted model")

    # Run prediction
    prediction_3d = predict(model=model, data=data)

    # Run evaluation
    evaluate(prediction_3d=prediction_3d, data=data, model=model, cfg=cfg, plot=args['--plot'], output_dir=args['--o'])


def migrate_to_device(data):
    # These are the only tensors needed on device to run this experiment
    data = data._replace(x_std=data.x_std.to(device),
                         y_std=data.y_std.to(device),
                         z=data.z.to(device),
                         h_by_column=data.h_by_column.to(device))
    return data


def make_model(cfg, data):
    # Create aggregation operator over standardized heights
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column.unsqueeze(-1), dim=-2)
        return aggregated_grid

    # Define warping transformation
    if cfg['model']['transform'] == 'linear':
        transform = lambda x: x
    elif cfg['model']['transform'] == 'softplus':
        transform = lambda x: torch.log(1 + torch.exp(x))
    elif cfg['model']['transform'] == 'smooth_abs':
        transform = lambda x: torch.nn.functional.smooth_l1_loss(x, torch.zeros_like(x), reduction='none')
    elif cfg['model']['transform'] == 'square':
        transform = torch.square
    elif cfg['model']['transform'] == 'exp':
        transform = torch.exp
    else:
        raise ValueError("Unknown transform")

    # Initialize RFF 2D covariates kernel
    ard_num_dims_2d = len(cfg['dataset']['2d_covariates']) + 3
    kernel_2d = RFFKernel(nu=cfg['model']['nu_2d'],
                          num_samples=cfg['model']['num_samples_2d'],
                          ard_num_dims=ard_num_dims_2d)
    kernel_2d.lengthscale = cfg['model']['lengthscale_2d'] * torch.ones(ard_num_dims_2d)
    kernel_2d.raw_lengthscale.requires_grad = False

    # Initialize RFF 3D covariates kernel
    ard_num_dims_3d = len(cfg['dataset']['3d_covariates']) + 4
    kernel_3d = RFFKernel(nu=cfg['model']['nu_3d'],
                          num_samples=cfg['model']['num_samples_3d'],
                          ard_num_dims=ard_num_dims_3d)
    kernel_3d.lengthscale = cfg['model']['lengthscale_3d'] * torch.ones(ard_num_dims_3d)
    kernel_3d.raw_lengthscale.requires_grad = False

    # Fix weights initialization seed
    torch.random.manual_seed(cfg['model']['seed'])

    # Instantiate model
    model = WarpedTwoStageAggregateKernelRidgeRegression(kernel_2d=kernel_2d.to(device),
                                                         kernel_3d=kernel_3d.to(device),
                                                         training_covariates_3d=data.x_std,
                                                         training_covariates_2d=data.y_std,
                                                         lbda_2d=cfg['model']['lbda_2d'],
                                                         lbda_3d=cfg['model']['lbda_3d'],
                                                         transform=transform,
                                                         aggregate_fn=trpz)
    return model.to(device)


def fit(model, data, cfg):
    # Define optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg['training']['lr'])

    # Initialize progress bar
    n_epochs = cfg['training']['n_epochs']
    bar = Bar("Epoch", max=n_epochs)

    for epoch in range(n_epochs):
        # Zero-out remaining gradients
        optimizer.zero_grad()

        # Compute prediction
        prediction = model(data.x_std)
        prediction_3d = prediction.reshape(*data.h_by_column.shape)
        aggregate_prediction_2d = model.aggregate_prediction(prediction_3d.unsqueeze(-1))

        # Compute loss
        loss = torch.square(aggregate_prediction_2d - data.z).mean()
        loss += model.regularization_term()

        # Take gradient step
        loss.backward()
        optimizer.step()

        # Update progress bar
        bar.suffix = f"Loss {loss.item():e}"
        bar.next()

    return model


def predict(model, data):
    with torch.no_grad():
        # Predict on standardized 3D covariates
        prediction = model(data.x_std)

        # Reshape as (time * lat * lon, lev) grid
        prediction_3d = prediction.reshape(*data.h_by_column.shape)
    return prediction_3d


def evaluate(prediction_3d, data, model, cfg, plot, output_dir):
    # Define aggregation wrt non-standardized height for evaluation
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column.unsqueeze(-1).cpu(), dim=-2)
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
