"""
Description : Runs hyperparameter search for warped aggregate ridge regression experiment

Usage: run_hyperparams_search_warped_ridge_regression.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --device=<device_index>          Device to use [default: cpu]
"""
import os
import yaml
import logging
from docopt import docopt
from progress.bar import Bar
import torch
from sklearn.model_selection import KFold
from src.preprocessing import make_data, split_data
from src.models import WarpedAggregateRidgeRegression
from src.evaluation import dump_scores
from utils import product_dict, flatten_dict_as_str
from run_warped_ridge_regression import migrate_to_device, fit, predict


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    data = make_data(cfg=cfg, include_2d=False)

    # Move needed tensors only to device
    data = migrate_to_device(data=data, device=device)

    # Create cartesian product of grid search parameters
    hyperparams_grid = list(product_dict(**cfg['search']['grid']))
    n_grid_points = len(hyperparams_grid)
    search_bar = Bar("Grid Search", max=n_grid_points)
    search_bar.finish()

    # Iterate over combinations of hyperparameters
    for j, hyperparams in enumerate(hyperparams_grid):

        # Flatten out hyperparameters into string to name output directory
        dirname = flatten_dict_as_str(hyperparams)
        output_dir = os.path.join(args['--o'], dirname)

        # Create directory and dump current set of hyperparameters
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'hyperparams.yaml'), 'w') as f:
            yaml.dump(hyperparams, f)

        # Make k-fold iterator
        kfold = KFold(n_splits=cfg['search']['n_splits'], shuffle=False)
        cv_bar = Bar("Folds", max=cfg['search']['n_splits'])

        # Iterate over folds
        for i, (train_idx, test_idx) in enumerate(kfold.split(data.x_by_column_std)):

            # Split training and testing data
            train_data, test_data = split_data(data=data, train_idx=train_idx, test_idx=test_idx)

            # Instantiate model with training data
            model = make_model(cfg=cfg, data=train_data, hyperparams=hyperparams)

            # Fit model on training data
            model = fit(model=model, data=train_data, cfg=cfg)

            # Run prediction on testing data
            prediction_3d = predict(model=model, data=test_data)

            # Run evaluation
            fold_output_dir = os.path.join(output_dir, f"fold_{i + 1}")
            evaluate(prediction_3d=prediction_3d, data=test_data, output_dir=fold_output_dir)

            # Update progress bar
            cv_bar.next()

        # Update progress bar
        search_bar.suffix = f"{j + 1}/{n_grid_points} | {hyperparams}"
        search_bar.next()
        search_bar.finish()


def make_model(cfg, data, hyperparams):
    # Create aggregation operator
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column.unsqueeze(-1), dim=-2)
        return aggregated_grid

    # Define warping transformation
    if hyperparams['transform'] == 'linear':
        transform = lambda x: x
    elif hyperparams['transform'] == 'softplus':
        transform = lambda x: torch.log(1 + torch.exp(x))
    elif hyperparams['transform'] == 'smooth_abs':
        transform = lambda x: torch.nn.functional.smooth_l1_loss(x, torch.zeros_like(x), reduction='none')
    elif hyperparams['transform'] == 'square':
        transform = torch.square
    elif hyperparams['transform'] == 'exp':
        transform = torch.exp
    else:
        raise ValueError("Unknown transform")

    # Fix weights initialization seed
    torch.random.manual_seed(cfg['model']['seed'])

    # Instantiate model
    model = WarpedAggregateRidgeRegression(lbda=hyperparams['lbda'],
                                           transform=transform,
                                           aggregate_fn=trpz,
                                           ndim=len(cfg['dataset']['3d_covariates']) + 4,
                                           fit_intercept=hyperparams['fit_intercept'])
    return model.to(device)


def evaluate(prediction_3d, data, output_dir):
    # Create output directory if doesn't exists
    os.makedirs(output_dir, exist_ok=True)

    # Define aggregation wrt non-standardized height for evaluation
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column.unsqueeze(-1).cpu(), dim=-2)
        return aggregated_grid

    # Dump scores in output dir
    dump_scores(prediction_3d=prediction_3d.cpu(),
                groundtruth_3d=data.gt_by_column,
                targets_2d=data.z.cpu(),
                aggregate_fn=trpz,
                output_dir=output_dir)


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
    logging.info("Grid search completed")
