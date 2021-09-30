"""
Description : Runs hyperparameter search for aggregate ridge regression experiment

Usage: run_hyperparams_search_ridge_regression.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --device=<device_index>          Device to use [default: cpu]
"""
import os
import yaml
import logging
from docopt import docopt
from sklearn.model_selection import KFold
from src.preprocessing import make_data, split_data
from .run_ridge_regression import migrate_to_device, make_model, fit, predict, evaluate


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    data = make_data(cfg=cfg, include_2d=False)

    # Move needed tensors only to device
    data = migrate_to_device(data=data)

    # Make k-fold iterator
    kfold = KFold(n_splits=cfg['search']['n_splits'])

    # Iterate over folds
    for train_idx, test_idx in kfold.split(data.x_by_column_std):

        # Split training and testing data
        train_data, test_data = split_data(data=data, train_idx=train_idx, test_idx=test_idx)

        # Instantiate model with training data
        model = make_model(cfg=cfg, data=train_data)

        # Fit model on training data
        model = fit(model=model, data=train_data)

        # Run prediction on testing data
        prediction_3d = predict(model=model, data=test_data)

        # Run evaluation
        evaluate(prediction_3d=prediction_3d, data=test_data, model=model, cfg=cfg, plot=False, output_dir=args['--o'])


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
