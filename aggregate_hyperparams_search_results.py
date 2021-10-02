"""
Description : Aggregate together in dataframe outputs from grid search run.

Usage: aggregate_hyperparams_search_results.py  [options] --i=<input_dir> --o=<output_dir>

Options:
  --i=<input_dir>                  Path to directory where grid search outputs are saved.
  --o=<output_dir>                 Output directory.
  --k=<top_k>                      Return k best set of hyperparameters [default: 3]
"""
import os
import yaml
import logging
from docopt import docopt
import xarray as xr
import numpy as np
from utils import product_dict, flatten_dict, flatten_dict_as_str


def main(args):
    # Load configuration file corresponding to grid search
    with open(os.path.join(args['--i'], 'cfg.yaml'), "r") as f:
        cfg = yaml.safe_load(f)

    # Load all scores in xarray dataset with hyperparameters as dimensions
    logging.info("Loading grid search scores... (can take a while)")
    scores_dataset = open_scores_as_xarray(dirpath=args['--i'], cfg=cfg)
    logging.info(f"Loaded grid search scores \n {scores_dataset}")

    # Extract best set of hyperparameters for each metric
    best_hyperparams_by_metric = extract_best_hyperparams(scores_dataset=scores_dataset, k=int(args['--k']))
    logging.info("Computed candidate set of hyperparameters")

    # Dump entire scores dataset and best candidates
    dump_results(scores_dataset=scores_dataset,
                 best_hyperparams_by_metric=best_hyperparams_by_metric,
                 output_dir=args['--o'])
    logging.info(f"Dumped results in {args['--o']}")


def open_scores_as_xarray(dirpath, cfg):
    """Loads k-fold grid search scores (as saved by execution of grid search script)
    into xarray with dimensions (hyperparam_1, hyperparam_2, ..., hyperparam_n, fold)
    and with variables the different metrics.

    Args:
        dirpath (str): path to directory where grid search scores are saved.
        cfg (dict): configuration file used to perform grid search (typically saved in dirpath)

    Returns:
        type: xarray.Dataset

    """
    # Initialize hyperparameters grid
    hyperparams_grid = product_dict(**cfg['search']['grid'])

    # Initialize xarray dataset to record scores
    scores_dataset = init_scores_dataset(cfg=cfg)

    # Iterate over sets of hyperparameters
    for hyperparams in hyperparams_grid:

        # Look up corresponding directory
        hyperparams_dirname = flatten_dict_as_str(hyperparams)
        hyperparams_dirpath = os.path.join(dirpath, hyperparams_dirname)

        # Iterate of CV folds
        for i in range(cfg['search']['n_splits']):
            # Include fold number as hyperparam
            hyperparams_and_fold = {**hyperparams, **{'fold': i + 1}}

            # Look up corresponding directory
            fold_dirpath = os.path.join(hyperparams_dirpath, f'fold_{i + 1}')

            # Load corresponding scores
            scores_path = os.path.join(fold_dirpath, 'scores.metrics')
            with open(scores_path, "r") as f:
                scores = flatten_dict(yaml.safe_load(f))

            # Initialize metrics dataarrays in scores dataset
            if not scores_dataset.data_vars.variables:
                scores_dataset = init_metrics_dataarrays(scores_dataset=scores_dataset, metrics=list(scores.keys()))

            # Record in dataset the value of each metric
            for metric, value in scores.items():
                scores_dataset[metric].loc[hyperparams_and_fold] = value
    return scores_dataset


def init_scores_dataset(cfg):
    """Initializes empty xarray dataset with dimensions (hyperparam_1, hyperparam_2, ..., hyperparam_n, fold)
    and no variables.

    Args:
        cfg (dict): configuration file used to perform grid search

    Returns:
        type: xarray.Dataset

    """
    cv_search_grid = cfg['search']['grid'].copy()
    cv_search_grid.update({'fold': list(range(1, cfg['search']['n_splits'] + 1))})
    scores_dataset = xr.Dataset(coords=cv_search_grid)
    return scores_dataset


def init_metrics_dataarrays(scores_dataset, metrics):
    """Initializes empty datarrays for each metric as variables of the scores dataset.

    Args:
        scores_dataset (xarray.Dataset): Empty xarray dataset with no variables.
        metrics (list[str]): list of metrics names.

    Returns:
        type: xarray.Dataset

    """
    dims = list(scores_dataset.dims.keys())
    shape = list(scores_dataset.dims.values())
    for metric in metrics:
        scores_dataset[metric] = (dims, np.empty(shape))
    return scores_dataset


def extract_best_hyperparams(scores_dataset, k):
    """For each metric in the dataset, extracts:
        - k set of hyperparameters maximizing it, with corresponding mean score
            accross folds and standard deviation
        - k set of hyperparameters minimizing it, with corresponding mean score
            accross folds and standard deviation


        Returns dictionnary of the form
        ```
            {metric_name: {'head': {1: {hyperparmeter_1: value_1,
                                        hyperparameter_2: value_2,
                                        ...
                                        hyperparameter_n: value_n,
                                        mean: mean_score,
                                        std: std_score},
                                    2: ...},

                          {'tail': {1: {hyperparmeter_1: value_1,
                                        hyperparameter_2: value_2,
                                        ...
                                        hyperparameter_n: value_n,
                                        mean: mean_score,
                                        std: std_score},
                                    2: ...},
             }
        ```

    Args:
        scores_dataset (xarray.Dataset): xarray dataset of grid search scores.
        k (int): number of top/last sets of hyperparameters to retrieve.

    Returns:
        type: dict

    """
    # Take mean and standard deviation of metrics accross folds
    mean_scores = scores_dataset.mean(dim='fold')
    std_scores = scores_dataset.std(dim='fold')

    # Initialize emtpy output dictionnary
    output_dict = dict()

    # Iterate over metrics dataarrats
    for metric, values in mean_scores.data_vars.variables.items():
        # Compute multidimensional sorting indices
        sorted_idx = np.stack(np.unravel_index(np.argsort(values.values, axis=None), values.shape))

        # Get indices of top/last values
        top_k_idx = sorted_idx.T[-k::-1].tolist()
        last_k_idx = sorted_idx.T[:k].tolist()

        # Initialize empty subdictionnaries for this metric
        output_dict[metric] = dict()
        output_dict[metric]['head'] = dict()
        output_dict[metric]['tail'] = dict()

        # Iterate over ith top/last values
        for i in range(k):
            # Extract ith top hyperparameters, mean score and standard deviation
            ith_top_idx = top_k_idx[i]
            ith_top_hyperparams = {key: values.values[ith_top_idx[j]].item()
                                   for j, (key, values) in enumerate(mean_scores.coords.items())}
            ith_best_scores = {'mean': float(mean_scores[metric].sel(ith_top_hyperparams).values),
                               'std': float(std_scores[metric].sel(ith_top_hyperparams).values)}
            output_dict[metric]['head'][i + 1] = {**ith_top_hyperparams, **ith_best_scores}

            # Extract ith last hyperparameters, mean score and standard deviation
            ith_last_idx = last_k_idx[i]
            ith_last_hyperparams = {key: values.values[ith_last_idx[j]].item()
                                    for j, (key, values) in enumerate(mean_scores.coords.items())}
            ith_last_scores = {'mean': float(mean_scores[metric].sel(ith_last_hyperparams).values),
                               'std': float(std_scores[metric].sel(ith_last_hyperparams).values)}
            output_dict[metric]['tail'][i + 1] = {**ith_last_hyperparams, **ith_last_scores}
    return output_dict


def dump_results(scores_dataset, best_hyperparams_by_metric, output_dir):
    dataset_path = os.path.join(output_dir, 'cv-search-scores.nc')
    scores_dataset.to_netcdf(path=dataset_path)

    best_hyperparams_path = os.path.join(output_dir, 'best_hyperparameters.yaml')
    with open(best_hyperparams_path, 'w') as f:
        yaml.safe_dump(best_hyperparams_by_metric, f)


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Arguments: {args}\n')

    # Create output directory if doesn't exists
    os.makedirs(args['--o'], exist_ok=True)

    # Run
    main(args)
