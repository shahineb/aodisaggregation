"""
Variables naming conventions:

    - `dataset` : original xarray dataset
    - `standard_dataset` : standardized xarray dataset (i.e. all variables have mean 0 and variance 1)
    - `x_<whatever>` : 3D covariates
    - `τ_<whatever>` : aggregate targets
    - `gt_<whatever>` : 3D groundtruth
    - `h_<whatever>` : heights levels
    - `x_grid, τ_grid, gt_grid, h_grid` : tensors in the original gridding shape i.e. (time, lat, lon, [lev], [-1])
    - `x_by_column`, `gt_by_column`, `h_by_column` : tensors reshaped by column i.e. (time * lat * lon, lev, [-1])
    - `x, y, τ, gt` : tensors flattened as (time * lat * lon * [lev], -1)
    - `<whatever>_std` : standardized tensor, i.e. has mean 0 and variance 1

N.B.
    - Not all variables necessarily appear in the code
    - brackets `[<whatever>]` used above to denote optional dimensions since some fields are 2D only
"""
from collections import namedtuple
from .preprocess_model_data import load_dataset, standardize
from .tensors import make_3d_covariates_tensors, make_3d_tensor, make_2d_tensor


field_names = ['dataset',
               'standard_dataset',
               'x_by_column_std',
               'x',
               'x_std',
               'τ',
               'τ_std',
               'τ_smooth',
               'τ_smooth_std',
               'gt_by_column',
               'h_by_column',
               'h_by_column_std',
               'mask']
Data = namedtuple(typename='Data', field_names=field_names, defaults=(None,) * len(field_names))


def make_data(cfg):
    """Prepares and formats data to be used for training and testing.

    Returns all data objects needed to run experiment encapsulated in a namedtuple.
    Returned elements are not comprehensive and minimially needed to run experiments,
        they can be subject to change depending on needs.

    Args:
        cfg (dict): configuration file

    Returns:
        type: Data

    """
    # Load dataset as defined in main code
    dataset = load_dataset(cfg['dataset']['path'])

    # TODO : DISABLE IN FINAL VERSION – Subset to speed up development
    # dataset = dataset.isel(lat=slice(30, 60), lon=slice(35, 55), time=slice(0, 3))

    # Compute standardized version
    standard_dataset = standardize(dataset)

    # Make 3D covariate tensors
    x_grid_std, x_by_column_std, x_std, x = _make_x_tensors(cfg, dataset, standard_dataset)

    # Make 2D aggregate target tensors
    τ_grid_std, τ_grid, τ_std, τ, τ_smooth_std, τ_smooth = _make_τ_tensors(cfg, dataset, standard_dataset)

    # Make 3D groundtruth tensor
    gt_grid, gt_by_column = _make_groundtruth_tensors(cfg, dataset, standard_dataset)

    # Make height tensors for integration
    h_grid_std, h_grid, h_by_column_std, h_by_column = _make_h_tensors(cfg, dataset, standard_dataset)

    # Encapsulate into named tuple object
    kwargs = {'dataset': dataset,
              'standard_dataset': standard_dataset,
              'x_by_column_std': x_by_column_std,
              'x': x,
              'x_std': x_std,
              'τ': τ,
              'τ_std': τ_std,
              'τ_smooth': τ_smooth,
              'τ_smooth_std': τ_smooth_std,
              'gt_by_column': gt_by_column,
              'h_by_column': h_by_column,
              'h_by_column_std': h_by_column_std,
              }
    data = Data(**kwargs)
    return data


def _make_x_tensors(cfg, dataset, standard_dataset):
    """Returns 3D covariates tensors in different formats for different purposes

        `x_grid_std`:
            - (time, lat, lon, lev, ndim + 4) standardized tensor
            - No usage

        `x_by_column_std`:
            - (time * lat * lon, lev, ndim + 4) standardized tensor
            - Used for fitting model since each column is a single training sample

        `x_std`:
            - (time * lat * lon * lev, ndim + 4) standardized tensor
            - Used for prediction since fitted regressor acts on 3D covariates

    Returns:
        type: torch.Tensor, torch.Tensor, torch.Tensor

    """
    # Convert into pytorch tensors
    x_grid_std = make_3d_covariates_tensors(dataset=standard_dataset, variables_keys=cfg['dataset']['3d_covariates'], standardize_coords=True)
    x_grid = make_3d_covariates_tensors(dataset=dataset, variables_keys=cfg['dataset']['3d_covariates'], standardize_coords=False)

    # Reshape tensors
    x_by_column_std = x_grid_std.reshape(-1, x_grid_std.size(-2), x_grid_std.size(-1))
    x_std = x_by_column_std.view(-1, x_grid_std.size(-1))
    x = x_grid.reshape(-1, x_grid.size(-1))
    return x_grid_std, x_by_column_std, x_std, x


def _make_τ_tensors(cfg, dataset, standard_dataset):
    """Returns aggregate targets tensors in different formats for different purposes

        `τ_grid_std`:
            - (time, lat, lon) standardized tensor
            - No usage

        `τ_grid`:
            - (time, lat, lon) non-standardized tensor
            - Used to unstandardize prediction and evaluation


        `τ_std`:
            - (time * lat * lon) standardized tensor
            - Used for fitting

        `τ`:
            - (time * lat * lon) non-standardized tensor
            - Used for fitting

        `τ_smooth_std`:
            - (time * lat * lon) standardized tensor
            - Used for rescaling

        `τ_smooth`:
            - (time * lat * lon) non-standardized tensor
            - Used for rescaling


    Returns:
        type: torch.Tensor, torch.Tensor

    """
    # Convert into pytorch tensors
    τ_grid_std = make_2d_tensor(dataset=standard_dataset, variable_key=cfg['dataset']['target'])
    τ_grid = make_2d_tensor(dataset=dataset, variable_key=cfg['dataset']['target'])
    τ_smooth_grid_std = make_2d_tensor(dataset=standard_dataset, variable_key=cfg['dataset']['smoothed_target'])
    τ_smooth_grid = make_2d_tensor(dataset=dataset, variable_key=cfg['dataset']['smoothed_target'])

    # Reshape tensors
    τ_std = τ_grid_std.view(-1)
    τ = τ_grid.view(-1)
    τ_smooth_std = τ_smooth_grid_std.view(-1)
    τ_smooth = τ_smooth_grid.view(-1)
    return τ_grid_std, τ_grid, τ_std, τ, τ_smooth_std, τ_smooth


def _make_groundtruth_tensors(cfg, dataset, standard_dataset):
    """Returns 3D groundtruth tensors in different formats for different purposes

        `gt_grid`:
            - (time, lat, lon, lev) non-standardized tensor
            - No usage

        `gt_by_column`:
            - (time * lat * lon, lev) non-standardized tensor
            - Used for evaluation

    Returns:
        type: torch.Tensor

    """
    # Convert into pytorch tensors
    gt_grid = make_3d_tensor(dataset=dataset, variable_key=cfg['dataset']['groundtruth'])

    # Reshape tensors
    gt_by_column = gt_grid.reshape(-1, gt_grid.size(-1))
    return gt_grid, gt_by_column


def _make_h_tensors(cfg, dataset, standard_dataset):
    """Returns heights levels tensors in different formats for different purposes

        `h_grid_std`:
            - (time, lat, lon, lev) standardized tensor
            - No usage

        `h_grid`:
            - (time, lat, lon, lev) non-standardized tensor
            - No usage

        `h_by_column_std`:
            - (time * lat * lon, lev) standardized tensor
            - Used to aggregate against standardized height levels for training – stabilizes fitting computations

        `h_by_column`:
            - (time * lat * lon, lev) non-standardized tensor
            - Used to unstandardized prediction and aggregate groundtruth against height at evaluation

    Returns:
        type: torch.Tensor

    """
    # Convert into pytorch tensors
    h_grid_std = make_3d_tensor(dataset=standard_dataset, variable_key='height')
    h_grid = make_3d_tensor(dataset=dataset, variable_key='height')

    # Reshape tensors
    h_by_column_std = h_grid_std.reshape(-1, h_grid.size(-1))
    h_by_column = h_grid.reshape(-1, h_grid.size(-1))
    return h_grid_std, h_grid, h_by_column_std, h_by_column
