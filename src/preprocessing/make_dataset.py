"""
Variables naming nomenclature:

    - `dataset` : original xarray dataset
    - `standard_dataset` : standardized xarray dataset (i.e. all variables have mean 0 and variance 1)
    - `x_<whatever>` : 3D covariates
    - `y_<whatever>` : 2D covariates
    - `z_<whatever>` : aggregate targets
    - `gt_<whatever>` : 3D groundtruth
    - `h_<whatever>` : heights levels
    - `x_grid, y_grid, z_grid, gt_grid, h_grid` : tensors in the original gridding shape i.e. (time, lat, lon, [lev], [-1])
    - `x, y, z, gt` : tensors flattened as (time * lat * lon * [lev], -1)
    - `<whatever>_std` : standardized tensor, i.e. has mean 0 and variance 1

N.B.
    - Not all variables necessarily appear in the code
    - [<whatever>] denotes optional dimensions since some fields are 2D only
    - h tensors are an exception which we conceed for code convenience
"""
from collections import namedtuple
from .preprocess_model_data import load_dataset, standardize
from .tensors import make_3d_covariates_tensors, make_2d_covariates_tensors, make_3d_tensor, make_2d_tensor


field_names = ['dataset', 'standard_dataset', 'x_by_column_std', 'x_std', 'y_std', 'z_std', 'z_grid', 'gt_grid', 'h', 'h_std']
Data = namedtuple(typename='Data', field_names=field_names, defaults=(None,) * len(field_names))


def make_dataset(cfg, include_2d=False):
    """Prepares and formats data to be used for training and testing.

    Returns all data objects needed to run experiment encapsulated in a namedtuple.
    Returned elements are not comprehensive and minimially needed to run experiments,
        they can be subject to change depending on needs.

    Args:
        cfg (dict): configuration file
        include_2d (bool): if True, prepares 2D covariates tensors as well

    Returns:
        type: Data

    """
    # Load dataset as defined in main code
    dataset = load_dataset(cfg['dataset']['path'])

    # Subset to speed up testing
    dataset = dataset.isel(lat=slice(30, 60), lon=slice(35, 55), time=slice(0, 3))

    # Compute standardized version
    standard_dataset = standardize(dataset)

    # Make 3D covariate tensors
    x_grid_std, x_by_column_std, x_std = _make_x_tensors(cfg, dataset, standard_dataset)

    # Make 2D covariate tensors
    if include_2d:
        y_grid_std, y_std = _make_y_tensors(cfg, dataset, standard_dataset)

    # Make 2D aggregate target tensors
    z_grid_std, z_grid, z_std = _make_z_tensors(cfg, dataset, standard_dataset)

    # Make 3D groundtruth tensor
    gt_grid = _make_groundtruth_tensors(cfg, dataset, standard_dataset)

    # Make height tensors for integration
    h_grid_std, h_grid, h_std, h = _make_h_tensors(cfg, dataset, standard_dataset)

    # Encapsulate into named tuple object
    kwargs = {'dataset': dataset,
              'standard_dataset': standard_dataset,
              'x_by_column_std': x_by_column_std,
              'x_std': x_std,
              'z_std': z_std,
              'z_grid': z_grid,
              'gt_grid': gt_grid,
              'h': h,
              'h_std': h_std,
              }
    if include_2d:
        kwargs.update({'y_std': y_std})
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
    x_grid_std = make_3d_covariates_tensors(dataset=standard_dataset, variables_keys=cfg['dataset']['3d_covariates'])

    # Reshape tensors
    x_by_column_std = x_grid_std.reshape(-1, x_grid_std.size(-2), x_grid_std.size(-1))
    x_std = x_by_column_std.reshape(-1, x_grid_std.size(-1))
    return x_grid_std, x_by_column_std, x_std


def _make_y_tensors(cfg, dataset, standard_dataset):
    """Returns 2D covariates tensors in different formats for different purposes

        `y_grid_std`:
            - (time, lat, lon, ndim + 3) standardized tensor
            - No usage

        `y_std`:
            - (time * lat * lon, ndim + 3) standardized tensor
            - Used for fitting

    Returns:
        type: torch.Tensor, torch.Tensor

    """
    # Convert into pytorch tensors
    y_grid_std = make_2d_covariates_tensors(dataset=standard_dataset, variables_keys=cfg['dataset']['2d_covariates'])

    # Reshape tensors
    y_std = y_grid_std.reshape(-1, y_grid_std.size(-1))
    return y_grid_std, y_std


def _make_z_tensors(cfg, dataset, standard_dataset):
    """Returns aggregate targets tensors in different formats for different purposes

        `z_grid_std`:
            - (time, lat, lon) standardized tensor
            - No usage

        `z_grid`:
            - (time, lat, lon) non-standardized tensor
            - Used to unstandardize prediction and evaluation


        `z_std`:
            - (time * lat * lon) standardized tensor
            - Used for fitting

    Returns:
        type: torch.Tensor, torch.Tensor

    """
    # Convert into pytorch tensors
    z_grid_std = make_2d_tensor(dataset=standard_dataset, variable_key=cfg['dataset']['target'])
    z_grid = make_2d_tensor(dataset=dataset, variable_key=cfg['dataset']['target'])

    # Reshape tensors
    z_std = z_grid_std.flatten()
    return z_grid_std, z_grid, z_std


def _make_groundtruth_tensors(cfg, dataset, standard_dataset):
    """Returns 3D groundtruth tensors in different formats for different purposes

        `gt_grid`:
            - (time, lat, lon, lev) non-standardized tensor
            - Used for evaluation

    Returns:
        type: torch.Tensor

    """
    # Convert into pytorch tensors
    gt_grid = make_3d_tensor(dataset=dataset, variable_key=cfg['dataset']['groundtruth'])
    return gt_grid


def _make_h_tensors(cfg, dataset, standard_dataset):
    """Returns heights levels tensors in different formats for different purposes

        `h_grid_std`:
            - (time, lat, lon, lev) standardized tensor
            - No usage

        `h_grid`:
            - (time, lat, lon, lev) non-standardized tensor
            - No usage

        `h_std`:
            - (time * lat * lon, lev) standardized tensor
            - Used to aggregate against standardized height levels for training – stabilizes fitting computations

        `h`:
            - (time * lat * lon, lev) non-standardized tensor
            - Used to unstandardized prediction and aggregate groundtruth against height at evaluation

    Returns:
        type: torch.Tensor

    """
    # Convert into pytorch tensors
    h_grid_std = make_3d_tensor(dataset=standard_dataset, variable_key='height')
    h_grid = make_3d_tensor(dataset=dataset, variable_key='height')

    # Reshape tensors
    h = h_grid.reshape(-1, h_grid.size(-1))
    h_std = h_grid_std.reshape(-1, h_grid.size(-1))
    return h_grid_std, h_grid, h_std, h
