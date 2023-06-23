import numpy as np
import torch

standardize = lambda x: (x - x.mean()) / x.std()


def make_grid_tensor(field, coords_keys):
    """Makes ND tensor grid corresponding to
    provided xarray dataarray coordinates
    """
    coords = []
    for key in coords_keys:
        coord = torch.from_numpy(field[key].values.astype('float'))
        coords.append(coord)
    grid = torch.stack(torch.meshgrid(*coords), dim=-1).float()
    return grid


def make_3d_covariates_tensors(dataset, variables_keys, standardize_coords=False):
    """Makes (time, lat, lon, lev, 4 + n_variable) tensor of 3D+t covariates

    Args:
        dataset (xr.Dataset): source dataset formatted as (time, lev, lat, lon, n_variable)
        variables_keys (list): list of 3D covariates to include
        standardize_coords (bool): if True, standardises coordinates

    Returns:
        type: torch.Tensor

    """
    grid_3d_t = make_grid_tensor(dataset, coords_keys=['time', 'lev', 'lat', 'lon'])

    # Correct for change in longitude distance at different latitudes
    wlat = torch.from_numpy(np.cos(np.deg2rad(dataset.lat.values))).float().view(1, 1, -1, 1)
    grid_3d_t[..., -1].mul_(wlat)

    if standardize_coords:
        mean = grid_3d_t.view(-1, 1).mean(dim=0)
        stddev = grid_3d_t.view(-1, 1).std(dim=0)
        grid_3d_t = (grid_3d_t - mean) / stddev

    covariates_grid = np.stack([dataset[key] for key in variables_keys], axis=-1)
    grid = torch.cat([grid_3d_t, torch.from_numpy(covariates_grid)], dim=-1).float()
    grid = grid.permute(0, 2, 3, 1, 4)
    return grid


def make_2d_covariates_tensors(dataset, variables_keys):
    """Makes (time, lat, lon, 3 + n_variable) tensor of 2D+t covariates

    Args:
        dataset (xr.Dataset): source dataset
        variables_keys (list): list of 2D covariates to include

    Returns:
        type: torch.Tensor

    """
    grid_2d_t = make_grid_tensor(dataset, coords_keys=['time', 'lat', 'lon'])
    covariates_grid = np.stack([dataset[key] for key in variables_keys], axis=-1)
    grid = torch.cat([grid_2d_t, torch.from_numpy(covariates_grid)], dim=-1).float()
    return grid


def make_2d_tensor(dataset, variable_key):
    """Makes (time, lat, lon) tensor of 2D+t values

    Args:
        dataset (xr.Dataset): source dataset
        variable_key (str): name of variable to use

    Returns:
        type: torch.Tensor

    """
    grid = torch.from_numpy(dataset[variable_key].values).float()
    return grid


def make_3d_tensor(dataset, variable_key):
    """Makes (time, lat, lon, lev) tensor of 3D+t values

    Args:
        dataset (xr.Dataset): source dataset formatted as (time, lev, lat, lon)
        variable_key (str): name of variable to use

    Returns:
        type: torch.Tensor

    """
    grid = torch.from_numpy(dataset[variable_key].values).float()
    grid = grid.permute(0, 2, 3, 1)
    return grid
