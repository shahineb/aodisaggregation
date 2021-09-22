import os
import yaml
import logging
import torch
from collections import namedtuple
import src.preprocessing as preproc


def make_toy_data(cfg):
    # Load dataset as defined in main code
    dataset = preproc.load_dataset(cfg['dataset']['path'])

    # Subset to speed up testing
    dataset = dataset.isel(lat=slice(30, 60), lon=slice(35, 55), time=slice(0, 3))

    # Compute standardized version
    standard_dataset = preproc.standardize(dataset)

    # Convert into pytorch tensors
    x_grid_std = preproc.make_3d_covariates_tensors(dataset=standard_dataset, variables_keys=cfg['dataset']['3d_covariates'])
    z_grid_std = preproc.make_2d_tensor(dataset=standard_dataset, variable_key=cfg['dataset']['target'])
    h_grid_std = preproc.make_3d_tensor(dataset=standard_dataset, variable_key='height')
    z_grid = preproc.make_2d_tensor(dataset=dataset, variable_key=cfg['dataset']['target'])
    gt_grid = preproc.make_3d_tensor(dataset=dataset, variable_key=cfg['dataset']['groundtruth'])
    h_grid = preproc.make_3d_tensor(dataset=dataset, variable_key='height')

    # Reshape tensors
    x_by_column_std = x_grid_std.reshape(-1, x_grid_std.size(-2), x_grid_std.size(-1))
    x_std = x_by_column_std.reshape(-1, x_grid_std.size(-1))
    z_std = z_grid_std.flatten()
    h = h_grid.reshape(-1, x_grid_std.size(-2))
    h_std = h_grid_std.reshape(-1, x_grid_std.size(-2))

    # Encapsulate into named tuple
    ToyData = namedtuple('ToyData', ['dataset', 'standard_dataset', 'x_by_column_std', 'x_std', 'z_grid', 'z_std', 'gt_grid', 'h', 'h_std'])
    toy_data = ToyData(dataset=dataset, standard_dataset=standard_dataset,
                       x_by_column_std=x_by_column_std, x_std=x_std, z_grid=z_grid, z_std=z_std, gt_grid=gt_grid, h=h, h_std=h_std)
    return toy_data
