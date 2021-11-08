import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'text.usetex': False,
    'font.family': 'serif',
}
matplotlib.rcParams.update(params)


def colorbar(mappable):
    """
    Stolen from https://joseph-long.com/writing/colorbars/ (thank you!)
    """
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def plot_2d_covariates(dataset, time_idx, covariates_keys):
    """Plots next to each other lat/lon fields of 2D covariates

    Args:
        dataset (xr.Dataset): source dataset
        time_idx (int): index of time to use for slice
        covariates_keys (list[str]): list of variable names to plot

    Returns:
        type: matplotlib.figure.Figure, numpy.ndarray

    """

    field_set = dataset.isel(time=time_idx)
    lon = dataset.lon.values
    lat = dataset.lat.values

    nrows = len(covariates_keys)
    fig, ax = plt.subplots(nrows, 1, figsize=(len(lon) // 4, 6 * nrows))
    cmap = 'magma'
    n_x_ticks = 10
    n_y_ticks = 10
    title_fontsize = 26
    labels_fontsize = 18
    cbar_fontsize = 18
    ticks_fontsize = 14

    for i in range(nrows):
        key = covariates_keys[i]
        im = ax[i].imshow(field_set[key].values, cmap=cmap)
        ax[i].set_xticks(range(0, len(lon), len(lon) // n_x_ticks))
        ax[i].set_xticklabels(np.round(lon[::len(lon) // n_x_ticks], 1), rotation=90, fontsize=ticks_fontsize)
        ax[i].set_yticks(range(0, len(lat), len(lat) // n_y_ticks))
        ax[i].set_yticklabels(np.round(lat[::len(lat) // n_y_ticks], 1), rotation=10, fontsize=ticks_fontsize)
        ax[i].set_title(key, fontsize=title_fontsize)
        cbar = colorbar(im)
        cbar.ax.tick_params(labelsize=cbar_fontsize)
    ax[0].set_xlabel('longitude', fontsize=labels_fontsize)
    ax[0].set_ylabel('latitude', fontsize=labels_fontsize)

    plt.tight_layout()
    return fig, ax


def plot_3d_covariates_slices(dataset, lat_idx, time_idx, covariates_keys):
    """Plots next to each other lon/lev slices of 3D covariates

    Args:
        dataset (xr.Dataset): source dataset
        lat_idx (int): index of latitude to use for slice
        time_idx (int): index of time to use for slice
        covariates_keys (list[str]): list of variable names to plot

    Returns:
        type: matplotlib.figure.Figure, numpy.ndarray

    """
    slice_set = dataset.isel(lat=lat_idx, time=time_idx)
    h = slice_set.isel(lon=len(slice_set.lon) // 2).height.values
    lon = dataset.lon.values

    nrows = len(covariates_keys)
    fig, ax = plt.subplots(nrows, 1, figsize=(len(lon) // 4, 6 * nrows))
    cmap = 'magma'
    n_x_ticks = 10
    n_y_ticks = 4
    title_fontsize = 26
    labels_fontsize = 18
    cbar_fontsize = 18
    ticks_fontsize = 14

    for i in range(nrows):
        key = covariates_keys[i]
        im = ax[i].imshow(slice_set[key].values, cmap=cmap)
        ax[i].set_xticks(range(0, len(lon), len(lon) // n_x_ticks))
        ax[i].set_xticklabels(np.round(lon[::len(lon) // n_x_ticks], 1), fontsize=ticks_fontsize, rotation=90)
        ax[i].set_yticks(range(0, len(h), len(h) // n_y_ticks))
        ax[i].set_yticklabels(np.round(h[::len(h) // n_y_ticks], 1), fontsize=ticks_fontsize, rotation=10)
        ax[i].set_title(key.replace('_', ' '), fontsize=title_fontsize)
        cbar = colorbar(im)
        cbar.ax.tick_params(labelsize=cbar_fontsize)
    ax[0].set_xlabel('longitude', fontsize=labels_fontsize)
    ax[0].set_ylabel('altitude', fontsize=labels_fontsize)

    plt.tight_layout()
    return fig, ax


def plot_aggregate_2d_predictions(dataset, target_key, prediction_3d_grid, aggregate_fn):
    """Plots aggregation of 3D+t prediction, 2D+t aggregate targets used for training and difference

    Args:
        dataset (xr.Dataset): source dataset
        target_key (str): name of target variable
        prediction_3d_grid (torch.Tensor): (time, lat, lon, lev)
        aggregate_fn (callable): callable used to aggregate (time, lat, lon, lev, -1) -> (time, lat, lon)

    Returns:
        type: matplotlib.figure.Figure, numpy.ndarray

    """
    n_row = prediction_3d_grid.size(0)
    n_col = prediction_3d_grid.size(0) * prediction_3d_grid.size(1) * prediction_3d_grid.size(2)
    prediction_3d = prediction_3d_grid.reshape(n_col, -1)

    aggregate_prediction_2d = aggregate_fn(prediction_3d.unsqueeze(-1)).squeeze()
    aggregate_prediction_2d = aggregate_prediction_2d.reshape(n_row, prediction_3d_grid.size(1), prediction_3d_grid.size(2))

    fig, ax = plt.subplots(n_row, 3, figsize=(5 * n_row, 5 * n_row))
    cmap = 'magma'
    title_fontsize = 16
    cbar_fontsize = 12

    for i in range(n_row):
        groundtruth = dataset.isel(time=i)[target_key].values
        pred = aggregate_prediction_2d[i].numpy()
        difference = groundtruth - pred
        vmin = np.minimum(groundtruth, pred).min()
        vmax = np.maximum(groundtruth, pred).max()
        diffmax = np.abs(difference).max()
        rmse = round(np.sqrt(np.mean(difference ** 2)), 4)

        im0 = ax[i, 0].imshow(groundtruth, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = colorbar(im0)
        cbar.ax.tick_params(labelsize=cbar_fontsize)
        ax[i, 0].axis('off')
        ax[i, 0].set_title(f'Groundtruth - Time step {i}', fontsize=title_fontsize)

        im1 = ax[i, 1].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = colorbar(im1)
        cbar.ax.tick_params(labelsize=cbar_fontsize)
        ax[i, 1].axis('off')
        ax[i, 1].set_title(f'Prediction - Time step {i}', fontsize=title_fontsize)

        im2 = ax[i, 2].imshow(difference, cmap='bwr', vmin=-diffmax, vmax=diffmax)
        cbar = colorbar(im2)
        cbar.ax.tick_params(labelsize=cbar_fontsize)
        ax[i, 2].axis('off')
        ax[i, 2].set_title(f'Difference - RMSE {rmse}', fontsize=title_fontsize)

    plt.tight_layout()
    return fig, ax


def plot_vertical_prediction_slice(dataset, lat_idx, time_idx, groundtruth_key, prediction_3d_grid):
    """Plots lon/alt slice of 3D prediction next to groundtruth, difference and RMSE as a function of height

    Args:
        dataset (xr.Dataset): source dataset
        lat_idx (int): index of latitude to use for slice
        time_idx (int): index of time to use for slice
        groundtruth_key (str): name of groundtruth variable
        prediction_3d_grid (torch.Tensor): (time, lat, lon, lev)

    Returns:
        type: matplotlib.figure.Figure, numpy.ndarray

    """

    h = dataset.isel(lat=lat_idx, time=time_idx, lon=0).height.values
    lon = dataset.lon.values

    predicted_slice = prediction_3d_grid[time_idx, lat_idx]
    groundtruth_slice = dataset.isel(lat=lat_idx, time=time_idx)[groundtruth_key].values.T

    difference = groundtruth_slice - predicted_slice.numpy()
    squared_error = difference ** 2
    total_rmse = round(np.sqrt(np.mean(squared_error)), 4)

    vmin = min(groundtruth_slice.min(), predicted_slice.min())
    vmax = max(groundtruth_slice.max(), predicted_slice.max())
    diffmax = np.abs(difference).max()

    fig, ax = plt.subplots(4, 1, figsize=(10, 18))
    cmap = 'magma'
    n_x_ticks = 10
    n_y_ticks = 4
    title_fontsize = 22
    labels_fontsize = 12
    ticks_fontsize = 12

    im0 = ax[0].imshow(groundtruth_slice.T, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = colorbar(im0)
    cbar.ax.tick_params(labelsize=labels_fontsize)
    title = groundtruth_key.replace('_', ' ')
    ax[0].set_title(f'Groundtruth \n ({title})', fontsize=title_fontsize)

    im1 = ax[1].imshow(predicted_slice.T, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = colorbar(im1)
    cbar.ax.tick_params(labelsize=labels_fontsize)
    ax[1].set_title('Prediction', fontsize=title_fontsize)

    im2 = ax[2].imshow(difference.T, cmap='bwr', vmin=-diffmax, vmax=diffmax)
    cbar = colorbar(im2)
    cbar.ax.tick_params(labelsize=labels_fontsize)
    ax[2].set_title('Difference', fontsize=title_fontsize)

    for i in range(3):
        ax[i].set_xticks(range(0, len(lon), len(lon) // n_x_ticks))
        ax[i].set_xticklabels(np.round(lon[::len(lon) // n_x_ticks], 1), fontsize=ticks_fontsize, rotation=90)
        ax[i].set_yticks(range(0, len(h), len(h) // n_y_ticks))
        ax[i].set_yticklabels(np.round(h[::len(h) // n_y_ticks], 1), fontsize=ticks_fontsize, rotation=10)
        ax[i].set_xlabel('longitude', fontsize=labels_fontsize)
        ax[i].set_ylabel('altitude', fontsize=labels_fontsize)

    ax[3].plot(h, np.sqrt(np.mean(squared_error, axis=0)), '--.', label=f'RMSE={total_rmse}')
    ax[3].set_yscale('log')
    ax[3].grid(alpha=0.5)
    ax[3].set_xlabel('altitude', fontsize=labels_fontsize)
    ax[3].set_ylabel('RMSE', fontsize=labels_fontsize)
    ax[3].set_title("RMSE profile", fontsize=title_fontsize)

    plt.legend(fontsize=labels_fontsize)
    plt.tight_layout()
    return fig, ax
