import numpy as np
import matplotlib.pyplot as plt


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
    fig, ax = plt.subplots(nrows, 1, figsize=(5 * nrows, 5 * nrows))
    cmap = 'magma'
    n_x_ticks = 100
    n_y_ticks = 100
    title_fontsize = 20
    labels_fontsize = 12
    cbar_fontsize = 12

    for i in range(nrows):
        key = covariates_keys[i]
        im = ax[i].imshow(field_set[key].values, cmap=cmap)
        ax[i].set_xticks(range(0, len(lon), n_x_ticks))
        ax[i].set_xticklabels(lon[::n_x_ticks], Fontsize=cbar_fontsize)
        ax[i].set_yticks(range(0, len(lat), n_y_ticks))
        ax[i].set_yticklabels(lat[::n_x_ticks], rotation=10, Fontsize=cbar_fontsize)
        ax[i].set_title(key, fontsize=title_fontsize)
        cbar = plt.colorbar(im, orientation="vertical", ax=ax[i], shrink=0.8)
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
    h = slice_set.isel(lon=len(slice_set.lon) // 2).grheightm1.values
    lon = dataset.lon.values

    nrows = len(covariates_keys)
    fig, ax = plt.subplots(nrows, 1, figsize=(2 * nrows, 4 * nrows))
    cmap = 'magma'
    n_x_ticks = 20
    n_y_ticks = 20
    title_fontsize = 26
    labels_fontsize = 18
    cbar_fontsize = 18

    for i in range(nrows):
        key = covariates_keys[i]
        im = ax[i].imshow(slice_set[key].values, cmap=cmap)
        ax[i].set_xticks(range(0, len(lon), n_x_ticks))
        ax[i].set_xticklabels(lon[::n_x_ticks])
        ax[i].set_yticks(range(0, len(h), n_y_ticks))
        ax[i].set_yticklabels(h[::n_x_ticks], rotation=10)
        ax[i].set_title(key, fontsize=title_fontsize)
        cbar = plt.colorbar(im, orientation="vertical", ax=ax[i], shrink=0.7)
        cbar.ax.tick_params(labelsize=cbar_fontsize)
    ax[0].set_xlabel('longitude', fontsize=labels_fontsize)
    ax[0].set_ylabel('altitude', fontsize=labels_fontsize)

    plt.tight_layout()
    return fig, ax
