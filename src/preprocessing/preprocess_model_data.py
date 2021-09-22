import numpy as np
import xarray as xr

standardize = lambda x: (x - x.mean()) / x.std()


def load_dataset(file_path):
    # Load dataset
    dataset = xr.open_dataset(file_path).isel(lat=slice(0, 100),
                                              lon=slice(100, 200),
                                              time=slice(0, 3))
    return dataset


def to_log_domain(dataset, variables_keys):
    for key in variables_keys:
        dataset['log_' + key] = np.log(dataset[key])
    return dataset
