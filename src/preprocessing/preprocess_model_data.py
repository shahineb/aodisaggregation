import xarray as xr

standardize = lambda x: (x - x.mean()) / x.std()


def load_dataset(file_path):
    # Load dataset
    dataset = xr.open_dataset(file_path)
    return dataset
