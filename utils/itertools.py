import itertools


def product_dict(**kwargs):
    """
    Cartesian product of dictionaries.

    Stolen from oxcsml github (itself stolen from https://stackoverflow.com/a/5228294/3160671)

    Example:
    >>> list(product_dict(lr=[0.005, 0.001], n_epochs=[2, 3]))
    >>> [{"lr": 0.005, "n_epoch": 2}, {"lr": 0.005, "n_epoch": 3}, {"lr": 0.001, "n_epoch": 2}, {"lr": 0.001, "n_epoch": 3}]
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))
