import itertools


def product_dict(**kwargs):
    keys = list(kwargs.keys())
    vals = list(kwargs.values())
    sorted_idx = sorted(range(len(keys)), key=keys.__getitem__)
    sorted_keys = [keys[i] for i in sorted_idx]
    sorted_vals = [vals[i] for i in sorted_idx]
    for instance in itertools.product(*sorted_vals):
        yield dict(zip(sorted_keys, instance))


def flatten_dict(buffer, parent_key="", output_dict=None):
    if output_dict is None:
        output_dict = dict()

    if isinstance(buffer, dict) and len(buffer.keys()) > 0:
        for child_key in buffer.keys():
            key = '_'.join([parent_key, child_key])
            flatten_dict(buffer[child_key], key, output_dict)
    else:
        output_dict.update({parent_key[1:]: buffer})
    return output_dict


def flatten_dict_as_str(my_dict):
    flattened_dict = flatten_dict(my_dict)
    dict_as_str = '_'.join(f'{k}={v}' for k, v in flattened_dict.items())
    return dict_as_str
