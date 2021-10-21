import pytest
import torch
from src.evaluation import metrics


"""
TESTING FIXTURES
"""


@pytest.fixture(scope='module')
def ones_tensor():
    return torch.ones(100, 50)


@pytest.fixture(scope='module')
def zeros_tensor():
    return torch.zeros(100, 50)


@pytest.fixture(scope='module')
def randn_tensor_1():
    torch.random.manual_seed(1)
    return torch.randn(100, 50)


@pytest.fixture(scope='module')
def randn_tensor_2():
    torch.random.manual_seed(2)
    return torch.randn(100, 50)


@pytest.fixture(scope='module')
def aggregate_fn():
    return lambda x: x.sum(dim=-2)


"""
TESTS
"""


def test_spearman_correlation(randn_tensor_1, randn_tensor_2):
    positive_randn_corr = metrics.spearman_correlation(randn_tensor_1.flatten(), randn_tensor_1.flatten())
    assert positive_randn_corr == 1

    negative_randn_corr = metrics.spearman_correlation(randn_tensor_1.flatten(), -randn_tensor_1.flatten())
    assert negative_randn_corr == -1

    randn_corr = metrics.spearman_correlation(randn_tensor_1.flatten(), randn_tensor_2.flatten())
    assert randn_corr < positive_randn_corr and randn_corr > negative_randn_corr


def test_compute_2d_aggregate_metrics(randn_tensor_1, randn_tensor_2, aggregate_fn):
    # randn_tensor_1 vs randn_tensor_1
    positive_output_11 = metrics.compute_2d_aggregate_metrics(prediction_3d=randn_tensor_1,
                                                              targets_2d=randn_tensor_1.sum(dim=-1),
                                                              aggregate_fn=aggregate_fn)
    assert positive_output_11['rmse'] == 0
    assert positive_output_11['mae'] == 0
    assert positive_output_11['nrmse'] == 0
    assert positive_output_11['nmae'] == 0
    assert positive_output_11['corr'] == 1

    # -randn_tensor_1 vs randn_tensor_1
    negative_output_11 = metrics.compute_2d_aggregate_metrics(prediction_3d=-randn_tensor_1,
                                                              targets_2d=randn_tensor_1.sum(dim=-1),
                                                              aggregate_fn=aggregate_fn)
    expected_rmse = torch.sqrt(torch.square(randn_tensor_1.sum(dim=-1).mul(2)).mean())
    expected_mae = torch.abs(randn_tensor_1.sum(dim=-1).mul(2)).mean()
    q25, q75 = torch.quantile(randn_tensor_1.sum(dim=-1), q=torch.tensor([0.25, 0.75]))
    assert negative_output_11['rmse'] == expected_rmse.item()
    assert negative_output_11['mae'] == expected_mae.item()
    assert negative_output_11['nrmse'] == expected_rmse.div(q75 - q25).item()
    assert negative_output_11['nmae'] == expected_mae.div(q75 - q25).item()
    assert negative_output_11['corr'] == -1

    # randn_tensor_1 vs randn_tensor_2
    output_12 = metrics.compute_2d_aggregate_metrics(prediction_3d=randn_tensor_1,
                                                     targets_2d=randn_tensor_2.sum(dim=-1),
                                                     aggregate_fn=aggregate_fn)
    expected_rmse = torch.sqrt(torch.square(randn_tensor_1.sum(dim=-1) - randn_tensor_2.sum(dim=-1)).mean())
    expected_mae = torch.abs(randn_tensor_1.sum(dim=-1) - randn_tensor_2.sum(dim=-1)).mean()
    q25, q75 = torch.quantile(randn_tensor_2.sum(dim=-1), q=torch.tensor([0.25, 0.75]))
    assert output_12['rmse'] == expected_rmse.item()
    assert output_12['mae'] == expected_mae.item()
    assert output_12['nrmse'] == expected_rmse.div(q75 - q25).item()
    assert output_12['nmae'] == expected_mae.div(q75 - q25).item()
    assert output_12['corr'] > -1 and output_12['corr'] < 1


def test_compute_3d_isotropic_metrics(randn_tensor_1, randn_tensor_2):
    # randn_tensor_1 vs randn_tensor_1
    positive_output_11 = metrics.compute_3d_isotropic_metrics(prediction_3d=randn_tensor_1,
                                                              groundtruth_3d=randn_tensor_1)
    assert positive_output_11['rmse'] == 0
    assert positive_output_11['mae'] == 0
    assert positive_output_11['nrmse'] == 0
    assert positive_output_11['nmae'] == 0
    assert positive_output_11['corr'] == 1

    # -randn_tensor_1 vs randn_tensor_1
    negative_output_11 = metrics.compute_3d_isotropic_metrics(prediction_3d=-randn_tensor_1,
                                                              groundtruth_3d=randn_tensor_1)
    expected_rmse = torch.sqrt(torch.square(randn_tensor_1.mul(2)).mean())
    expected_mae = torch.abs(randn_tensor_1.mul(2)).mean()
    q25, q75 = torch.quantile(randn_tensor_1, q=torch.tensor([0.25, 0.75]))
    assert negative_output_11['rmse'] == expected_rmse.item()
    assert negative_output_11['mae'] == expected_mae.item()
    assert negative_output_11['nrmse'] == expected_rmse.div(q75 - q25).item()
    assert negative_output_11['nmae'] == expected_mae.div(q75 - q25).item()
    assert negative_output_11['corr'] == -1

    # randn_tensor_1 vs randn_tensor_2
    output_12 = metrics.compute_3d_isotropic_metrics(prediction_3d=randn_tensor_1,
                                                     groundtruth_3d=randn_tensor_2)
    expected_rmse = torch.sqrt(torch.square(randn_tensor_1 - randn_tensor_2).mean())
    expected_mae = torch.abs(randn_tensor_1 - randn_tensor_2).mean()
    q25, q75 = torch.quantile(randn_tensor_2, q=torch.tensor([0.25, 0.75]))
    assert output_12['rmse'] == expected_rmse.item()
    assert output_12['mae'] == expected_mae.item()
    assert output_12['nrmse'] == expected_rmse.div(q75 - q25).item()
    assert output_12['nmae'] == expected_mae.div(q75 - q25).item()
    assert output_12['corr'] > -1 and output_12['corr'] < 1


def test_compute_3d_vertical_metrics(randn_tensor_1, randn_tensor_2):
    # randn_tensor_1 vs randn_tensor_1
    positive_output_11 = metrics.compute_3d_vertical_metrics(prediction_3d=randn_tensor_1,
                                                             groundtruth_3d=randn_tensor_1)
    assert positive_output_11['rmse'] == 0
    assert positive_output_11['mae'] == 0
    assert positive_output_11['nrmse'] == 0
    assert positive_output_11['nmae'] == 0
    assert positive_output_11['corr'] == 1

    # -randn_tensor_1 vs randn_tensor_1
    negative_output_11 = metrics.compute_3d_vertical_metrics(prediction_3d=-randn_tensor_1,
                                                             groundtruth_3d=randn_tensor_1)
    expected_rmse = torch.sqrt(torch.square(randn_tensor_1.mul(2)).mean(dim=-1)).mean()
    expected_mae = torch.abs(randn_tensor_1.mul(2)).mean()
    q25, q75 = torch.quantile(randn_tensor_1, q=torch.tensor([0.25, 0.75]))
    assert negative_output_11['rmse'] == expected_rmse.item()
    assert negative_output_11['mae'] == expected_mae.item()
    assert negative_output_11['nrmse'] == expected_rmse.div(q75 - q25).item()
    assert negative_output_11['nmae'] == expected_mae.div(q75 - q25).item()
    assert negative_output_11['corr'] == -1

    # randn_tensor_1 vs randn_tensor_2
    output_12 = metrics.compute_3d_vertical_metrics(prediction_3d=randn_tensor_1,
                                                    groundtruth_3d=randn_tensor_2)
    expected_rmse = torch.sqrt(torch.square(randn_tensor_1 - randn_tensor_2).mean(dim=-1)).mean()
    expected_mae = torch.abs(randn_tensor_1 - randn_tensor_2).mean()
    q25, q75 = torch.quantile(randn_tensor_2, q=torch.tensor([0.25, 0.75]))
    assert output_12['rmse'] == expected_rmse.item()
    assert output_12['mae'] == expected_mae.item()
    assert output_12['nrmse'] == expected_rmse.div(q75 - q25).item()
    assert output_12['nmae'] == expected_mae.div(q75 - q25).item()
    assert output_12['corr'] > -1 and output_12['corr'] < 1
