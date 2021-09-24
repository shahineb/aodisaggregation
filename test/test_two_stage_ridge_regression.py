import yaml
import pytest
import torch
from src.models import TwoStageAggregateRidgeRegression
from src.evaluation import metrics
from test.toy import make_2d_and_3d_toy_data


@pytest.fixture(scope='module')
def cfg():
    cfg_path = 'test/toy/config/two_staged_ridge_regression.yaml'
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


@pytest.fixture(scope='module')
def toy_scores():
    scores_path = 'test/toy/scores/two-stage-ridge-regression/scores.metrics'
    with open(scores_path, 'r') as f:
        toy_scores = yaml.safe_load(f)
    return toy_scores


@pytest.fixture(scope='module')
def toy_data(cfg):
    toy_data = make_2d_and_3d_toy_data(cfg=cfg)
    return toy_data


@pytest.fixture(scope='module')
def dataset(toy_data):
    return toy_data.dataset


@pytest.fixture(scope='module')
def standard_dataset(toy_data):
    return toy_data.standard_dataset


@pytest.fixture(scope='module')
def x_by_column_std(toy_data):
    return toy_data.x_by_column_std


@pytest.fixture(scope='module')
def x_std(toy_data):
    return toy_data.x_std


@pytest.fixture(scope='module')
def y_grid_std(toy_data):
    return toy_data.y_grid_std


@pytest.fixture(scope='module')
def y_std(toy_data):
    return toy_data.y_std


@pytest.fixture(scope='module')
def z_grid(toy_data):
    return toy_data.z_grid


@pytest.fixture(scope='module')
def z_std(toy_data):
    return toy_data.z_std


@pytest.fixture(scope='module')
def gt_grid(toy_data):
    return toy_data.gt_grid


@pytest.fixture(scope='module')
def h(toy_data):
    return toy_data.h


@pytest.fixture(scope='module')
def h_std(toy_data):
    return toy_data.h_std


@pytest.fixture(scope='module')
def model(cfg, h_std, x_by_column_std, y_std, z_std):
    # Create aggregation operator
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=h_std.unsqueeze(-1), dim=-2)
        return aggregated_grid

    # Instantiate model
    model = TwoStageAggregateRidgeRegression(lbda_2d=cfg['model']['lbda_2d'],
                                             lbda_3d=cfg['model']['lbda_3d'],
                                             aggregate_fn=trpz,
                                             fit_intercept_2d=cfg['model']['fit_intercept_2d'],
                                             fit_intercept_3d=cfg['model']['fit_intercept_3d'])

    # Fit it
    model.fit(x_by_column_std, y_std, z_std)
    return model


def test_prediction(dataset, model, x_std, gt_grid, z_grid, h, toy_scores):
    # Run prediction
    with torch.no_grad():
        prediction = model(x_std)
        prediction_3d_std = prediction.reshape(*gt_grid.shape)
        prediction_3d = z_grid.std() * (prediction_3d_std + z_grid.mean()) / h.std()

    # Define aggregation wrt height with trapezoidal rule
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=h.unsqueeze(-1), dim=-2)
        return aggregated_grid

    scores = metrics.compute_scores(prediction_3d, gt_grid, z_grid, trpz)
    assert scores == toy_scores
