import yaml
import pytest
import torch
from src.models import TwoStageAggregateRidgeRegression
from src.evaluation import metrics
from test.toy import make_toy_data


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
    toy_data = make_toy_data(cfg=cfg, include_2d=True)
    return toy_data


@pytest.fixture(scope='module')
def model(cfg, toy_data):
    # Create aggregation operator
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=toy_data.h_std.unsqueeze(-1), dim=-2)
        return aggregated_grid

    # Instantiate model
    model = TwoStageAggregateRidgeRegression(lbda_2d=cfg['model']['lbda_2d'],
                                             lbda_3d=cfg['model']['lbda_3d'],
                                             aggregate_fn=trpz,
                                             fit_intercept_2d=cfg['model']['fit_intercept_2d'],
                                             fit_intercept_3d=cfg['model']['fit_intercept_3d'])

    # Fit it
    model.fit(toy_data.x_by_column_std, toy_data.y_std, toy_data.z_std)
    return model


def test_prediction(model, toy_data, toy_scores):
    # Run prediction
    with torch.no_grad():
        prediction = model(toy_data.x_std)
        prediction_3d_std = prediction.reshape(*toy_data.gt_grid.shape)
        prediction_3d = toy_data.z_grid.std() * (prediction_3d_std + toy_data.z_grid.mean()) / toy_data.h.std()

    # Define aggregation wrt height with trapezoidal rule
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=toy_data.h.unsqueeze(-1), dim=-2)
        return aggregated_grid

    scores = metrics.compute_scores(prediction_3d, toy_data.gt_grid, toy_data.z_grid, trpz)
    assert scores == toy_scores
