import yaml
import pytest
import torch
from src.models import TwoStageAggregateRidgeRegression
from src.evaluation import metrics
from test.toy import make_toy_data


"""
PATHS FIXTURES
"""


@pytest.fixture(scope='module')
def toy_cfg_path():
    cfg_path = 'test/toy/config/two_stage_ridge_regression.yaml'
    return cfg_path


@pytest.fixture(scope='module')
def toy_scores_path():
    scores_path = 'test/toy/outputs/two-stage-ridge-regression/scores.metrics'
    return scores_path


@pytest.fixture(scope='module')
def toy_state_dict_path():
    state_dict_path = 'test/toy/outputs/two-stage-ridge-regression/state_dict.pt'
    return state_dict_path


"""
OBJECTS FIXTURES
"""


@pytest.fixture(scope='module')
def toy_cfg(toy_cfg_path):
    with open(toy_cfg_path, 'r') as f:
        toy_cfg = yaml.safe_load(f)
    return toy_cfg


@pytest.fixture(scope='module')
def toy_scores(toy_scores_path):
    with open(toy_scores_path, 'r') as f:
        toy_scores = yaml.safe_load(f)
    return toy_scores


@pytest.fixture(scope='module')
def toy_state_dict(toy_state_dict_path):
    toy_state_dict = torch.load(toy_state_dict_path)
    return toy_state_dict


@pytest.fixture(scope='module')
def toy_data(toy_cfg):
    toy_data = make_toy_data(cfg=toy_cfg, include_2d=True)
    return toy_data


"""
TESTING MODEL
"""


def make_model(cfg, data):
    # Create aggregation operator
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h_std.unsqueeze(-1), dim=-2)
        return aggregated_grid

    # Instantiate model
    model = TwoStageAggregateRidgeRegression(lbda_2d=cfg['model']['lbda_2d'],
                                             lbda_3d=cfg['model']['lbda_3d'],
                                             aggregate_fn=trpz,
                                             fit_intercept_2d=cfg['model']['fit_intercept_2d'],
                                             fit_intercept_3d=cfg['model']['fit_intercept_3d'])

    return model


def fit(model, data):
    model.fit(data.x_by_column_std, data.y_std, data.z_std)
    return model


@pytest.fixture(scope='module')
def fitted_model(toy_cfg, toy_data):
    model = make_model(cfg=toy_cfg, data=toy_data)
    fitted_model = fit(model, toy_data)
    return fitted_model


"""
TESTS
"""


def test_parameters(fitted_model, toy_state_dict):
    # Extract state dict from fitted model
    fitted_state_dict = fitted_model.state_dict()

    # Check if models match
    for (fitted_name, fitted_param), (toy_name, toy_param) in zip(fitted_state_dict.items(), toy_state_dict.items()):
        assert fitted_name == toy_name
        assert torch.equal(fitted_param, toy_param)



def test_prediction(fitted_model, toy_data, toy_scores):
    # Run prediction
    with torch.no_grad():
        prediction = fitted_model(toy_data.x_std)
        prediction_3d_std = prediction.reshape(*toy_data.gt_grid.shape)
        prediction_3d = toy_data.z_grid.std() * (prediction_3d_std + toy_data.z_grid.mean()) / toy_data.h.std()

    # Define aggregation wrt height with trapezoidal rule
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=toy_data.h.unsqueeze(-1), dim=-2)
        return aggregated_grid

    scores = metrics.compute_scores(prediction_3d, toy_data.gt_grid, toy_data.z_grid, trpz)
    assert scores == toy_scores
