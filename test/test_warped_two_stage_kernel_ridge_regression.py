import yaml
import pytest
import torch
from progress.bar import Bar
from src.models import WarpedTwoStageAggregateKernelRidgeRegression
from src.kernels import RFFKernel
from src.evaluation import metrics
from test.toy import make_toy_data


"""
PATHS FIXTURES
"""


@pytest.fixture(scope='module')
def toy_cfg_path():
    cfg_path = 'test/toy/config/warped_two_stage_kernel_ridge_regression.yaml'
    return cfg_path


@pytest.fixture(scope='module')
def toy_scores_path():
    scores_path = 'test/toy/outputs/warped-two-stage-kernel-ridge-regression/scores.metrics'
    return scores_path


@pytest.fixture(scope='module')
def toy_state_dict_path():
    state_dict_path = 'test/toy/outputs/warped-two-stage-kernel-ridge-regression/state_dict.pt'
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
    # Create aggregation operator over standardized heights
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h.unsqueeze(-1), dim=-2)
        return aggregated_grid

    # Define warping transformation
    if cfg['model']['transform'] == 'linear':
        transform = lambda x: x
    elif cfg['model']['transform'] == 'softplus':
        transform = lambda x: torch.log(1 + torch.exp(x))
    elif cfg['model']['transform'] == 'smooth_abs':
        transform = lambda x: torch.nn.functional.smooth_l1_loss(x, torch.zeros_like(x), reduction='none')
    elif cfg['model']['transform'] == 'square':
        transform = torch.square
    elif cfg['model']['transform'] == 'exp':
        transform = torch.exp
    else:
        raise ValueError("Unknown transform")

    # Initialize RFF 2D covariates kernel
    ard_num_dims = len(cfg['dataset']['2d_covariates']) + 3
    kernel_2d = RFFKernel(nu=cfg['model']['nu_2d'],
                          num_samples=cfg['model']['num_samples_2d'],
                          ard_num_dims=ard_num_dims)
    kernel_2d.lengthscale = cfg['model']['lengthscale_2d'] * torch.ones(ard_num_dims)
    kernel_2d.raw_lengthscale.requires_grad = False

    # Initialize RFF 3D covariates kernel
    ard_num_dims = len(cfg['dataset']['3d_covariates']) + 4
    kernel_3d = RFFKernel(nu=cfg['model']['nu_3d'],
                          num_samples=cfg['model']['num_samples_3d'],
                          ard_num_dims=ard_num_dims)
    kernel_3d.lengthscale = cfg['model']['lengthscale_3d'] * torch.ones(ard_num_dims)
    kernel_3d.raw_lengthscale.requires_grad = False

    # Fix weights initialization seed
    torch.random.manual_seed(cfg['model']['seed'])

    # Instantiate model
    model = WarpedTwoStageAggregateKernelRidgeRegression(kernel_2d=kernel_2d,
                                                         kernel_3d=kernel_3d,
                                                         training_covariates_3d=data.x_std,
                                                         training_covariates_2d=data.y_std,
                                                         lbda_2d=cfg['model']['lbda_2d'],
                                                         lbda_3d=cfg['model']['lbda_3d'],
                                                         transform=transform,
                                                         aggregate_fn=trpz)
    return model


def fit(model, data, cfg):
    # Define optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg['training']['lr'])

    # Initialize progress bar
    n_epochs = cfg['training']['n_epochs']
    bar = Bar("Epoch", max=n_epochs)

    for epoch in range(n_epochs):
        # Zero-out remaining gradients
        optimizer.zero_grad()

        # Compute prediction
        prediction = model(data.x_std)
        prediction_3d = prediction.reshape(*data.x_by_column_std.shape[:-1])
        aggregate_prediction_2d = model.aggregate_prediction(prediction_3d.unsqueeze(-1))

        # Compute loss
        loss = torch.square(aggregate_prediction_2d - data.z).mean()
        loss += model.regularization_term()

        # Take gradient step
        loss.backward()
        optimizer.step()

        # Update progress bar
        bar.suffix = f"Loss {loss.item():e}"
        bar.next()

    return model


@pytest.fixture(scope='module')
def fitted_model(toy_cfg, toy_data):
    model = make_model(cfg=toy_cfg, data=toy_data)
    fitted_model = fit(model=model, data=toy_data, cfg=toy_cfg)
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


def test_prediction(fitted_model, toy_data, toy_scores, toy_cfg):
    # Run prediction
    with torch.no_grad():
        prediction = fitted_model(toy_data.x_std)
        prediction_3d = prediction.reshape(*toy_data.gt_grid.shape)

    # Compute scores
    scores = metrics.compute_scores(prediction_3d, toy_data.gt_grid, toy_data.z_grid, fitted_model.aggregate_fn)

    # Check if matches what is expected
    assert scores == toy_scores
