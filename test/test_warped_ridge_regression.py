import yaml
import pytest
import torch
from progress.bar import Bar
from src.models import WarpedAggregateRidgeRegression
from src.evaluation import metrics
from test.toy import make_toy_data


@pytest.fixture(scope='module')
def cfg():
    cfg_path = 'test/toy/config/warped_ridge_regression.yaml'
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


@pytest.fixture(scope='module')
def toy_scores():
    scores_path = 'test/toy/scores/warped-ridge-regression/scores.metrics'
    with open(scores_path, 'r') as f:
        toy_scores = yaml.safe_load(f)
    return toy_scores


@pytest.fixture(scope='module')
def toy_data(cfg):
    toy_data = make_toy_data(cfg=cfg, include_2d=False)
    return toy_data


@pytest.fixture(scope='module')
def model(cfg, toy_data):
    # Create aggregation operator
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=toy_data.h.unsqueeze(-1), dim=-2)
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

    # Fix weights initialization seed
    torch.random.manual_seed(cfg['model']['seed'])

    # Instantiate model
    model = WarpedAggregateRidgeRegression(lbda=cfg['model']['lbda'],
                                           transform=transform,
                                           aggregate_fn=trpz,
                                           ndim=len(cfg['dataset']['3d_covariates']) + 4,
                                           fit_intercept=cfg['model']['fit_intercept'])
    return model


def fit(model, toy_data, cfg):
    # Define optimizer and exact loglikelihood module
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg['training']['lr'])

    # Initialize progress bar
    n_epochs = cfg['training']['n_epochs']
    bar = Bar("Epoch", max=n_epochs)

    for epoch in range(n_epochs):
        # Zero-out remaining gradients
        optimizer.zero_grad()

        # Compute prediction
        prediction = model(toy_data.x_std)
        prediction_3d = prediction.reshape(*toy_data.x_by_column_std.shape[:-1])
        aggregate_prediction_2d = model.aggregate_prediction(prediction_3d.unsqueeze(-1)).squeeze()

        # Compute loss
        loss = torch.square(aggregate_prediction_2d - toy_data.z).mean()
        loss += model.regularization_term()

        # Take gradient step
        loss.backward()
        optimizer.step()

        # Update progress bar
        bar.suffix = f"Loss {loss.item()}"
        bar.next()

    return model


def test_prediction(model, toy_data, toy_scores, cfg):
    # Fit model
    model = fit(model=model, toy_data=toy_data, cfg=cfg)

    # Run prediction
    with torch.no_grad():
        prediction = model(toy_data.x_std)
        prediction_3d = prediction.reshape(*toy_data.gt_grid.shape)

    # Compute scores
    scores = metrics.compute_scores(prediction_3d, toy_data.gt_grid, toy_data.z_grid, model.aggregate_fn)

    # Check if matches what is expected
    assert scores == toy_scores
