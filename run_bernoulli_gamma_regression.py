"""
Description : Runs bernoulli gamma regression experiment

Usage: run_bernoulli_gamma_regression.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --device=<device_index>          Device to use [default: cpu]
  --plot                           Outputs plots.
"""
import os
import yaml
import logging
from docopt import docopt
from progress.bar import Bar
import torch
from src.preprocessing import make_data
from src.models import AggregateBernoulliGammaRegression
from src.evaluation import dump_scores, dump_plots, dump_state_dict
from src.preprocessing import make_2d_tensor, load_dataset


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    data = make_data(cfg=cfg, include_2d=False)

    # Add classification mask
    dataset = load_dataset(cfg['dataset']['path'])
    dataset = dataset.isel(lat=slice(30, 60), lon=slice(35, 55), time=slice(0, 3))
    pi = make_2d_tensor(dataset=dataset, variable_key='TAU_2D_550nm_mask').view(-1)
    data = data._replace(pi=pi)

    # Move needed tensors only to device
    data = migrate_to_device(data=data, device=device)

    # Instantiate model
    model = make_model(cfg=cfg, data=data)
    logging.info(f"{model}")

    # Fit model
    model = fit(model=model, data=data, cfg=cfg)
    logging.info("Fitted model")

    # Run prediction
    prediction_3d = predict(model=model, data=data)

    # Run evaluation
    evaluate(prediction_3d=prediction_3d, data=data, model=model, cfg=cfg, plot=args['--plot'], output_dir=args['--o'])


def migrate_to_device(data, device):
    # These are the only tensors needed on device to run this experiment
    data = data._replace(x_std=data.x_std.to(device),
                         z=data.z.to(device),
                         h_by_column=data.h_by_column.to(device))
    return data


def make_model(cfg, data):
    # Create aggregation operator
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column.unsqueeze(-1), dim=-2)
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
    model = AggregateBernoulliGammaRegression(transform=transform,
                                              aggregate_fn=trpz,
                                              ndim=len(cfg['dataset']['3d_covariates']) + 4,
                                              fit_intercept=cfg['model']['fit_intercept'])
    return model.to(device)


def fit(model, data, cfg):
    # Define optimizer and exact loglikelihood module
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg['training']['lr'])

    # Initialize progress bar
    n_epochs = cfg['training']['n_epochs']
    bar = Bar("Epoch", max=n_epochs)

    for epoch in range(n_epochs):
        # Zero-out remaining gradients
        optimizer.zero_grad()

        # Compute prediction
        predicted_mean = model.predict_mean(data.x_std)
        predicted_pi = model.predict_pi(data.x_std)
        predicted_mean_3d = predicted_mean.reshape(*data.h_by_column.shape)
        predicted_pi_3d = predicted_pi.reshape(*data.h_by_column.shape)
        aggregate_predicted_mean_2d = model.aggregate_prediction(predicted_mean_3d.unsqueeze(-1)).squeeze()
        aggregate_predicted_pi_2d = model.aggregate_prediction(predicted_pi_3d.unsqueeze(-1)).squeeze()
        aggregate_predicted_pi_2d.div_(data.h_by_column[:, 0])

        # Compute loss
        alpha, beta, pi = model.reparametrize(mu=aggregate_predicted_mean_2d, pi=aggregate_predicted_pi_2d)
        aggregate_prediction_2d = torch.distributions.gamma.Gamma(alpha, beta)
        log_prob_gamma = aggregate_prediction_2d.log_prob(data.z)
        bernoulli_1 = torch.log(pi) + log_prob_gamma
        bernoulli_0 = torch.log(1 - pi)
        log_prob = bernoulli_1.mul(data.pi) + bernoulli_0.mul(1 - data.pi)
        loss = -log_prob.sum()

        # Take gradient step
        loss.backward()
        optimizer.step()

        # Update progress bar
        bar.suffix = f"Loss {loss.item():e}"
        bar.next()

    return model


def predict(model, data):
    with torch.no_grad():
        # Predict on standardized 3D covariates
        prediction = model.predict_mean(data.x_std)

        # Reshape as (time * lat * lon, lev) grid
        prediction_3d = prediction.reshape(*data.h_by_column.shape)
    return prediction_3d


def evaluate(prediction_3d, data, model, cfg, plot, output_dir):
    # Define aggregation wrt non-standardized height for evaluation
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column.unsqueeze(-1).cpu(), dim=-2)
        return aggregated_grid

    # Dump scores in output dir
    dump_scores(prediction_3d=prediction_3d.cpu(),
                groundtruth_3d=data.gt_by_column,
                targets_2d=data.z,
                aggregate_fn=trpz,
                output_dir=output_dir)
    logging.info("Dumped scores")

    # Dump plots in output dir
    if plot:
        dump_plots(cfg=cfg,
                   dataset=data.dataset,
                   prediction_3d=prediction_3d.cpu(),
                   aggregate_fn=trpz,
                   output_dir=output_dir)
        logging.info("Dumped plots")

    # Dump model weights in output dir
    dump_state_dict(model=model, output_dir=output_dir)
    logging.info("Dumped weights")


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load config file
    with open(args['--cfg'], "r") as f:
        cfg = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Arguments: {args}\n')
    logging.info(f'Configuration file: {cfg}\n')

    # Create output directory if doesn't exists
    os.makedirs(args['--o'], exist_ok=True)
    with open(os.path.join(args['--o'], 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # Setup global variable for device
    if torch.cuda.is_available() and args['--device'].isdigit():
        device = torch.device(f"cuda:{args['--device']}")
    else:
        device = torch.device('cpu')

    # Run session
    main(args, cfg)
