"""
Description : Runs gamma regression experiment

Usage: run_gamma_map_gp_regression.py  [options] --cfg=<path_to_config> --o=<output_dir>

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
from gpytorch import kernels
from src.preprocessing import make_data
from src.models import AggregateMAPGPGammaRegression
from src.evaluation import dump_scores, dump_plots, dump_state_dict


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    data = make_data(cfg=cfg, include_2d=False)

    # Move needed tensors only to device
    data = migrate_to_device(data=data, device=device)

    # Instantiate model
    model = make_model(cfg=cfg, data=data)
    logging.info(f"{model}")

    # Fit model
    logging.info("\n Fitting model")
    model = fit(model=model, data=data, cfg=cfg)

    # Fit MAP model
    logging.info("\n Fitting MAP estimate")
    model = fitMAP(model=model, data=data, cfg=cfg)

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

    # Define kernel
    # height_kernel = kernels.ScaleKernel(kernels.MaternKernel(nu=1.5, active_dims=[1]))
    covariates_kernel = kernels.ScaleKernel(kernels.MaternKernel(nu=1.5, ard_num_dims=2, active_dims=[4, 5]))
    kernel = covariates_kernel

    # Fix weights initialization seed
    torch.random.manual_seed(cfg['model']['seed'])

    # Instantiate model
    model = AggregateMAPGPGammaRegression(x=data.h_by_column,
                                          transform=transform,
                                          aggregate_fn=trpz,
                                          kernel=kernel,
                                          ndim=len(cfg['dataset']['3d_covariates']) + 4,
                                          fit_intercept=cfg['model']['fit_intercept'])
    return model.to(device)


def fit(model, data, cfg):
    # Aggregation operator for MC approximation
    n_MC_samples = cfg['training']['n_samples']

    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column.tile((n_MC_samples, 1, 1)).unsqueeze(-1), dim=-2)
        return aggregated_grid

    # Define optimizer and exact loglikelihood module
    model.fMAP.requires_grad = False
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg['training']['lr'])

    # Initialize progress bar
    n_epochs = cfg['training']['n_epochs']
    bar = Bar("Epoch", max=n_epochs)
    torch.random.manual_seed(cfg['training']['seed'])

    for epoch in range(n_epochs):
        # Zero-out remaining gradients
        optimizer.zero_grad()

        # Draw multiple GP samples for each column and aggregate
        K = model.kernel(data.x_by_column_std)
        fs = K.zero_mean_mvn_samples(num_samples=n_MC_samples)
        predicted_means = model.transform(model.bias + fs)
        predicted_means_3d = predicted_means.reshape((n_MC_samples,) + data.h_by_column.shape)
        aggregate_predicted_means_2d = trpz(predicted_means_3d.unsqueeze(-1)).squeeze()

        # Reparametrize into gamma logprob
        alpha, beta = model.reparametrize(mu=aggregate_predicted_means_2d)
        aggregate_prediction_2d = torch.distributions.gamma.Gamma(alpha, beta)
        prob_gamma = aggregate_prediction_2d.log_prob(data.z.tile((n_MC_samples, 1))).exp()
        mc_log_prob_tau = torch.log(prob_gamma.mean(dim=0))

        # Take gradient step
        loss = -mc_log_prob_tau.sum()
        loss.backward()
        optimizer.step()

        # Update progress bar
        bar.suffix = f"Loss {loss.item():e}"
        bar.next()
    return model


def fitMAP(model, data, cfg):
    # Define optimizer and exact loglikelihood module
    model.fMAP.requires_grad = True
    optimizer = torch.optim.Adam(params=[model.fMAP], lr=cfg['MAP']['lr'])

    # Initialize progress bar
    n_epochs = cfg['MAP']['n_epochs']
    bar = Bar("Epoch", max=n_epochs)

    for epoch in range(n_epochs):
        # Zero-out remaining gradients
        optimizer.zero_grad()

        # Compute prediction
        predicted_mean = model.transform(model.bias + model.fMAP)
        predicted_mean_3d = predicted_mean.reshape(*data.h_by_column.shape)
        aggregate_predicted_mean_2d = model.aggregate_prediction(predicted_mean_3d.unsqueeze(-1)).squeeze()

        # Compute gamma logprob
        alpha, beta = model.reparametrize(mu=aggregate_predicted_mean_2d)
        aggregate_prediction_2d = torch.distributions.gamma.Gamma(alpha, beta)
        log_prob_gamma = aggregate_prediction_2d.log_prob(data.z).sum()

        # Compute prior logprob
        K = model.kernel(data.x_by_column_std)
        inv_quad = K.inv_quad(model.fMAP.unsqueeze(-1))
        log_prob_prior = -0.5 * inv_quad.sum()

        # Take gradient step
        loss = -log_prob_gamma - log_prob_prior
        loss.backward()
        optimizer.step()

        # Update progress bar
        bar.suffix = f"Loss {loss.item():e}"
        bar.next()

    return model


def predict(model, data):
    with torch.no_grad():
        # Predict on standardized 3D covariates
        prediction = model.transform(model.bias + model.fMAP)

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
