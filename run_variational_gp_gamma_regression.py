"""
Description : Runs variational GP gamma regression experiment

Usage: run_variational_gp_gamma_regression.py  [options] --cfg=<path_to_config> --o=<output_dir>

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
from gpytorch import kernels, constraints
from src.preprocessing import make_data
from src.models import AggregateVariationalGPGammaRegression
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

    # Run prediction
    prediction_3d_dist = predict(model=model, data=data)

    # Run evaluation
    evaluate(prediction_3d_dist=prediction_3d_dist, data=data, model=model, cfg=cfg, plot=args['--plot'], output_dir=args['--o'])


def migrate_to_device(data, device):
    # These are the only tensors needed on device to run this experiment
    data = data._replace(x_std=data.x_std.to(device),
                         x_by_column_std=data.x_by_column_std.to(device),
                         z=data.z.to(device),
                         h_by_column_std=data.h_by_column_std.to(device))
    return data


def make_model(cfg, data):
    # Create aggregation operator
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column_std.unsqueeze(-1), dim=-2)
        return aggregated_grid

    # Define kernel
    height_kernel = kernels.ScaleKernel(kernels.RBFKernel(active_dims=[1]),
                                        outputscale_constraint=constraints.GreaterThan(0))
    covariates_kernel = kernels.ScaleKernel(kernels.RBFKernel(ard_num_dims=2, active_dims=[4, 5]),
                                            outputscale_constraint=constraints.GreaterThan(0))
    kernel = height_kernel + covariates_kernel

    # Fix initialization seed
    torch.random.manual_seed(cfg['model']['seed'])

    # Initialize inducing points regularly across grid
    rdm_idx = torch.randperm(len(data.x_std))[:cfg['model']['n_inducing_points']]
    inducing_points = data.x_std[rdm_idx].float()

    # Instantiate model
    model = AggregateVariationalGPGammaRegression(inducing_points=inducing_points,
                                                  transform=torch.exp,
                                                  aggregate_fn=trpz,
                                                  kernel=kernel,
                                                  ndim=len(cfg['dataset']['3d_covariates']) + 4,
                                                  fit_intercept=cfg['model']['fit_intercept'])
    return model.to(device)


def fit(model, data, cfg):
    # Define iterator
    def batch_iterator(batch_size):
        rdm_indices = torch.randperm(data.z.size(0))
        for idx in rdm_indices.split(batch_size):
            x_by_column_std = data.x_by_column_std[idx]
            h_by_column_std = data.h_by_column_std[idx]
            z = data.z[idx]
            yield x_by_column_std, h_by_column_std, z

    # Define optimizer and exact loglikelihood module
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg['training']['lr'])

    # Initialize progress bar
    n_epochs = cfg['training']['n_epochs']
    batch_size = cfg['training']['batch_size']
    n_MC_samples = cfg['training']['n_samples']
    n_samples = len(data.z)
    epoch_bar = Bar("Epoch", max=n_epochs)
    epoch_bar.finish()
    torch.random.manual_seed(cfg['training']['seed'])

    for epoch in range(n_epochs):

        batch_bar = Bar("Batch", max=n_samples // batch_size)
        epoch_ell, epoch_kl = 0, 0

        for i, (x_by_column_std, h_by_column_std, z) in enumerate(batch_iterator(cfg['training']['batch_size'])):
            # Zero-out remaining gradients
            optimizer.zero_grad()

            # Compute q(f)
            qf_by_column = model(x_by_column_std)

            # Draw a bunch of samples from the variational posterior
            fs = qf_by_column.lazy_covariance_matrix.zero_mean_mvn_samples(num_samples=n_MC_samples)
            fs = fs.add(qf_by_column.mean)

            # Transform into aggregate predictions
            predicted_means = model.transform(fs)
            predicted_means_3d = predicted_means.reshape((n_MC_samples,) + h_by_column_std.shape)
            predicted_means_3d = predicted_means_3d.mul(torch.exp(-1.8 * h_by_column_std))
            aggregate_predicted_means_2d = -torch.trapz(y=predicted_means_3d.unsqueeze(-1),
                                                        x=h_by_column_std.tile((n_MC_samples, 1, 1)).unsqueeze(-1),
                                                        dim=-2).squeeze()

            # Reparametrize into gamma logprob
            alpha, beta = model.reparametrize(mu=aggregate_predicted_means_2d)
            aggregate_prediction_2d = torch.distributions.gamma.Gamma(alpha, beta)
            prob_gamma = aggregate_prediction_2d.log_prob(z.tile((n_MC_samples, 1))).exp()
            ell_MC = torch.log(prob_gamma.mean(dim=0)).mean()

            # Compute KL term
            kl_divergence = model.variational_strategy.kl_divergence().div(n_samples)

            # Take gradient step
            loss = kl_divergence - ell_MC
            loss.backward()
            optimizer.step()

            # Update progress bar
            epoch_ell += ell_MC.item()
            epoch_kl += kl_divergence.item()
            batch_bar.suffix = f"ELL {epoch_ell / (i + 1):e} | KL {epoch_kl / (i + 1):e}"
            batch_bar.next()

        # Complete progress bar
        epoch_bar.next()
        epoch_bar.finish()
    return model


def predict(model, data):
    with torch.no_grad():
        # Predict on standardized 3D covariates
        qf_by_column = model(data.x_by_column_std)
        fs = qf_by_column.lazy_covariance_matrix.zero_mean_mvn_samples(num_samples=2000)
        fs = fs.add(qf_by_column.mean)
        prediction = model.transform(fs).div(data.h_by_column.std())

        # Reshape as (time * lat * lon, lev) grid
        prediction_3d = prediction.mul(torch.exp(-1.8 * data.h_by_column_std)).mean(dim=0)

        # Make distribution
        alpha, beta = model.reparametrize(prediction_3d)
        prediction_3d_dist = torch.distributions.Gamma(alpha, beta)
    return prediction_3d_dist


def evaluate(prediction_3d_dist, data, model, cfg, plot, output_dir):
    # Define aggregation wrt non-standardized height for evaluation
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column.unsqueeze(-1).cpu(), dim=-2)
        return aggregated_grid

    # Dump model weights in output dir
    dump_state_dict(model=model, output_dir=output_dir)
    logging.info("Dumped weights")

    # Dump plots in output dir
    if plot:
        dump_plots(cfg=cfg,
                   dataset=data.dataset,
                   prediction_3d_dist=prediction_3d_dist,
                   aggregate_fn=trpz,
                   output_dir=output_dir)
        logging.info("Dumped plots")

    # Dump scores in output dir
    dump_scores(prediction_3d=prediction_3d_dist.mean.cpu(),
                groundtruth_3d=data.gt_by_column,
                targets_2d=data.z,
                aggregate_fn=trpz,
                output_dir=output_dir)
    logging.info("Dumped scores")


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