"""
Description : Runs sparse variational GP aerosol vertical profile reconstruction

Usage: run_svgp_product_kernel.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --device=<device_index>          Device to use [default: cpu]
  --seed=<seed>                    Random seed.
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
from src.models import AggregateLogNormalSVGP
from src.evaluation import dump_scores, dump_plots, dump_state_dict


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    data = make_data(cfg=cfg)

    # Move needed tensors only to device
    data = migrate_to_device(data=data, device=device)

    # Instantiate model
    model = make_model(cfg=cfg, data=data).to(device)
    logging.info(f"{model}")

    # Fit model
    logging.info("\n Fitting model")
    model = fit(model=model, data=data, cfg=cfg)

    # Run prediction
    prediction_3d_dist, bext_dist = predict(model=model, data=data, cfg=cfg)

    # Run evaluation
    evaluate(prediction_3d_dist=prediction_3d_dist, bext_dist=bext_dist, data=data, model=model, cfg=cfg, plot=args['--plot'], output_dir=args['--o'])


def migrate_to_device(data, device):
    # These are the only tensors needed on device to run this experiment
    data = data._replace(x_std=data.x_std.to(device),
                         x_by_column_std=data.x_by_column_std.to(device),
                         τ=data.τ.to(device),
                         τ_smooth=data.τ_smooth.to(device),
                         h_by_column=data.h_by_column.to(device),
                         h_by_column_std=data.h_by_column_std.to(device))

    return data


def make_model(cfg, data):
    # Create trapezoidal integation operator
    def trpz(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column_std.unsqueeze(-1), dim=-2)
        return aggregated_grid

    # Define GP kernel
    time_kernel = kernels.MaternKernel(nu=1.5, ard_num_dims=1, active_dims=[0])
    lat_kernel = kernels.MaternKernel(nu=1.5, ard_num_dims=1, active_dims=[2])
    lon_kernel = kernels.ScaleKernel(kernels.MaternKernel(nu=1.5, ard_num_dims=1, active_dims=[3]))
    pressure_kernel = kernels.MaternKernel(nu=0.5, ard_num_dims=1, active_dims=[4])
    st_kernel = kernels.MaternKernel(nu=0.5, ard_num_dims=1, active_dims=[5])
    relhum_kernel = kernels.MaternKernel(nu=0.5, ard_num_dims=1, active_dims=[6])
    omega_kernel = kernels.MaternKernel(nu=0.5, ard_num_dims=1, active_dims=[7])
    kernel = time_kernel * lat_kernel * lon_kernel * pressure_kernel * st_kernel * relhum_kernel * omega_kernel

    # Fix random seed for inducing points intialization
    torch.random.manual_seed(cfg['model']['seed'])

    # Initialize inducing points at low altitude levels
    lowaltitude_x_std = data.x_std[data.x_std[:, -1] < -cfg['model']['L']]
    rdm_idx = torch.randperm(len(lowaltitude_x_std))[:cfg['model']['n_inducing_points']]
    inducing_points = lowaltitude_x_std[rdm_idx].float()

    # Instantiate model
    model = AggregateLogNormalSVGP(inducing_points=inducing_points,
                                   transform=torch.exp,
                                   aggregate_fn=trpz,
                                   kernel=kernel)
    return model


def fit(model, data, cfg):
    # Define iterator
    def batch_iterator(batch_size):
        rdm_indices = torch.randperm(data.τ.size(0))
        for idx in rdm_indices.split(batch_size):
            x_by_column_std = data.x_by_column_std[idx]
            h_by_column_std = data.h_by_column_std[idx]
            τ = data.τ[idx]
            yield x_by_column_std, h_by_column_std, τ

    # Define optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg['training']['lr'])

    # Extract useful variables
    L = cfg['model']['L']
    n_epochs = cfg['training']['n_epochs']
    batch_size = cfg['training']['batch_size']
    n_samples = len(data.τ)

    # Initialize progress bar
    epoch_bar = Bar("Epoch", max=n_epochs)
    epoch_bar.finish()

    # Fix random seed for batch shuffling
    torch.random.manual_seed(cfg['training']['seed'])

    for epoch in range(n_epochs):

        batch_bar = Bar("Batch", max=n_samples // batch_size)
        epoch_ell, epoch_kl = 0, 0

        for i, (x_by_column_std, h_by_column_std, τ) in enumerate(batch_iterator(cfg['training']['batch_size'])):
            # Zero-out remaining gradients
            optimizer.zero_grad()

            # Compute variational posterior q(f)
            qf_by_column = model(x_by_column_std)

            # Draw a sample from the variational posterior
            fs = qf_by_column.lazy_covariance_matrix.zero_mean_mvn_samples(num_samples=1)
            fs = fs.add(qf_by_column.mean).squeeze()

            # Transform into extinction predictions and integrate
            predicted_means = model.transform(fs)
            predicted_means_3d = predicted_means.mul(torch.exp(-h_by_column_std / L))
            aggregate_predicted_means_2d = -torch.trapz(y=predicted_means_3d.unsqueeze(-1),
                                                        x=h_by_column_std.unsqueeze(-1),
                                                        dim=-2).squeeze()

            # Reparametrize into lognormal likelihood
            loc, scale = model.reparametrize(mu=aggregate_predicted_means_2d)
            aggregate_prediction_2d = torch.distributions.LogNormal(loc, scale)

            # Take logprob of τ to estimate expected log-likelihood
            log_prob = aggregate_prediction_2d.log_prob(τ)
            ell_MC = log_prob.mean()

            # Compute KL term and adjust for batch size
            kl_divergence = model.variational_strategy.kl_divergence().mul(batch_size / n_samples)

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


def predict(model, data, cfg):
    # Initialize empty tensor for lognormal locations and scales
    prediction_3d_locs = torch.zeros_like(data.h_by_column_std)
    prediction_3d_scales = torch.zeros_like(data.h_by_column_std)
    h_stddev = data.h_by_column.std().to(device)

    # Extract useful variables
    n_samples = len(data.τ)
    L = cfg['model']['L']

    # Setup index iteration and progress bar
    indices = torch.arange(len(data.x_by_column_std)).to(device)
    batch_size = cfg['evaluation']['batch_size']
    batch_bar = Bar("Inference", max=n_samples // batch_size)

    # Predict over batches of columns for computational efficiency
    with torch.no_grad():
        for idx in indices.split(batch_size):
            # Predict variational posterior on standardized 3D covariates
            qf_by_column = model(data.x_by_column_std[idx])

            # Register in grid
            prediction_3d_locs[idx] = qf_by_column.mean - data.h_by_column_std[idx] / L
            prediction_3d_scales[idx] = qf_by_column.stddev

            # Update progress bar
            batch_bar.suffix = f"{idx[-1]}/{n_samples} columns"
            batch_bar.next()

        # Make up for height standardization in integration
        prediction_3d_locs.sub_(torch.log(h_stddev))

        # Compute posterior mean E[φ|τ] from lognormal location and scale
        Eφ_τ = torch.exp(prediction_3d_locs + 0.5 * prediction_3d_scales.square())

        # Rescale predictions by τ/∫E[φ|τ]dh
        def trpz(grid):
            aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column.unsqueeze(-1), dim=-2)
            return aggregated_grid
        aggregate_Eφ_τ = trpz(Eφ_τ.unsqueeze(-1)).squeeze()
        correction = torch.log(data.τ_smooth) - torch.log(aggregate_Eφ_τ)
        prediction_3d_locs.add_(correction.unsqueeze(-1))

        # Make latent vertical profile φ distribution
        prediction_3d_dist = torch.distributions.LogNormal(prediction_3d_locs.cpu(), prediction_3d_scales.cpu())

        # Make bext observation model distribution from φ
        φ = prediction_3d_dist.sample((cfg['evaluation']['n_samples'],))
        sigma_ext = torch.tensor(cfg['evaluation']['sigma_ext'])
        loc = torch.log(φ.clip(min=torch.finfo(torch.float64).eps)) - sigma_ext.square().div(2)
        bext_dist = torch.distributions.LogNormal(loc, sigma_ext)
    return prediction_3d_dist, bext_dist


def evaluate(prediction_3d_dist, bext_dist, data, model, cfg, plot, output_dir):
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
                   bext_dist=bext_dist,
                   sigma=model.sigma.detach().cpu(),
                   aggregate_fn=trpz,
                   output_dir=output_dir)
        logging.info("Dumped plots")

    # Dump scores in output dir
    dump_scores(cfg=cfg,
                prediction_3d_dist=prediction_3d_dist,
                bext_dist=bext_dist,
                groundtruth_3d=data.gt_by_column,
                output_dir=output_dir)
    logging.info("Dumped scores")


def update_cfg(cfg, args):
    if args['--seed']:
        cfg['model']['seed'] = int(args['--seed'])
        cfg['training']['seed'] = int(args['--seed'])
    return cfg


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load config file
    with open(args['--cfg'], "r") as f:
        cfg = yaml.safe_load(f)
    cfg = update_cfg(cfg, args)

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
