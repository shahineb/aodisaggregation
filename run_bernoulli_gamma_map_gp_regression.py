"""
Description : Runs bernoulli gamma MAP gp regression experiment

Usage: run_bernoulli_gamma_map_gp_regression.py  [options] --cfg=<path_to_config> --o=<output_dir>

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
from src.models import AggregateMAPGPBernoulliGammaRegression
from src.evaluation import dump_scores, dump_plots, dump_state_dict
from src.preprocessing import make_2d_tensor, load_dataset


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    data = make_data(cfg=cfg, include_2d=False)

    # Add classification mask
    dataset = load_dataset(cfg['dataset']['path'])
    dataset = dataset.isel(lat=slice(30, 60), lon=slice(35, 55), time=slice(0, 3))
    mask = make_2d_tensor(dataset=dataset, variable_key='TAU_2D_550nm_mask').view(-1)
    data = data._replace(mask=mask)

    # Move needed tensors only to device
    data = migrate_to_device(data=data, device=device)

    # Instantiate model
    model = make_model(cfg=cfg, data=data)
    logging.info(f"{model}")

    # Fit classification model parameters
    logging.info("\n Fitting classification model")
    model = fitg(model=model, data=data, cfg=cfg)

    # Fit classification model MAP
    logging.info("\n Fitting classification MAP estimate")
    model = fitgMAP(model=model, data=data, cfg=cfg)

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
        aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column_std.unsqueeze(-1), dim=-2)
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

    # Define kernels
    covariates_kernel_f = kernels.ScaleKernel(kernels.MaternKernel(nu=1.5, ard_num_dims=2, active_dims=[4, 5]))
    kernel_f = covariates_kernel_f

    covariates_kernel_g = kernels.RBFKernel(active_dims=[4])
    kernel_g = covariates_kernel_g

    # Fix weights initialization seed
    torch.random.manual_seed(cfg['model']['seed'])

    # Instantiate model
    model = AggregateMAPGPBernoulliGammaRegression(x=data.h_by_column,
                                                   transform=transform,
                                                   aggregate_fn=trpz,
                                                   kernel_f=kernel_f,
                                                   kernel_g=kernel_g,
                                                   ndim=len(cfg['dataset']['3d_covariates']) + 4,
                                                   fit_intercept=cfg['model']['fit_intercept'])
    return model.train().to(device)


def fitg(model, data, cfg):
    # Aggregation operator for MC approximation
    n_MC_samples = cfg['training']['n_samples']

    def trpz_MC(grid):
        aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column_std.tile((n_MC_samples, 1, 1)).unsqueeze(-1), dim=-2)
        return aggregated_grid

    # Define optimizer and exact loglikelihood module
    model.fMAP.requires_grad = False
    model.gMAP.requires_grad = False
    model.bias_fMAP.requires_grad = False
    model.bias_gMAP.requires_grad = False
    optimizer = torch.optim.LBFGS(params=model.parameters(), lr=cfg['training']['lr'])

    # Initialize progress bar
    n_epochs = cfg['training']['n_epochs']
    bar = Bar("Epoch", max=n_epochs)
    torch.random.manual_seed(cfg['training']['seed'])

    for epoch in range(n_epochs):
        def closure():
            # Zero-out remaining gradients
            optimizer.zero_grad()

            # Draw multiple GP samples for each column and aggregate
            Kg = model.kernel_g(data.x_by_column_std).add_jitter(1e-6)
            gs = Kg.zero_mean_mvn_samples(num_samples=n_MC_samples)
            predicted_3d_pis = torch.sigmoid(gs)
            pis = predicted_3d_pis.max(dim=-1).values

            # Reparametrize into gamma logprob
            prob = pis * data.mask + (1 - pis) * (1 - data.mask)
            mc_log_prob = torch.log(prob.mean(dim=0))

            # Take gradient step
            loss = -mc_log_prob.sum()
            loss.backward()

            # Update progress bar
            bar.suffix = f"Loss {loss.item():e}"
            return loss
        optimizer.step(closure)
        bar.next()
    return model


def fitgMAP(model, data, cfg):
    # Define optimizer and exact loglikelihood module
    for p in model.parameters():
        p.requires_grad = False
    model.gMAP.requires_grad = True
    model.bias_gMAP.requires_grad = True
    optimizer = torch.optim.LBFGS(params=[model.gMAP], lr=cfg['MAP']['lr'])

    # Initialize progress bar
    n_epochs = cfg['MAP']['n_epochs']
    bar = Bar("Epoch", max=n_epochs)

    for epoch in range(n_epochs):
        def closure():
            # Zero-out remaining gradients
            optimizer.zero_grad()

            # Compute prediction
            predicted_pi = torch.sigmoid(model.gMAP)
            pi = predicted_pi.max(dim=-1).values

            # Compute gamma logprob
            log_prob = torch.log(pi) * data.mask + torch.log(1 - pi) * (1 - data.mask)
            log_prob_mask = log_prob.sum()

            # Compute prior logprob
            Kg = model.kernel_g(data.x_by_column_std)
            inv_quad_g = Kg.inv_quad(model.gMAP.unsqueeze(-1))
            log_prob_prior = -0.005 * inv_quad_g.sum()

            # Take gradient step
            loss = -log_prob_mask - log_prob_prior
            loss.backward()

            # Update progress bar
            bar.suffix = f"log N(g|m,K) {log_prob_prior.item():e} | log Ber(mask|g) {log_prob_mask.item():e}"
            return loss
        optimizer.step(closure)
        bar.next()
    return model


def predict(model, data):
    with torch.no_grad():
        # Predict on standardized 3D covariates
        prediction = torch.sigmoid(model.gMAP)

        # Reshape as (time * lat * lon, lev) grid
        prediction_3d = prediction.reshape(*data.h_by_column.shape)
    return prediction_3d


def evaluate(prediction_3d, data, model, cfg, plot, output_dir):
    # Define aggregation wrt non-standardized height for evaluation
    def trpz(grid):
        # aggregated_grid = -torch.trapz(y=grid, x=data.h_by_column.unsqueeze(-1).cpu(), dim=-2).squeeze()
        # aggregated_grid.div_(data.h_by_column_std[:, 0] - data.h_by_column_std[:, -1])
        aggregated_grid = grid.max(dim=-2).values
        return aggregated_grid

    # Dump model weights in output dir
    dump_state_dict(model=model, output_dir=output_dir)
    logging.info("Dumped weights")

    # Dump plots in output dir
    if plot:
        dump_plots(cfg=cfg,
                   dataset=data.dataset,
                   prediction_3d=prediction_3d.cpu(),
                   aggregate_fn=trpz,
                   output_dir=output_dir)
        logging.info("Dumped plots")

    # Dump scores in output dir
    dump_scores(prediction_3d=prediction_3d.cpu(),
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
