import torch
import torch.nn as nn
import gpytorch


class AggregateRidgeRegression(nn.Module):
    """Ridge Regression model when aggregated targets only are observed

        *** Current implementation assumes all columns have same size ***

    Args:
        lbda (float): regularization weight, greater = stronger L2 penalization
        aggregate_fn (callable): aggregation operator
        fit_intercept (bool): if True, pads inputs with constant offset
    """

    def __init__(self, lbda, aggregate_fn, fit_intercept=False):
        super().__init__()
        self.lbda = lbda
        self.aggregate_fn = aggregate_fn
        self.fit_intercept = fit_intercept

    def pad_input(self, x):
        """Pads x with 1 along last dimension

        Args:
            x (torch.Tensor)

        Returns:
            type: torch.Tensor

        """
        x = torch.cat([x, torch.ones(x.shape[:-1], device=x.device).unsqueeze(-1)], dim=-1)
        return x

    def fit(self, individuals_covariates, aggregate_targets):
        """Fits model following sklearn syntax

        Args:
            individuals_covariates (torch.Tensor): (n_bags, bags_size, n_dim_individuals)
                samples must be organized by bags following which aggregation is taken
            aggregate_targets (torch.Tensor): (n_bags,) of aggregate targets observed for each bag

        """
        if self.fit_intercept:
            individuals_covariates = self.pad_input(individuals_covariates)
        n_bags = individuals_covariates.size(0)
        d = individuals_covariates.size(-1)

        aggX = self.aggregate_fn(individuals_covariates).t()
        Q = aggX @ aggX.t() + n_bags * self.lbda * torch.eye(d)

        beta = gpytorch.inv_matmul(Q, aggX @ aggregate_targets)
        self.register_buffer('beta', beta)

    def forward(self, x):
        """Runs prediction

        Args:
            x (torch.Tensor): (n_samples, covariate_dimenionality)
                samples must not need to be organized by bags

        Returns:
            type: torch.Tensor

        """
        if self.fit_intercept:
            x = self.pad_input(x)
        return x @ self.beta
