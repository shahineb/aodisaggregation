import torch
import torch.nn as nn
import gpytorch
from .utils import pad_input


class AggregateRidgeRegression(nn.Module):
    """Ridge Regression model when aggregated targets only are observed

        *** Current implementation assumes all columns have same size ***

    Args:
        lbda (float): regularization weight, greater = stronger L2 penalty
        aggregate_fn (callable): aggregation operator
        fit_intercept (bool): if True, pads inputs with constant offset
    """

    def __init__(self, lbda, aggregate_fn, fit_intercept=False):
        super().__init__()
        self.lbda = lbda
        self.aggregate_fn = aggregate_fn
        self.fit_intercept = fit_intercept

    def fit(self, individuals_covariates, aggregate_targets):
        """Fits model following sklearn syntax

        Args:
            individuals_covariates (torch.Tensor): (n_bags, bags_size, n_dim_individuals)
                samples must be organized by bags following which aggregation is taken
            aggregate_targets (torch.Tensor): (n_bags,) of aggregate targets observed for each bag

        """
        if self.fit_intercept:
            individuals_covariates = pad_input(individuals_covariates)
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
            x = pad_input(x)
        return x @ self.beta


class TwoStageAggregateRidgeRegression(nn.Module):
    """Two-stage Ridge Regression model when aggregate targets only are observed

        *** Current implementation assumes all columns have same size ***

        Two-stage regression procedure:
            (1): Regresses 2D covariates against aggregated predictors
            (2): Regresses first step againt aggregate targetrs

    Args:
        lbda_2d (float): regularization weight for first stage, greater = stronger L2 penalty
        lbda_3d (float): regularization weight for second stage, greater = stronger L2 penalty
        aggregate_fn (callable): aggregation operator
        fit_intercept_2d (bool): if True, pads 2D inputs with constant offset
        fit_intercept_3d (bool): if True, pads 3D inputs with constant offset
    """

    def __init__(self, lbda_2d, lbda_3d, aggregate_fn, fit_intercept_2d=False, fit_intercept_3d=False):
        super().__init__()
        self.lbda_2d = lbda_2d
        self.lbda_3d = lbda_3d
        self.aggregate_fn = aggregate_fn
        self.fit_intercept_2d = fit_intercept_2d
        self.fit_intercept_3d = fit_intercept_3d

    def fit(self, individuals_covariates, bags_covariates, aggregate_targets):
        """Fits model following sklearn syntax

        Args:
            individuals_covariates (torch.Tensor): (n_bags, bags_size, n_dim_individuals)
                samples must be organized by bags following which aggregation is taken
            bags_covariates (torch.Tensor): (n_bags, n_dim_bags_covariates)
            aggregate_targets (torch.Tensor): (n_bags,) of aggregate targets observed for each bag

        """
        if self.fit_intercept_2d:
            bags_covariates = pad_input(bags_covariates)
        if self.fit_intercept_3d:
            individuals_covariates = pad_input(individuals_covariates)

        # Extract tensors dimensionalities
        n_bags = aggregate_targets.size(0)
        d_2d = bags_covariates.size(-1)
        d_3d = individuals_covariates.size(-1)

        # Compute first regression stage
        Q_2d = (bags_covariates.t() @ bags_covariates + n_bags * self.lbda_2d * torch.eye(d_2d))
        aggX = self.aggregate_fn(individuals_covariates)
        upsilon = gpytorch.inv_matmul(Q_2d, bags_covariates.t() @ aggX)
        y_upsilon = bags_covariates @ upsilon

        # Compute second regression stage
        Q_3d = (y_upsilon.t() @ y_upsilon + n_bags * self.lbda_3d * torch.eye(d_3d))
        with gpytorch.settings.cholesky_jitter(1e-3):
            beta = gpytorch.inv_matmul(Q_3d, y_upsilon.t() @ aggregate_targets)
        self.register_buffer('beta', beta)

    def forward(self, x):
        """Runs prediction

        Args:
            x (torch.Tensor): (n_samples, covariate_dimenionality)
                samples must not need to be organized by bags

        Returns:
            type: torch.Tensor

        """
        if self.fit_intercept_3d:
            x = pad_input(x)
        return x @ self.beta


class WarpedAggregateRidgeRegression(nn.Module):
    """Ridge regression warped with link function when aggregate targets only are observed

        *** Current implementation assumes all columns have same size ***

    Args:
        lbda (float): regularization weight, greater = stronger L2 penalty
        transform (callable): link function to apply to prediction
        aggregate_fn (callable): aggregation operator
        ndim (int): dimensionality of input samples
        fit_intercept (bool): if True, fits a constant offset term
    """
    def __init__(self, lbda, transform, aggregate_fn, ndim, fit_intercept=False):
        super().__init__()
        self.lbda = lbda
        self.transform = transform
        self.aggregate_fn = aggregate_fn
        self.fit_intercept = fit_intercept
        self.ndim = ndim
        if self.fit_intercept:
            self.bias = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.randn(self.ndim))

    def forward(self, x):
        """Runs prediction.

        Args:
            x (torch.Tensor): (n_samples, ndim)
                samples must not need to be organized by bags
        Returns:
            type: torch.Tensor
        """
        output = x @ self.beta
        if self.fit_intercept:
            output = output + self.bias
        return self.transform(output)

    def aggregate_prediction(self, prediction):
        """Computes aggregation of individuals output prediction.

        Args:
            prediction (torch.Tensor): (n_bag, bags_size) tensor output of forward
        Returns:
            type: torch.Tensor
        """
        aggregated_prediction = self.aggregate_fn(prediction)
        return aggregated_prediction

    def regularization_term(self):
        """Square L2 norm of beta

        Returns:
            type: torch.Tensor
        """
        return self.lbda * torch.dot(self.beta, self.beta)


class WarpedTwoStageAggregateRidgeRegression(nn.Module):
    """Two-stage aggregate ridge regression warped with link function when aggregate
    targets only are observed

        *** Current implementation assumes all columns have same size ***

    Args:
        lbda_2d (float): regularization weight for first stage, greater = stronger L2 penalty
        lbda_3d (float): regularization weight for second stage, greater = stronger L2 penalty
        transform (callable): link function to apply to prediction
        aggregate_fn (callable): aggregation operator
        ndim (int): dimensionality of input samples
        fit_intercept_2d (bool): if True, pads 2D inputs with constant offset
        fit_intercept_3d (bool): if True, fits a constant offset term on 3D predictor
    """
    def __init__(self, lbda_2d, lbda_3d, transform, aggregate_fn, ndim, fit_intercept_2d=False, fit_intercept_3d=False):
        super().__init__()
        self.lbda_2d = lbda_2d
        self.lbda_3d = lbda_3d
        self.transform = transform
        self.aggregate_fn = aggregate_fn
        self.fit_intercept_2d = fit_intercept_2d
        self.fit_intercept_3d = fit_intercept_3d
        self.ndim = ndim
        if self.fit_intercept_3d:
            self.bias_3d = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.randn(self.ndim))

    def forward(self, x):
        """Runs prediction

        Args:
            x (torch.Tensor): (n_samples, ndim)
                samples must not need to be organized by bags
        Returns:
            type: torch.Tensor
        """
        output = x @ self.beta
        if self.fit_intercept_3d:
            output = output + self.bias_3d
        return self.transform(output)

    def aggregate_prediction(self, prediction, bags_covariates):
        """Computes aggregation of individuals output prediction and fits first
        regression stage against it.

        The output is used to regress against aggregate target outputs z

        Args:
            prediction (torch.Tensor): (n_bag, bags_size) tensor output of forward
            bags_covariates (torch.Tensor): (n_bags, n_dim_bags_covariates)

        Returns:
            type: torch.Tensor

        """
        if self.fit_intercept_2d:
            bags_covariates = pad_input(bags_covariates)

        # Extract tensors dimensionalities
        n_bags = bags_covariates.size(0)
        d_2d = bags_covariates.size(-1)

        # Aggregate predictions of over 3D covariates
        aggregated_prediction = self.aggregate_fn(prediction).squeeze()

        # Fit first regression stage against aggregated predictions
        Q_2d = (bags_covariates.t() @ bags_covariates + n_bags * self.lbda_2d * torch.eye(d_2d))
        upsilon = gpytorch.inv_matmul(Q_2d, bags_covariates.t() @ aggregated_prediction)

        # Predict out of first regression stage
        first_stage_aggregated_predictions = bags_covariates @ upsilon
        return first_stage_aggregated_predictions

    def regularization_term(self):
        """Square L2 norm of beta

        Returns:
            type: torch.Tensor
        """
        return self.lbda_3d * torch.dot(self.beta, self.beta)
