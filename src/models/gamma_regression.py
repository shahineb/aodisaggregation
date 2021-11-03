import torch
import torch.nn as nn
from torch.distributions.gamma import Gamma


class AggregateGammaRegression(nn.Module):
    """Gamma regression model on aggregated targets - uses GLM reparametrization
    and regresses mean only

    Args:
        transform (callable): positive link function to apply to prediction
        aggregate_fn (callable): aggregation operator
        ndim (int): dimensionality of input samples
        fit_intercept (bool): if True, fits a constant offset term

    """
    def __init__(self, transform, aggregate_fn, ndim, fit_intercept=False):
        super().__init__()
        self.raw_scale = nn.Parameter(torch.zeros(1))
        self.transform = transform
        self.aggregate_fn = aggregate_fn
        self.fit_intercept = fit_intercept
        self.ndim = ndim
        if self.fit_intercept:
            self.bias = nn.Parameter(torch.zeros(1))
        self.weights = nn.Parameter(torch.zeros(self.ndim))

    @property
    def scale(self):
        return torch.log(1 + torch.exp(self.raw_scale))

    def reparametrize(self, mu):
        """Computes gamma distribution concentrations (i.e. αi and βi) out of
        mean values μi and shared precision Φ

        Args:
            mu (torch.tensor): (n_samples,) tensor of means of each sample

        Returns:
            type: torch.Tensor, torch.Tensor

        """
        alpha = self.scale
        beta = self.scale / mu
        return alpha, beta

    def predict_mean(self, x):
        f = x @ self.weights
        if self.fit_intercept:
            f = f + self.bias
        mu = self.transform(f)
        return mu

    def forward(self, x):
        """Runs prediction.

        Args:
            x (torch.Tensor): (n_samples, ndim)
                samples must not need to be organized by bags
        Returns:
            type: torch.Tensor
        """
        mu = self.predict_mean(x)
        alpha, beta = self.reparametrize(mu=mu)
        output = Gamma(concentration=alpha, rate=beta)
        return output

    def aggregate_prediction(self, prediction):
        """Computes aggregation of individuals output prediction.

        Args:
            prediction (torch.Tensor): (n_bag, bags_size) tensor output of forward
        Returns:
            type: torch.Tensor
        """
        aggregated_prediction = self.aggregate_fn(prediction)
        return aggregated_prediction
