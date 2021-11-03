import torch
import torch.nn as nn
from torch.distributions.beta import Beta


class AggregateBetaRegression(nn.Module):
    """Beta regression model on aggregated targets - regresses mean only and
    considers fixed precision accross samples

    Args:
        transform (callable): link function to apply to prediction
        aggregate_fn (callable): aggregation operator
        ndim (int): dimensionality of input samples
        fit_intercept (bool): if True, fits a constant offset term

    """
    def __init__(self, transform, aggregate_fn, ndim, fit_intercept=False):
        super().__init__()
        self.raw_sigma2 = nn.Parameter(torch.zeros(1))
        self.transform = transform
        self.aggregate_fn = aggregate_fn
        self.fit_intercept = fit_intercept
        self.ndim = ndim
        if self.fit_intercept:
            self.bias = nn.Parameter(torch.zeros(1))
        self.weights = nn.Parameter(torch.randn(self.ndim))

    @property
    def sigma2(self):
        return torch.log(1 + torch.exp(self.raw_sigma2))

    @property
    def precision(self):
        precision = 1 / self.sigma2
        return precision

    def compute_concentrations(self, mu):
        precision = 1 / torch.square(self.sigma)
        alpha = ((1 - mu) * precision - 1 / mu) * torch.square(mu)
        beta = alpha * (1 / mu - 1)
        return alpha, beta

    def forward(self, x):
        """Runs prediction.

        Args:
            x (torch.Tensor): (n_samples, ndim)
                samples must not need to be organized by bags
        Returns:
            type: torch.Tensor
        """
        f = x @ self.beta
        if self.fit_intercept:
            f = f + self.bias
        transformed_f = self.transform(f)
        alpha, beta = self.compute_concentrations(mu=transformed_f)
        output = Beta(concentration1=alpha, concentration2=beta)
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
