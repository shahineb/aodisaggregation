import torch
import torch.nn as nn
from torch.distributions.gamma import Gamma


class AggregateMAPGPGammaRegression(nn.Module):
    """MAP gamma regression over aggregate targets with GP prior over mean

    Args:
        shapeMAP (tuple[int]): shape of MAP tensor
        transform (callable): positive link function to apply to prediction
        kernel (gpytorch.kernels.Kernel): GP prior kernel
        aggregate_fn (callable): aggregation operator
        ndim (int): dimensionality of input samples
        fit_intercept (bool): if True, GP has constant mean, else GP has zero mean

    """
    def __init__(self, shapeMAP, transform, kernel, aggregate_fn, ndim, fit_intercept=False):
        super().__init__()
        self.fMAP = nn.Parameter(torch.zeros(shapeMAP))
        self.biasMAP = nn.Parameter(torch.zeros(1))
        self.raw_scale = nn.Parameter(torch.zeros(1))
        self.transform = transform
        self.kernel = kernel
        self.aggregate_fn = aggregate_fn
        self.fit_intercept = fit_intercept
        self.ndim = ndim
        if self.fit_intercept:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.bias = 0.

    @property
    def scale(self):
        return torch.log(1 + torch.exp(self.raw_scale))

    def reparametrize(self, mu):
        """Computes gamma distribution concentrations (i.e. βi) out of
        mean values μi and shared precision α

        Args:
            mu (torch.tensor): (n_samples,) tensor of means of each sample

        Returns:
            type: torch.Tensor, torch.Tensor

        """
        alpha = self.scale
        beta = self.scale / mu
        return alpha, beta

    def forward(self):
        """Runs prediction.

        Args:
            x (torch.Tensor): (n_samples, ndim)
                samples must not need to be organized by bags
        Returns:
            type: torch.Tensor
        """
        mu = self.transform(self.bias + self.fMAP)
        alpha, beta = self.reparametrize(mu=mu)
        output = Gamma(concentration=alpha, rate=beta)
        return output

    def aggregate_prediction(self, prediction):
        """Computes aggregation of individuals output prediction.

        Args:
            prediction (torch.Tensor): (n_bag, bags_size) tensor output of forward
        Returns:
            type: Gamma
        """
        aggregated_prediction = self.aggregate_fn(prediction)
        return aggregated_prediction
