import torch
import torch.nn as nn
from gpytorch import means, distributions
from gpytorch.models import ApproximateGP
from gpytorch import variational


class AggregateGammaSVGP(ApproximateGP):
    """Sparse Variational GP model that aggregates into the mean of a Gamma likelihood

    Args:
        inducing_points (torch.Tensor): sparse GP inducing points intialization.
        transform (callable): positive link function mapping the GP onto the Gamma mean.
        kernel (gpytorch.kernels.Kernel): GP covariance function.
        aggregate_fn (callable): aggregation operator that aggregates the transformed GP.
        fit_intercept (bool): if True, uses constant mean for the GP.

    """
    def __init__(self, inducing_points, transform, kernel, aggregate_fn, fit_intercept=False):
        variational_strategy = self._set_variational_strategy(inducing_points)
        super().__init__(variational_strategy=variational_strategy)
        self.raw_shape = nn.Parameter(torch.zeros(1))
        self.kernel = kernel
        self.transform = transform
        self.aggregate_fn = aggregate_fn
        self.fit_intercept = fit_intercept
        if self.fit_intercept:
            self.mean = means.ConstantMean()
        else:
            self.mean = means.ZeroMean()

    @property
    def shape(self):
        return torch.log(1 + torch.exp(self.raw_shape))

    def reparametrize(self, mu):
        """Computes gamma distribution concentrations (i.e. βi) out of
        mean values μi and shared shape α

        Args:
            mu (torch.tensor): (n_samples,) tensor of means of each sample

        Returns:
            type: torch.Tensor, torch.Tensor

        """
        alpha = self.shape
        beta = self.shape / mu
        return alpha, beta

    def _set_variational_strategy(self, inducing_points):
        """Sets variational family of distribution to use and variational approximation
            strategy module
        Args:
            inducing_points (torch.Tensor): tensor of landmark points from which to
                compute inducing values
        Returns:
            type: gpytorch.variational.VariationalStrategy
        """
        # Use gaussian variational family
        variational_distribution = variational.CholeskyVariationalDistribution(num_inducing_points=inducing_points.size(0))

        # Set default variational approximation strategy
        variational_strategy = variational.VariationalStrategy(model=self,
                                                               inducing_points=inducing_points,
                                                               variational_distribution=variational_distribution,
                                                               learn_inducing_locations=True)
        return variational_strategy

    def forward(self, inputs):
        """Defines prior distribution on input x as multivariate normal N(m(x), k(x, x))
        Args:
            inputs (torch.Tensor): input values
        Returns:
            type: gpytorch.distributions.MultivariateNormal
        """
        # Compute mean vector and covariance matrix on input samples
        mean = self.mean(inputs)
        covar = self.kernel(inputs)

        # Build multivariate normal distribution of model evaluated on input samples
        prior_distribution = distributions.MultivariateNormal(mean=mean,
                                                              covariance_matrix=covar)
        return prior_distribution
