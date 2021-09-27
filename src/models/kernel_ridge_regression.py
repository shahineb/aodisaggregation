import torch
import torch.nn as nn
import gpytorch


class AggregateKernelRidgeRegression(nn.Module):
    """RFF Kernel Ridge Regression when aggregated targets only are observed

        *** Current implementation assumes all columns have same size ***

    Args:
        kernel (kernels.RFFKernel): Random Fourier features kernel instance
        lbda (float): regularization weight, greater = stronger L2 penalty
        aggregate_fn (callable): aggregation operator
    """

    def __init__(self, kernel, lbda, aggregate_fn):
        super().__init__()
        self.kernel = kernel
        self.lbda = lbda
        self.aggregate_fn = aggregate_fn

    def fit(self, individuals_covariates, aggregate_targets):
        """Fits aggregate kernel ridge regression model using random fourier features
        approximation.

        The learnt weights `beta_rff` does not correspond to the usual KRR weights,
        but are instead meant to enable prediction based on random fourier features.

        ```
            β_RFF = (Agg(Z) @ Agg(Z).t() + nλI)^{-1} @ Agg(Z) @ z

            f(x^*) = Z(x^*).t() @ β_RFF
        ```

        where
            - Z : random fourier features on training samples
            - z : aggregate targets
            - Z(x^*) : random fourier features on prediction samples

        Args:
            individuals_covariates (torch.Tensor): (n_bags, bags_size, n_dim_individuals)
                samples must be organized by bags following which aggregation is taken
            aggregate_targets (torch.Tensor): (n_bags,) of aggregate targets observed for each bag

        """
        # Extract tensors dimensions
        n_bags = individuals_covariates.size(0)
        d = individuals_covariates.size(-1)

        # Register training samples for prediction
        self.register_buffer('training_covariates', individuals_covariates)

        # Sample random fourier features weights
        self.kernel._init_weights(d, self.kernel.num_samples)

        # Compute random fourier features - Shape = (n_bags, bags_size, self.kernel.num_samples)
        Z = self.kernel._featurize(self.training_covariates, normalize=True)

        # Aggregate random fourier features - Shape = (self.kernel.num_samples, n_bags)
        aggZ = self.aggregate_fn(Z).t()

        # Compute kernel ridge regression solution - Shape = (self.kernel.num_samples,)
        Q = aggZ @ aggZ.t() + n_bags * self.lbda * torch.eye(2 * self.kernel.num_samples)
        beta_rff = gpytorch.inv_matmul(Q, aggZ @ aggregate_targets)
        self.register_buffer('beta_rff', beta_rff)

    def forward(self, x):
        """Runs prediction

        Args:
            x (torch.Tensor): (n_samples, n_dim_individuals)
                samples must not need to be organized by bags

        Returns:
            type: torch.Tensor

        """
        # Compute random fourier features of prediction samples - Shape = (n_samples, self.kernel.num_samples)
        Z = self.kernel._featurize(x, normalize=True)

        # Multiply with learnt weight - Shape = (n_samples)
        output = Z @ self.beta_rff
        return output
