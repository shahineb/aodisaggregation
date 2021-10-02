import torch
import torch.nn as nn
import gpytorch


class AggregateKernelRidgeRegression(nn.Module):
    """RFF Kernel Ridge Regression when aggregated targets only are observed

        *** Current implementation assumes all columns have same size ***

    Args:
        kernel (kernels.RFFKernel): Random Fourier features kernel instance.
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

        # Sample random fourier features weights
        self.kernel._init_weights(d, self.kernel.num_samples)

        # Compute random fourier features - Shape = (n_bags, bags_size, self.kernel.num_samples)
        Z = self.kernel._featurize(individuals_covariates, normalize=True)

        # Aggregate random fourier features - Shape = (self.kernel.num_samples, n_bags)
        aggZ = self.aggregate_fn(Z).t()

        # Compute kernel ridge regression solution - Shape = (self.kernel.num_samples,)
        Q = aggZ @ aggZ.t() + n_bags * self.lbda * torch.eye(2 * self.kernel.num_samples, device=aggZ.device)
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


class TwoStageAggregateKernelRidgeRegression(nn.Module):
    """RFF Two-stage Kernel Ridge Regression when aggregated targets only are observed

        *** Current implementation assumes all columns have same size ***

    Args:
        kernel_2d (kernels.RFFKernel): Random Fourier features kernel instance for bags covariates.
        kernel_3d (kernels.RFFKernel): Random Fourier features kernel instance for individuals covariates.
        lbda_2d (float): regularization weight for first stage, greater = stronger L2 penalty
        lbda_3d (float): regularization weight for second stage, greater = stronger L2 penalty
        aggregate_fn (callable): aggregation operator

    """

    def __init__(self, kernel_2d, kernel_3d, lbda_2d, lbda_3d, aggregate_fn):
        super().__init__()
        self.kernel_2d = kernel_2d
        self.kernel_3d = kernel_3d
        self.lbda_2d = lbda_3d
        self.lbda_3d = lbda_3d
        self.aggregate_fn = aggregate_fn

    def fit(self, individuals_covariates, bags_covariates, aggregate_targets):
        """Fits model following sklearn syntax

        Args:
            individuals_covariates (torch.Tensor): (n_bags, bags_size, n_dim_individuals)
                samples must be organized by bags following which aggregation is taken
            bags_covariates (torch.Tensor): (n_bags, n_dim_bags_covariates)
            aggregate_targets (torch.Tensor): (n_bags,) of aggregate targets observed for each bag

        """
        # Extract tensors dimensions
        n_bags = individuals_covariates.size(0)
        d_2d = bags_covariates.size(-1)
        d_3d = individuals_covariates.size(-1)

        # Sample random fourier features weights for both kernel
        self.kernel_2d._init_weights(d_2d, self.kernel_2d.num_samples)
        self.kernel_3d._init_weights(d_3d, self.kernel_3d.num_samples)

        # Compute random fourier features
        Z_2d = self.kernel_2d._featurize(bags_covariates, normalize=True)  # (n_bags, self.kernel_2d.num_samples)
        Z_3d = self.kernel_3d._featurize(individuals_covariates, normalize=True)  # (n_bags, bags_size, self.kernel_3d.num_samples)

        # Fit first regression stage
        Q_2d = (Z_2d.t() @ Z_2d + n_bags * self.lbda_2d * torch.eye(2 * self.kernel_2d.num_samples, device=Z_2d.device))  # (self.kernel_2d.num_samples, self.kernel_2d.num_samples)
        aggZ_3d = self.aggregate_fn(Z_3d)  # (n_bags, self.kernel_3d.num_samples)
        upsilon = gpytorch.inv_matmul(Q_2d, Z_2d.t() @ aggZ_3d)  # (self.kernel_2d.num_samples, self.kernel_3d.num_samples)
        y_upsilon = Z_2d @ upsilon  # (n_bags, self.kernel_3d.num_samples)

        # Fit second regression stage
        Q_3d = (y_upsilon.t() @ y_upsilon + n_bags * self.lbda_3d * torch.eye(2 * self.kernel_3d.num_samples, device=y_upsilon.device))  # (self.kernel_3d.num_samples, self.kernel_3d.num_samples)
        beta_rff = gpytorch.inv_matmul(Q_3d, y_upsilon.t() @ aggregate_targets)  # (self.kernel_3d.num_samples)
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
        Z = self.kernel_3d._featurize(x, normalize=True)

        # Multiply with learnt weight - Shape = (n_samples)
        output = Z @ self.beta_rff
        return output


class WarpedAggregateKernelRidgeRegression(nn.Module):
    """RFF Warped Kernel Ridge Regression when aggregated targets only are observed

        *** Current implementation assumes all columns have same size ***

    Args:
        kernel (kernels.RFFKernel): covariance function over inputs 3D samples.
        training_covariates (torch.Tensor): (n_bags, bags_size, n_dim_individuals)
            samples must be organized by bags following which aggregation is taken
        lbda (float): regularization weight, greater = stronger L2 penalty
        transform (callable): link function to apply to prediction
        aggregate_fn (callable): aggregation operator
    """

    def __init__(self, kernel, training_covariates, lbda, transform, aggregate_fn):
        super().__init__()
        self.ndim = training_covariates.size(-1)
        self.kernel = kernel
        self.kernel._init_weights(self.ndim, self.kernel.num_samples)
        self.lbda = lbda
        self.transform = transform
        self.aggregate_fn = aggregate_fn
        self.register_buffer('rff_training_features', self.kernel._featurize(training_covariates, normalize=True))
        self.beta = nn.Parameter(torch.randn(training_covariates.size(0)))

    def forward(self, x):
        """Runs prediction.

        Args:
            x (torch.Tensor): (n_samples, ndim)
                samples must not need to be organized by bags
        Returns:
            type: torch.Tensor
        """
        Z = self.kernel._featurize(x, normalize=True)
        self._Z_train_beta = self.rff_training_features.t() @ self.beta
        K_beta = Z @ self._Z_train_beta
        return self.transform(K_beta)

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
        return self.lbda * torch.square(self._Z_train_beta).sum()


class WarpedTwoStageAggregateKernelRidgeRegression(nn.Module):
    """RFF Warped Two-stage Kernel Ridge Regression when aggregated targets only are observed

        *** Current implementation assumes all columns have same size ***

    Args:
        kernel_2d (kernels.RFFKernel): Random Fourier features kernel instance for bags covariates.
        kernel_3d (kernels.RFFKernel): Random Fourier features kernel instance for individuals covariates.
        lbda_2d (float): regularization weight for first stage, greater = stronger L2 penalty
        lbda_3d (float): regularization weight for second stage, greater = stronger L2 penalty
        aggregate_fn (callable): aggregation operator

    """

    def __init__(self, kernel_2d, kernel_3d, training_covariates_3d, training_covariates_2d, lbda_2d, lbda_3d, transform, aggregate_fn):
        super().__init__()
        # Setup dimensions attributes
        self.ndim_2d = training_covariates_2d.size(-1)
        self.ndim_3d = training_covariates_3d.size(-1)

        # Setup 2D kernel and random fourier features
        self.kernel_2d = kernel_2d
        self.kernel_2d._init_weights(training_covariates_2d.size(-1), self.kernel_2d.num_samples)
        self.register_buffer('rff_training_features_2d', self.kernel_2d._featurize(training_covariates_2d, normalize=True))

        # Setup 3D kernel and random fourier features
        self.kernel_3d = kernel_3d
        self.kernel_3d._init_weights(training_covariates_3d.size(-1), self.kernel_3d.num_samples)
        self.register_buffer('rff_training_features_3d', self.kernel_3d._featurize(training_covariates_3d, normalize=True))

        # Setup regularization weights
        self.lbda_2d = lbda_3d
        self.lbda_3d = lbda_3d

        # Setup callable attributess
        self.transform = transform
        self.aggregate_fn = aggregate_fn

        # Initialize weight tensor to learn
        self.beta = nn.Parameter(torch.randn(training_covariates_3d.size(0)))

    def forward(self, x):
        """Runs prediction.

        Args:
            x (torch.Tensor): (n_samples, ndim)
                samples must not need to be organized by bags
        Returns:
            type: torch.Tensor
        """
        Z = self.kernel_3d._featurize(x, normalize=True)
        self._Z_train_beta = self.rff_training_features_3d.t() @ self.beta
        K_beta = Z @ self._Z_train_beta
        return self.transform(K_beta)

    def aggregate_prediction(self, prediction):
        """Computes aggregation of individuals output prediction and fits first
        regression stage against it.

        The output is used to regress against aggregate target outputs z

        Args:
            prediction (torch.Tensor): (n_bag, bags_size) tensor output of forward

        Returns:
            type: torch.Tensor

        """
        # Extract tensors dimensionalities
        n_bags = prediction.size(0)

        # Fit first regression stage
        Z_2d = self.rff_training_features_2d  # (n_bags, R_2d)
        Q_2d = (Z_2d.t() @ Z_2d + n_bags * self.lbda_2d * torch.eye(2 * self.kernel_2d.num_samples, device=Z_2d.device))  # (R_2d, R_2d)
        aggregated_prediction = self.aggregate_fn(prediction).squeeze()  # (n_bags,)
        upsilon = gpytorch.inv_matmul(Q_2d, Z_2d.t() @ aggregated_prediction)  # (R_2d,)

        # Predict out of first stage
        first_stage_aggregated_predictions = Z_2d @ upsilon  # (n_bags,)
        return first_stage_aggregated_predictions

    def regularization_term(self):
        """Square L2 norm of beta

        Returns:
            type: torch.Tensor
        """
        return self.lbda_3d * torch.square(self._Z_train_beta).sum()
