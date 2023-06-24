import math
import torch
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
import gpytorch
from gpytorch import kernels


class HaversineMaternKernel(kernels.MaternKernel):
    _deg2rad = math.pi / 180
    _radius = 6371

    def __init__(self, nu, **kwargs):
        super().__init__(nu=nu, **kwargs)
        self.register_buffer('mu_latlon', torch.tensor([0., 179.0625]))
        self.register_buffer('sigma_latlon', torch.tensor([51.6863, 103.9217]))

    def forward(self, x1, x2, **kwargs):
        latlon1 = torch.deg2rad(x1).detach()
        latlon2 = torch.deg2rad(x2).detach()

        hav = np.stack([haversine_distances(foo.cpu(), bar.cpu()) for (foo, bar) in zip(latlon1, latlon2)])
        distance = torch.from_numpy(hav).float().to(x1.device).div(self.lengthscale)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)

        covar = constant_component * exp_component
        if covar.size(-1) == covar.size(-2):
            covar = gpytorch.add_jitter(covar)
        return covar
