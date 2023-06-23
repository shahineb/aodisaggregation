import math
import torch
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from gpytorch import kernels


# class HaversineMaternKernel(kernels.MaternKernel):
#     _deg2rad = math.pi / 180
#     _radius = 6371
#     _mu_latlon = torch.tensor([0., 179.0625])
#     _sigma_latlon = torch.tensor([51.6863, 103.9217])
#
#     def forward(self, x1, x2, **kwargs):
#         print(x1.shape)
#
#         latlon1_std = torch.deg2rad(x1).squeeze()
#         latlon2_std = torch.deg2rad(x2).squeeze()
#
#         latlon1 = self._mu_latlon + self._sigma_latlon * latlon1_std.detach()
#         latlon2 = self._mu_latlon + self._sigma_latlon * latlon2_std.detach()
#
#         distance = torch.from_numpy(haversine_distances(latlon1.view(-1, 2), latlon2.view(-1, 2))).float().div(self.lengthscale)
#         distance = distance.reshape()
#         exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)
#
#         if self.nu == 0.5:
#             constant_component = 1
#         elif self.nu == 1.5:
#             constant_component = (math.sqrt(3) * distance).add(1)
#         elif self.nu == 2.5:
#             constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
#
#         return constant_component * exp_component


class HaversineMaternKernel(kernels.MaternKernel):
    _deg2rad = math.pi / 180
    _radius = 6371

    def __init__(self, nu, **kwargs):
        super().__init__(nu=nu, **kwargs)
        self.register_buffer('mu_latlon', torch.tensor([0., 179.0625]))
        self.register_buffer('sigma_latlon', torch.tensor([51.6863, 103.9217]))

    def forward(self, x1, x2, **kwargs):
        latlon1_std = torch.deg2rad(x1).squeeze()
        latlon2_std = torch.deg2rad(x2).squeeze()
        latlon1 = self.mu_latlon + self.sigma_latlon * latlon1_std.detach()
        latlon2 = self.mu_latlon + self.sigma_latlon * latlon2_std.detach()

        distance = np.stack([haversine_distances(foo, bar) for (foo, bar) in zip(latlon1, latlon2)])
        distance = torch.from_numpy(distance).float().div(self.lengthscale)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)

        return constant_component * exp_component
