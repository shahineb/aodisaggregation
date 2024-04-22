import math
import torch
from gpytorch import kernels


eps = torch.finfo(torch.float32).eps


def batch_haversine_distances(latlon1, latlon2):
    latlon1, latlon2 = torch.deg2rad(latlon1), torch.deg2rad(latlon2)
    diff = latlon2.unsqueeze(-3) - latlon1.unsqueeze(-2) + eps
    dlat = diff[..., 0]
    dlon = diff[..., 1]
    coslat1 = torch.cos(latlon1[..., 0])
    coslat2 = torch.cos(latlon2[..., 0])
    coslat1_coslat2 = torch.bmm(coslat1.unsqueeze(2), coslat2.unsqueeze(1))
    a = torch.sin(dlat / 2)**2 + coslat1_coslat2 * torch.sin(dlon / 2)**2
    dist = 2 * torch.asin(torch.sqrt(a + eps).clip(max=1 - eps))
    return dist


class HaversineMaternKernel(kernels.MaternKernel):
    _deg2rad = math.pi / 180
    _radius = 6371

    def forward(self, x1, x2, **kwargs):
        distance = batch_haversine_distances(x1, x2).div(self.lengthscale)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)

        covar = constant_component * exp_component
        return covar
