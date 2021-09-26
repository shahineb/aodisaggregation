import torch


def pad_input(x):
    """Pads x with 1 along last dimension

    Args:
        x (torch.Tensor)

    Returns:
        type: torch.Tensor

    """
    x = torch.cat([x, torch.ones(x.shape[:-1], device=x.device).unsqueeze(-1)], dim=-1)
    return x
