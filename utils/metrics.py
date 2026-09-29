import torch
import torch.nn.functional as F


def mse_loss(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon.float(), x.float())


def r2_score(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    x_float = x.float()
    recon_float = recon.float()
    numerator = (x_float - recon_float).pow(2).sum()
    centered = x_float - x_float.mean(dim=0, keepdim=True)
    denominator = centered.pow(2).sum().clamp_min(1e-8)
    return 1.0 - numerator / denominator


def cosine_recon(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    x_float = x.float()
    recon_float = recon.float()
    return F.cosine_similarity(x_float, recon_float, dim=-1).mean()


def selected_l0(values: torch.Tensor) -> torch.Tensor:
    return (values > 0).float().sum(dim=-1).mean()
