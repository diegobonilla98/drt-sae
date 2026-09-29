import torch


def load_frequencies(indices: torch.Tensor, n_latents: int) -> torch.Tensor:
    counts = torch.bincount(indices.reshape(-1), minlength=n_latents).float()
    return counts / counts.sum().clamp_min(1.0)


def load_loss(indices: torch.Tensor, n_latents: int) -> torch.Tensor:
    freq = load_frequencies(indices, n_latents)
    return n_latents * freq.pow(2).sum() - 1.0


def normalized_load_entropy(indices: torch.Tensor, n_latents: int) -> torch.Tensor:
    freq = load_frequencies(indices, n_latents).clamp_min(1e-12)
    entropy = -(freq * freq.log()).sum()
    return entropy / torch.log(torch.tensor(float(n_latents), device=indices.device))


def update_router_bias_(model, indices: torch.Tensor, rate: float) -> None:
    with torch.no_grad():
        freq = load_frequencies(indices, model.n_latents).to(model.router_bias.device)
        target = torch.full_like(freq, 1.0 / model.n_latents)
        model.router_bias.add_(rate * (target - freq))
