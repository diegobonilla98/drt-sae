import torch
import torch.nn.functional as F


def coefficient_loss(values: torch.Tensor) -> torch.Tensor:
    return values.float().pow(2).mean()


def sampled_route_distillation_loss(
    model,
    x: torch.Tensor,
    teacher_indices: torch.Tensor,
    teacher_values: torch.Tensor,
    route_indices: torch.Tensor,
    negative_count: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    batch_size = x.shape[0]
    negatives = torch.randint(0, model.n_latents, (batch_size, negative_count), device=x.device)
    candidates = torch.cat([teacher_indices, route_indices.detach(), negatives], dim=-1)
    scores = model.score_candidates(x, candidates) / temperature
    teacher_weights = teacher_values.float().clamp_min(0.0)
    teacher_weights = teacher_weights / teacher_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    matches = candidates[:, :, None] == teacher_indices[:, None, :]
    targets = (matches.float() * teacher_weights[:, None, :]).sum(dim=-1)
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_probs = F.log_softmax(scores.float(), dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()
