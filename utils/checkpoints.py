from pathlib import Path
from dataclasses import asdict, is_dataclass
import torch


def cfg_to_dict(cfg) -> dict:
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    cfg=None,
    step: int = 0,
    extra: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "step": step}
    if cfg is not None:
        payload["cfg"] = cfg_to_dict(cfg)
    if extra is not None:
        payload["extra"] = extra
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location)
