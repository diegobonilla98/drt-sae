from pathlib import Path
import random
import torch


def shard_paths(cache_dir: str) -> list[Path]:
    return sorted(Path(cache_dir).glob("shard_*.pt"))


def save_activation_shard(cache_dir: str, shard_id: int, x: torch.Tensor, metadata: list[dict] | None = None) -> Path:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cache_dir) / f"shard_{shard_id:06d}.pt"
    payload = {"x": x.cpu()}
    if metadata is not None:
        payload["metadata"] = metadata
    torch.save(payload, path)
    return path


def iterate_activation_batches(cache_dir: str, batch_size: int, device: str, shuffle_files: bool = True, forever: bool = True):
    paths = shard_paths(cache_dir)
    if not paths:
        raise FileNotFoundError(f"No activation shards found in {cache_dir}")
    while True:
        current_paths = list(paths)
        if shuffle_files:
            random.shuffle(current_paths)
        for path in current_paths:
            payload = torch.load(path, map_location="cpu")
            x = payload["x"]
            order = torch.randperm(x.shape[0]) if shuffle_files else torch.arange(x.shape[0])
            for start in range(0, x.shape[0], batch_size):
                idx = order[start:start + batch_size]
                yield x.index_select(0, idx).to(device, non_blocking=True)
        if not forever:
            break


def load_metadata_shards(cache_dir: str) -> list[dict]:
    records = []
    for path in shard_paths(cache_dir):
        payload = torch.load(path, map_location="cpu")
        metadata = payload.get("metadata", [])
        records.extend(metadata)
    return records
