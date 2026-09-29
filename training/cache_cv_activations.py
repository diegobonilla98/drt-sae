from itertools import islice
from pathlib import Path
import sys
import time
import torch
from tqdm.auto import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.cv_clip_vit_b16 import cfg
from data_sources.image_sources import iter_image_files, batched_items, load_rgb_image
from data_sources.activation_store import save_activation_shard, shard_paths
from utils.device import resolve_device, dtype_from_name
from utils.cv_activations import extract_clip_patch_activations


limit_images = cfg.cache_image_limit
store_metadata = True
layer_id = cfg.default_layer_id
time_limit_seconds = None if cfg.cache_time_limit_hours is None else cfg.cache_time_limit_hours * 3600


device = resolve_device(cfg.device)
torch_dtype = dtype_from_name(cfg.dtype) if device.type == "cuda" else torch.float32
processor = CLIPImageProcessor.from_pretrained(cfg.model_name)
model = CLIPVisionModel.from_pretrained(cfg.model_name, torch_dtype=torch_dtype).to(device)
paths = iter_image_files(cfg.image_dataset_root, limit_images)
progress_path = Path(cfg.activation_cache_dir) / "cache_progress.pt"


def save_progress(state: dict) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(progress_path)

buffer_x = []
buffer_metadata = []
buffer_count = 0
source_items_seen = 0
existing_shards = shard_paths(cfg.activation_cache_dir)
shard_id = int(existing_shards[-1].stem.split("_")[-1]) + 1 if existing_shards else 0
if progress_path.exists():
    progress_payload = torch.load(progress_path, map_location="cpu", weights_only=False)
    if progress_payload.get("complete"):
        print(f"{cfg.activation_cache_dir} is already marked complete")
        raise SystemExit(0)
    buffer_x = progress_payload.get("buffer_x", [])
    buffer_metadata = progress_payload.get("buffer_metadata", [])
    buffer_count = int(progress_payload.get("buffer_count", 0))
    shard_id = int(progress_payload.get("shard_id", shard_id))
    source_items_seen = int(progress_payload.get("source_items_seen", 0))
    paths = islice(paths, source_items_seen, None)
    print(f"resuming activation cache at shard {shard_id} after {source_items_seen} images")
image_batches = batched_items(paths, cfg.cache_image_batch_size)
remaining_images = None if limit_images is None else max(0, limit_images - source_items_seen)
total_batches = None if remaining_images is None else (remaining_images + cfg.cache_image_batch_size - 1) // cfg.cache_image_batch_size
progress = tqdm(image_batches, total=total_batches, desc="Caching CV activations", unit="batch", dynamic_ncols=True)
started_at = time.monotonic()

for batch_paths in progress:
    images = [load_rgb_image(path) for path in batch_paths]
    x, metadata = extract_clip_patch_activations(model, processor, images, layer_id, device)
    source_items_seen += len(batch_paths)
    for item in metadata:
        item["image_path"] = batch_paths[item["sample_index"]]
    buffer_x.append(x)
    if store_metadata:
        buffer_metadata.extend(metadata)
    buffer_count += x.shape[0]
    while buffer_count >= cfg.shard_tokens:
        merged = torch.cat(buffer_x, dim=0)
        shard_x = merged[:cfg.shard_tokens]
        leftover = merged[cfg.shard_tokens:]
        shard_metadata = buffer_metadata[:cfg.shard_tokens] if store_metadata else None
        if store_metadata:
            buffer_metadata = buffer_metadata[cfg.shard_tokens:]
        save_activation_shard(cfg.activation_cache_dir, shard_id, shard_x, shard_metadata)
        progress.write(f"saved shard {shard_id} with {shard_x.shape[0]} activations")
        shard_id += 1
        buffer_x = [leftover] if leftover.shape[0] > 0 else []
        buffer_count = leftover.shape[0]
    save_progress(
        {
            "buffer_x": buffer_x,
            "buffer_metadata": buffer_metadata,
            "buffer_count": buffer_count,
            "shard_id": shard_id,
            "source_items_seen": source_items_seen,
            "complete": False,
        },
    )
    progress.set_postfix(shards=shard_id, buffered=buffer_count)
    if time_limit_seconds is not None and time.monotonic() - started_at >= time_limit_seconds:
        progress.write(f"stopping after {cfg.cache_time_limit_hours}h time limit; rerun to resume")
        raise SystemExit(0)

if buffer_count > 0:
    merged = torch.cat(buffer_x, dim=0)
    shard_metadata = buffer_metadata if store_metadata else None
    save_activation_shard(cfg.activation_cache_dir, shard_id, merged, shard_metadata)
    progress.write(f"saved shard {shard_id} with {merged.shape[0]} activations")
    shard_id += 1

save_progress(
    {
        "buffer_x": [],
        "buffer_metadata": [],
        "buffer_count": 0,
        "shard_id": shard_id,
        "source_items_seen": source_items_seen,
        "complete": True,
    },
)
