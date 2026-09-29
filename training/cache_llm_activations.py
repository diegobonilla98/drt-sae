from itertools import islice
from pathlib import Path
import sys
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.llm_gemma2_2b import cfg
from data_sources.text_sources import iter_hf_texts, batched_texts
from data_sources.activation_store import save_activation_shard, shard_paths
from utils.device import resolve_device, dtype_from_name
from utils.llm_activations import extract_llm_activations


limit_texts = None
store_metadata = False
layer_id = cfg.default_layer_id


device = resolve_device(cfg.device)
torch_dtype = dtype_from_name(cfg.dtype) if device.type == "cuda" else torch.float32
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch_dtype, device_map=None).to(device)
texts = iter_hf_texts(cfg.dataset_name, cfg.dataset_config, cfg.dataset_split, cfg.text_column, limit_texts)
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
    texts = islice(texts, source_items_seen, None)
    print(f"resuming activation cache at shard {shard_id} after {source_items_seen} texts")
text_batches = batched_texts(texts, cfg.cache_text_batch_size)
progress = tqdm(text_batches, desc="Caching LLM activations", unit="batch", dynamic_ncols=True)

for batch in progress:
    x, metadata = extract_llm_activations(model, tokenizer, batch, layer_id, cfg.max_length, device)
    source_items_seen += len(batch)
    if x.numel() == 0:
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
        continue
    buffer_x.append(x)
    if store_metadata:
        buffer_metadata.extend(metadata)
    buffer_count += x.shape[0]
    while buffer_count >= cfg.shard_tokens:
        merged = torch.cat(buffer_x, dim=0)
        shard_x = merged[:cfg.shard_tokens]
        leftover = merged[cfg.shard_tokens:]
        shard_metadata = None
        if store_metadata:
            shard_metadata = buffer_metadata[:cfg.shard_tokens]
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
