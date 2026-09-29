from itertools import islice
import json
from pathlib import Path
import sys
import time
import torch
from tqdm.auto import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.cv_clip_vit_b16 import cfg
from data_sources.image_sources import iter_image_files, batched_items, load_rgb_image
from models.routed_sae import RoutedSparseAutoencoder
from utils.checkpoints import load_checkpoint
from utils.device import resolve_device, dtype_from_name, autocast_context
from utils.cv_activations import extract_clip_patch_activations
from concepts.top_examples import TopExampleStore
from concepts.concept_store import read_jsonl, save_jsonl
from concepts.concept_namer import (
    batch_chat_completion_request,
    cv_concept_messages,
    download_file_text,
    fallback_concept,
    openai_client_from_env,
    parse_batch_chat_completions,
    submit_chat_completion_batch,
)
from concepts.vision_contact_sheet import make_contact_sheet


use_openai = True
use_openai_batch = True
cv_batch_examples_per_expert = 4
cv_batch_tile_size = 96
cv_batch_columns = 2
cv_batch_jpeg_quality = 65
max_batch_input_bytes = 180_000_000
max_image_batches = 2000
max_experts_to_label = None
activation_subbatch = 8192
layer_id = cfg.default_layer_id
concept_output_path = f"{cfg.concept_cache_dir}/concepts.jsonl"
contact_sheet_dir = f"{cfg.concept_cache_dir}/contact_sheets"
progress_path = Path(cfg.concept_cache_dir) / "concept_progress.pt"
batch_state_path = Path(cfg.concept_cache_dir) / "cv_concept_batch.json"
batch_input_prefix = Path(cfg.concept_cache_dir) / "cv_concept_batch_input"
batch_output_prefix = Path(cfg.concept_cache_dir) / "cv_concept_batch_output"
Path(cfg.concept_cache_dir).mkdir(parents=True, exist_ok=True)
openai_client = openai_client_from_env(cfg.openai_env_key) if use_openai else None


def save_progress(state: dict) -> None:
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(progress_path)


def batch_input_path(part_id: int) -> Path:
    return batch_input_prefix.with_name(f"{batch_input_prefix.name}_{part_id:03d}.jsonl")


def batch_output_path(part_id: int) -> Path:
    return batch_output_prefix.with_name(f"{batch_output_prefix.name}_{part_id:03d}.jsonl")


store = TopExampleStore(top_n=cfg.concept_top_n, random_n=cfg.concept_low_n)
start_batch = 0
if progress_path.exists():
    progress_payload = torch.load(progress_path, map_location="cpu", weights_only=False)
    store = progress_payload.get("store", store)
    start_batch = int(progress_payload.get("next_batch_id", 0))
    print(f"resuming CV concept extraction after batch {start_batch}")

if start_batch < max_image_batches:
    device = resolve_device(cfg.device)
    torch_dtype = dtype_from_name(cfg.dtype) if device.type == "cuda" else torch.float32
    payload = load_checkpoint(cfg.router_checkpoint, map_location=device)
    d_model = payload.get("extra", {}).get("d_model")
    n_latents = payload.get("extra", {}).get("n_latents")
    sae = RoutedSparseAutoencoder(d_model, n_latents, cfg.router_k, cfg.router_dim, cfg.score_chunk_size).to(device)
    sae.load_state_dict(payload["state_dict"])
    sae.eval()
    processor = CLIPImageProcessor.from_pretrained(cfg.model_name)
    model = CLIPVisionModel.from_pretrained(cfg.model_name, torch_dtype=torch_dtype).to(device)
    paths = iter_image_files(cfg.image_dataset_root, None)

    with torch.no_grad():
        image_batches = batched_items(paths, cfg.cache_image_batch_size)
        image_batches = islice(image_batches, start_batch, max_image_batches)
        progress = tqdm(image_batches, total=max(0, max_image_batches - start_batch), desc="Extracting CV concepts", unit="batch", dynamic_ncols=True)
        for batch_id, batch_paths in enumerate(progress, start=start_batch):
            images = [load_rgb_image(path) for path in batch_paths]
            x_cpu, metadata = extract_clip_patch_activations(model, processor, images, layer_id, device)
            for item in metadata[:16]:
                item = dict(item)
                item["image_path"] = batch_paths[item["sample_index"]]
                store.add_random(item)
            subbatches = range(0, x_cpu.shape[0], activation_subbatch)
            for start in tqdm(subbatches, desc="Scoring CV activations", unit="subbatch", leave=False, dynamic_ncols=True):
                x = x_cpu[start:start + activation_subbatch].to(device)
                with autocast_context(device, cfg.dtype):
                    out = sae(x)
                indices = out["indices"].detach().cpu()
                values = out["values"].detach().cpu()
                for row in range(indices.shape[0]):
                    meta = dict(metadata[start + row])
                    meta["image_path"] = batch_paths[meta["sample_index"]]
                    for slot in range(indices.shape[1]):
                        value = float(values[row, slot])
                        if value <= 0:
                            continue
                        expert_id = int(indices[row, slot])
                        payload_row = dict(meta)
                        payload_row["rank"] = slot
                        store.update(expert_id, value, payload_row)
            progress.set_postfix(discovered_experts=len(store.expert_ids()))
            if batch_id % 25 == 0:
                progress.write(f"processed_image_batches={batch_id} discovered_experts={len(store.expert_ids())}")
            save_progress({"store": store, "next_batch_id": batch_id + 1, "complete": False})

    save_progress({"store": store, "next_batch_id": max_image_batches, "complete": True})

rows = read_jsonl(concept_output_path) if Path(concept_output_path).exists() else []
existing_expert_ids = {int(row["expert_id"]) for row in rows}
expert_ids = store.expert_ids()
if max_experts_to_label is not None:
    expert_ids = expert_ids[:max_experts_to_label]
expert_ids = [expert_id for expert_id in expert_ids if expert_id not in existing_expert_ids]

if use_openai and use_openai_batch:
    eligible_expert_ids = []
    for expert_id in expert_ids:
        if store.stats(expert_id)["active_count"] >= cfg.concept_min_examples:
            eligible_expert_ids.append(expert_id)
    if not eligible_expert_ids:
        save_jsonl(concept_output_path, rows)
        print(f"saved {concept_output_path}")
        raise SystemExit(0)

    state = json.loads(batch_state_path.read_text(encoding="utf-8")) if batch_state_path.exists() else {}
    if "batch_id" in state:
        old_batch = openai_client.batches.retrieve(state["batch_id"])
        print(f"OpenAI batch {old_batch.id} status={old_batch.status}")
        if old_batch.status in ("failed", "cancelled", "expired"):
            archive_path = batch_state_path.with_suffix(f".{int(time.time())}.failed.json")
            batch_state_path.replace(archive_path)
            old_input_path = state.get("input_path")
            if old_input_path and Path(old_input_path).exists():
                Path(old_input_path).replace(Path(old_input_path).with_suffix(".failed.jsonl"))
            print(f"archived failed batch state to {archive_path}; resubmitting smaller batches")
            state = {}
        else:
            state = {"batches": [{"batch_id": state["batch_id"], "input_path": state.get("input_path"), "total": state.get("total"), "part_id": 0}]}
    batches = state.get("batches", [])
    if not batches:
        request_parts = [[]]
        part_sizes = [0]
        for expert_id in eligible_expert_ids:
            examples = store.top_examples(expert_id)
            stats = store.stats(expert_id)
            sheet_path = f"{contact_sheet_dir}/expert_{expert_id}.jpg"
            make_contact_sheet(
                examples[:cv_batch_examples_per_expert],
                sheet_path,
                tile_size=cv_batch_tile_size,
                columns=cv_batch_columns,
                quality=cv_batch_jpeg_quality,
            )
            custom_id = f"cv_expert_{expert_id}"
            compact_examples = examples[:cv_batch_examples_per_expert]
            messages = cv_concept_messages(expert_id, sheet_path, compact_examples, stats)
            request = batch_chat_completion_request(custom_id, cfg.concept_llm_model, messages)
            request_bytes = len(json.dumps(request, ensure_ascii=False).encode("utf-8")) + 1
            if request_parts[-1] and part_sizes[-1] + request_bytes > max_batch_input_bytes:
                request_parts.append([])
                part_sizes.append(0)
            request_parts[-1].append(request)
            part_sizes[-1] += request_bytes

        batches = []
        for part_id, requests in enumerate(request_parts):
            input_path = batch_input_path(part_id)
            batch = submit_chat_completion_batch(
                openai_client,
                requests,
                input_path,
                {"description": f"CV SAE concept labeling part {part_id}", "project": cfg.project_name},
            )
            batches.append({"batch_id": batch.id, "input_path": str(input_path), "total": len(requests), "part_id": part_id})
            print(f"submitted OpenAI batch {batch.id} with {len(requests)} CV experts ({part_sizes[part_id]} bytes)")
        batch_state_path.write_text(
            json.dumps({"batches": batches, "total": len(eligible_expert_ids)}, indent=2),
            encoding="utf-8",
        )
        print("rerun this script later to check status and collect completed labels")
        raise SystemExit(0)

    batch_rows = {}
    all_completed = True
    for item in batches:
        batch = openai_client.batches.retrieve(item["batch_id"])
        counts = getattr(batch, "request_counts", None)
        print(f"OpenAI batch {batch.id} status={batch.status} counts={counts}")
        if batch.status != "completed":
            all_completed = False
            continue
        output_file_id = getattr(batch, "output_file_id", None)
        if output_file_id is None:
            raise RuntimeError(f"batch {batch.id} completed without output_file_id")
        output_path = batch_output_path(int(item["part_id"]))
        output_path.write_text(download_file_text(openai_client, output_file_id), encoding="utf-8")
        batch_rows.update(parse_batch_chat_completions(output_path))
    if not all_completed:
        print("batch is not complete yet; rerun this script later")
        raise SystemExit(0)
    label_progress = tqdm(eligible_expert_ids, desc="Collecting CV batch labels", unit="expert", dynamic_ncols=True)
    for expert_id in label_progress:
        row = batch_rows.get(f"cv_expert_{expert_id}")
        if row is None:
            continue
        examples = store.top_examples(expert_id)
        sheet_path = f"{contact_sheet_dir}/expert_{expert_id}.jpg"
        row["expert_id"] = expert_id
        row["activation_stats"] = store.stats(expert_id)
        row["top_examples"] = examples
        row["contact_sheet_path"] = sheet_path
        row["modality"] = "cv"
        row["layer_id"] = layer_id
        rows.append(row)
        save_jsonl(concept_output_path, rows)
        label_progress.write(f"labeled expert {expert_id}")
    save_jsonl(concept_output_path, rows)
    print(f"saved {concept_output_path}")
    raise SystemExit(0)

label_progress = tqdm(expert_ids, desc="Labeling CV experts", unit="expert", dynamic_ncols=True)
for expert_id in label_progress:
    examples = store.top_examples(expert_id)
    stats = store.stats(expert_id)
    if stats["active_count"] < cfg.concept_min_examples:
        continue
    sheet_path = f"{contact_sheet_dir}/expert_{expert_id}.jpg"
    make_contact_sheet(examples[:cv_batch_examples_per_expert], sheet_path, tile_size=cv_batch_tile_size, columns=cv_batch_columns, quality=cv_batch_jpeg_quality)
    if use_openai:
        raise RuntimeError("Synchronous OpenAI labeling is disabled here; set use_openai_batch=True or use_openai=False.")
    else:
        row = fallback_concept(expert_id, "cv", examples, stats)
        row["top_examples"] = examples
        row["contact_sheet_path"] = sheet_path
    row["modality"] = "cv"
    row["layer_id"] = layer_id
    rows.append(row)
    save_jsonl(concept_output_path, rows)
    label_progress.write(f"labeled expert {expert_id}")

save_jsonl(concept_output_path, rows)
print(f"saved {concept_output_path}")
