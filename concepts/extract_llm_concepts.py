from itertools import islice
import json
from pathlib import Path
import sys
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.llm_gemma2_2b import cfg
from data_sources.text_sources import iter_hf_texts, batched_texts
from models.routed_sae import RoutedSparseAutoencoder
from utils.checkpoints import load_checkpoint
from utils.device import resolve_device, dtype_from_name, autocast_context
from utils.llm_activations import extract_llm_activations
from concepts.top_examples import TopExampleStore
from concepts.concept_store import read_jsonl, save_jsonl
from concepts.concept_namer import (
    batch_chat_completion_request,
    download_file_text,
    llm_concept_messages,
    openai_client_from_env,
    parse_batch_chat_completions,
    submit_chat_completion_batch,
    fallback_concept,
)


use_openai = True
use_openai_batch = True
max_text_batches = 2000
max_experts_to_label = None
activation_subbatch = 4096
layer_id = cfg.default_layer_id
concept_output_path = f"{cfg.concept_cache_dir}/concepts.jsonl"
progress_path = Path(cfg.concept_cache_dir) / "concept_progress.pt"
batch_state_path = Path(cfg.concept_cache_dir) / "llm_concept_batch.json"
batch_input_path = Path(cfg.concept_cache_dir) / "llm_concept_batch_input.jsonl"
batch_output_path = Path(cfg.concept_cache_dir) / "llm_concept_batch_output.jsonl"
Path(cfg.concept_cache_dir).mkdir(parents=True, exist_ok=True)
openai_client = openai_client_from_env(cfg.openai_env_key) if use_openai else None


def save_progress(state: dict) -> None:
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(progress_path)


store = TopExampleStore(top_n=cfg.concept_top_n, random_n=cfg.concept_low_n)
start_batch = 0
if progress_path.exists():
    progress_payload = torch.load(progress_path, map_location="cpu", weights_only=False)
    store = progress_payload.get("store", store)
    start_batch = int(progress_payload.get("next_batch_id", 0))
    print(f"resuming LLM concept extraction after batch {start_batch}")

if start_batch < max_text_batches:
    device = resolve_device(cfg.device)
    torch_dtype = dtype_from_name(cfg.dtype) if device.type == "cuda" else torch.float32
    payload = load_checkpoint(cfg.router_checkpoint, map_location=device)
    d_model = payload.get("extra", {}).get("d_model")
    n_latents = payload.get("extra", {}).get("n_latents")
    sae = RoutedSparseAutoencoder(d_model, n_latents, cfg.router_k, cfg.router_dim, cfg.score_chunk_size).to(device)
    sae.load_state_dict(payload["state_dict"])
    sae.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch_dtype, device_map=None).to(device)
    texts = iter_hf_texts(cfg.dataset_name, cfg.dataset_config, cfg.dataset_split, cfg.text_column, None)

    with torch.no_grad():
        text_batches = batched_texts(texts, cfg.cache_text_batch_size)
        text_batches = islice(text_batches, start_batch, max_text_batches)
        progress = tqdm(text_batches, total=max(0, max_text_batches - start_batch), desc="Extracting LLM concepts", unit="batch", dynamic_ncols=True)
        for batch_id, batch in enumerate(progress, start=start_batch):
            x_cpu, metadata = extract_llm_activations(model, tokenizer, batch, layer_id, cfg.max_length, device)
            if x_cpu.numel() == 0:
                save_progress({"store": store, "next_batch_id": batch_id + 1, "complete": False})
                continue
            for item in metadata[:16]:
                store.add_random(item)
            subbatches = range(0, x_cpu.shape[0], activation_subbatch)
            for start in tqdm(subbatches, desc="Scoring LLM activations", unit="subbatch", leave=False, dynamic_ncols=True):
                x = x_cpu[start:start + activation_subbatch].to(device)
                with autocast_context(device, cfg.dtype):
                    out = sae(x)
                indices = out["indices"].detach().cpu()
                values = out["values"].detach().cpu()
                for row in range(indices.shape[0]):
                    meta = metadata[start + row]
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
                progress.write(f"processed_text_batches={batch_id} discovered_experts={len(store.expert_ids())}")
            save_progress({"store": store, "next_batch_id": batch_id + 1, "complete": False})

    save_progress({"store": store, "next_batch_id": max_text_batches, "complete": True})

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
    batch_id = state.get("batch_id")
    if batch_id is None:
        requests = []
        for expert_id in eligible_expert_ids:
            examples = store.top_examples(expert_id)
            stats = store.stats(expert_id)
            custom_id = f"llm_expert_{expert_id}"
            messages = llm_concept_messages(expert_id, examples, store.random_examples, stats)
            requests.append(batch_chat_completion_request(custom_id, cfg.concept_llm_model, messages))
        batch = submit_chat_completion_batch(
            openai_client,
            requests,
            batch_input_path,
            {"description": "LLM SAE concept labeling", "project": cfg.project_name},
        )
        batch_state_path.write_text(
            json.dumps({"batch_id": batch.id, "input_path": str(batch_input_path), "total": len(requests)}, indent=2),
            encoding="utf-8",
        )
        print(f"submitted OpenAI batch {batch.id} with {len(requests)} LLM experts")
        print("rerun this script later to check status and collect completed labels")
        raise SystemExit(0)

    batch = openai_client.batches.retrieve(batch_id)
    counts = getattr(batch, "request_counts", None)
    print(f"OpenAI batch {batch.id} status={batch.status} counts={counts}")
    if batch.status != "completed":
        print("batch is not complete yet; rerun this script later")
        raise SystemExit(0)
    output_file_id = getattr(batch, "output_file_id", None)
    if output_file_id is None:
        raise RuntimeError(f"batch {batch.id} completed without output_file_id")
    batch_output_path.write_text(download_file_text(openai_client, output_file_id), encoding="utf-8")
    batch_rows = parse_batch_chat_completions(batch_output_path)
    label_progress = tqdm(eligible_expert_ids, desc="Collecting LLM batch labels", unit="expert", dynamic_ncols=True)
    for expert_id in label_progress:
        row = batch_rows.get(f"llm_expert_{expert_id}")
        if row is None:
            continue
        row["expert_id"] = expert_id
        row["activation_stats"] = store.stats(expert_id)
        row["top_examples"] = store.top_examples(expert_id)
        row["modality"] = "llm"
        row["layer_id"] = layer_id
        rows.append(row)
        save_jsonl(concept_output_path, rows)
        label_progress.write(f"labeled expert {expert_id}")
    save_jsonl(concept_output_path, rows)
    print(f"saved {concept_output_path}")
    raise SystemExit(0)

label_progress = tqdm(expert_ids, desc="Labeling LLM experts", unit="expert", dynamic_ncols=True)
for expert_id in label_progress:
    examples = store.top_examples(expert_id)
    stats = store.stats(expert_id)
    if stats["active_count"] < cfg.concept_min_examples:
        continue
    if use_openai:
        raise RuntimeError("Synchronous OpenAI labeling is disabled here; set use_openai_batch=True or use_openai=False.")
    else:
        row = fallback_concept(expert_id, "llm", examples, stats)
        row["top_examples"] = examples
    row["modality"] = "llm"
    row["layer_id"] = layer_id
    rows.append(row)
    save_jsonl(concept_output_path, rows)
    label_progress.write(f"labeled expert {expert_id}")

save_jsonl(concept_output_path, rows)
print(f"saved {concept_output_path}")
