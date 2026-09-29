from pathlib import Path
import json
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.llm_gemma2_2b import cfg
from models.routed_sae import RoutedSparseAutoencoder
from utils.checkpoints import load_checkpoint
from utils.device import resolve_device, dtype_from_name, autocast_context
from utils.llm_activations import get_transformer_layers, ActivationCapture, patch_layer_output, last_nonpad_position
from concepts.concept_store import load_concepts_by_id


prompt = "The Eiffel Tower is located in the city of"
target_token_text = None
concept_path = f"{cfg.concept_cache_dir}/concepts.jsonl"
output_path = "artifacts/explanations/llm_explanation.json"
max_explained_concepts = 12
layer_id = cfg.default_layer_id


def next_token_logprob(model, encoded, token_id: int):
    logits = model(**encoded).logits
    pos = int(last_nonpad_position(encoded["attention_mask"])[0])
    return F.log_softmax(logits[0, pos].float(), dim=-1)[token_id], logits


device = resolve_device(cfg.device)
torch_dtype = dtype_from_name(cfg.dtype) if device.type == "cuda" else torch.float32
concepts = load_concepts_by_id(concept_path)
sae_payload = load_checkpoint(cfg.router_checkpoint, map_location=device)
d_model = sae_payload["extra"]["d_model"]
n_latents = sae_payload["extra"]["n_latents"]
sae = RoutedSparseAutoencoder(d_model, n_latents, cfg.router_k, cfg.router_dim, cfg.score_chunk_size).to(device)
sae.load_state_dict(sae_payload["state_dict"])
sae.eval()
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch_dtype, device_map=None).to(device)
model.eval()
encoded = tokenizer(prompt, return_tensors="pt")
encoded = {k: v.to(device) for k, v in encoded.items()}
layers = get_transformer_layers(model)
module = layers[layer_id]

with torch.no_grad():
    with ActivationCapture(module) as capture:
        original_logits = model(**encoded).logits
    pos = int(last_nonpad_position(encoded["attention_mask"])[0])
    if target_token_text is None:
        target_id = int(original_logits[0, pos].argmax())
    else:
        ids = tokenizer.encode(target_token_text, add_special_tokens=False)
        target_id = int(ids[0])
    target_text = tokenizer.decode([target_id])
    original_logprob = F.log_softmax(original_logits[0, pos].float(), dim=-1)[target_id]
    x = capture.value[0, pos].detach().to(device).unsqueeze(0)
    with autocast_context(device, cfg.dtype):
        sae_out = sae(x)
    active_indices = sae_out["indices"][0].detach().cpu().tolist()
    active_values = sae_out["values"][0].detach().cpu().tolist()
    candidates = []
    for expert_id, value in zip(active_indices, active_values):
        if value > 0:
            candidates.append((expert_id, float(value)))
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:max_explained_concepts]
    rows = []
    for expert_id, value in candidates:
        decoder_vector = sae.decoder[expert_id].detach() * value
        def patch_fn(hidden, decoder_vector=decoder_vector):
            patched = hidden.clone()
            delta = decoder_vector.to(device=hidden.device, dtype=hidden.dtype)
            patched[0, pos, :] = patched[0, pos, :] - delta
            return patched
        with patch_layer_output(module, patch_fn):
            patched_logits = model(**encoded).logits
        patched_logprob = F.log_softmax(patched_logits[0, pos].float(), dim=-1)[target_id]
        delta = original_logprob - patched_logprob
        concept = concepts.get(expert_id, {"short_label": f"unlabeled_expert_{expert_id}", "description": "No saved concept description."})
        rows.append(
            {
                "expert_id": expert_id,
                "activation": value,
                "short_label": concept.get("short_label"),
                "description": concept.get("description"),
                "delta_logprob_when_ablated": float(delta.item()),
                "patched_logprob": float(patched_logprob.item()),
            }
        )

explanation = {
    "prompt": prompt,
    "target_token": target_text,
    "target_token_id": target_id,
    "original_logprob": float(original_logprob.item()),
    "layer_id": layer_id,
    "top_concept_activations": rows,
}
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
Path(output_path).write_text(json.dumps(explanation, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(explanation, indent=2, ensure_ascii=False))
print(f"saved {output_path}")
