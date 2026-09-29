from pathlib import Path
import json
import sys
import torch
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.cv_clip_vit_b16 import cfg
from data_sources.image_sources import load_rgb_image
from models.routed_sae import RoutedSparseAutoencoder
from utils.checkpoints import load_checkpoint
from utils.device import resolve_device, dtype_from_name, autocast_context
from utils.cv_activations import get_clip_vision_layers, patch_vision_layer_output
from utils.llm_activations import ActivationCapture
from concepts.concept_store import load_concepts_by_id


image_path = "examples/cow.jpg"
candidate_labels = ["cat", "dog", "car", "person", "tree", "building", "food", "cow"]
concept_path = f"{cfg.concept_cache_dir}/concepts.jsonl"
output_path = "artifacts/explanations/cv_explanation.json"
max_explained_concepts = 12
layer_id = cfg.default_layer_id


device = resolve_device(cfg.device)
torch_dtype = dtype_from_name(cfg.dtype) if device.type == "cuda" else torch.float32
concepts = load_concepts_by_id(concept_path)
sae_payload = load_checkpoint(cfg.router_checkpoint, map_location=device)
d_model = sae_payload["extra"]["d_model"]
n_latents = sae_payload["extra"]["n_latents"]
sae = RoutedSparseAutoencoder(d_model, n_latents, cfg.router_k, cfg.router_dim, cfg.score_chunk_size).to(device)
sae.load_state_dict(sae_payload["state_dict"])
sae.eval()
model = CLIPModel.from_pretrained(cfg.model_name, torch_dtype=torch_dtype).to(device)
processor = CLIPProcessor.from_pretrained(cfg.model_name)
model.eval()
image = load_rgb_image(image_path)
prompts = [f"a photo of a {label}" for label in candidate_labels]
inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
layers = get_clip_vision_layers(model)
module = layers[layer_id]

with torch.no_grad():
    with ActivationCapture(module) as capture:
        original = model(**inputs)
    logits = original.logits_per_image[0].float()
    pred_index = int(logits.argmax())
    pred_label = candidate_labels[pred_index]
    original_score = logits[pred_index]
    hidden = capture.value[:, 1:, :]
    patch_count = hidden.shape[1]
    x = hidden.reshape(-1, hidden.shape[-1]).to(device)
    with autocast_context(device, cfg.dtype):
        sae_out = sae(x)
    indices = sae_out["indices"].detach().cpu()
    values = sae_out["values"].detach().cpu()
    best_by_expert = {}
    for row in range(indices.shape[0]):
        patch_index = row % patch_count
        for slot in range(indices.shape[1]):
            expert_id = int(indices[row, slot])
            value = float(values[row, slot])
            if value <= 0:
                continue
            old = best_by_expert.get(expert_id)
            if old is None or value > old["activation"]:
                best_by_expert[expert_id] = {"expert_id": expert_id, "activation": value, "patch_index": patch_index, "rank": slot}
    candidates = sorted(best_by_expert.values(), key=lambda x: x["activation"], reverse=True)[:max_explained_concepts]
    rows = []
    for item in candidates:
        expert_id = item["expert_id"]
        patch_index = item["patch_index"]
        value = item["activation"]
        decoder_vector = sae.decoder[expert_id].detach() * value
        def patch_fn(hidden, decoder_vector=decoder_vector, patch_index=patch_index):
            patched = hidden.clone()
            delta = decoder_vector.to(device=hidden.device, dtype=hidden.dtype)
            patched[0, patch_index + 1, :] = patched[0, patch_index + 1, :] - delta
            return patched
        with patch_vision_layer_output(module, patch_fn):
            patched = model(**inputs)
        patched_score = patched.logits_per_image[0].float()[pred_index]
        delta = original_score - patched_score
        concept = concepts.get(expert_id, {"short_label": f"unlabeled_expert_{expert_id}", "description": "No saved concept description."})
        rows.append(
            {
                "expert_id": expert_id,
                "activation": value,
                "patch_index": patch_index,
                "short_label": concept.get("short_label"),
                "description": concept.get("description"),
                "delta_clip_score_when_ablated": float(delta.item()),
                "patched_score": float(patched_score.item()),
            }
        )

explanation = {
    "image_path": image_path,
    "candidate_labels": candidate_labels,
    "predicted_label": pred_label,
    "original_score": float(original_score.item()),
    "layer_id": layer_id,
    "top_concept_activations": rows,
}
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
Path(output_path).write_text(json.dumps(explanation, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(explanation, indent=2, ensure_ascii=False))
print(f"saved {output_path}")
