from base64 import b64decode
from io import BytesIO
from pathlib import Path
import json
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.cv_clip_vit_b16 import cfg as cv_cfg
from configs.llm_gemma2_2b import cfg as llm_cfg
from concepts.concept_store import load_concepts_by_id
from models.routed_sae import RoutedSparseAutoencoder
from utils.checkpoints import load_checkpoint
from utils.cv_activations import get_clip_vision_layers, patch_box_from_index, patch_vision_layer_output
from utils.device import autocast_context, dtype_from_name, resolve_device
from utils.llm_activations import ActivationCapture, get_transformer_layers, last_nonpad_position, patch_layer_output

app = FastAPI(title="Neuron Atlas")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://127.0.0.1:4321"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = resolve_device(llm_cfg.device)
torch_dtype = dtype_from_name(llm_cfg.dtype) if device.type == "cuda" else torch.float32
llm_resources = None
cv_resources = None
llm_tokenizer = None
gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")


def run_on_gpu(func, *args):
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(gpu_executor, func, *args)


def load_backbone(model_cls, model_name: str):
    kwargs = {"torch_dtype": torch_dtype, "low_cpu_mem_usage": True}
    if device.type == "cuda":
        kwargs["device_map"] = str(device)
        model = model_cls.from_pretrained(model_name, **kwargs)
    else:
        model = model_cls.from_pretrained(model_name, **kwargs).to(device)
    model.eval()
    return model


def unload_cv_resources():
    global cv_resources
    if cv_resources is None:
        return
    cv_resources = None
    if device.type == "cuda":
        torch.cuda.empty_cache()


def unload_llm_resources():
    global llm_resources
    if llm_resources is None:
        return
    llm_resources = None
    if device.type == "cuda":
        torch.cuda.empty_cache()


def get_llm_tokenizer():
    global llm_tokenizer
    if llm_tokenizer is None:
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_cfg.model_name)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
    return llm_tokenizer


class TextRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)
    target_token: str | None = None
    max_experts: int = Field(default=12, ge=1, le=32)
    max_new_tokens: int = Field(default=1, ge=1, le=32)
    intervention: "ExpertIntervention | None" = None


class ImageRequest(BaseModel):
    image_data: str = Field(min_length=32)
    candidate_labels: list[str] = Field(default_factory=lambda: ["cat", "dog", "car", "person", "tree", "building", "food", "cow"])
    max_experts: int = Field(default=12, ge=1, le=32)
    intervention: "ExpertIntervention | None" = None


class ExpertIntervention(BaseModel):
    expert_id: int = Field(ge=0)
    multiplier: float = Field(gt=1.0, le=5.0)


def load_sae(checkpoint_path: str, router_k: int, router_dim: int, score_chunk_size: int):
    payload = load_checkpoint(PROJECT_ROOT / checkpoint_path, map_location=device)
    d_model = payload["extra"]["d_model"]
    n_latents = payload["extra"]["n_latents"]
    sae = RoutedSparseAutoencoder(d_model, n_latents, router_k, router_dim, score_chunk_size).to(device)
    sae.load_state_dict(payload["state_dict"])
    sae.eval()
    return sae


def get_llm_resources():
    global llm_resources
    if llm_resources is not None:
        return llm_resources
    unload_cv_resources()
    tokenizer = get_llm_tokenizer()
    model = load_backbone(AutoModelForCausalLM, llm_cfg.model_name)
    llm_resources = {
        "tokenizer": tokenizer,
        "model": model,
        "sae": load_sae(llm_cfg.router_checkpoint, llm_cfg.router_k, llm_cfg.router_dim, llm_cfg.score_chunk_size),
        "concepts": load_concepts_by_id(PROJECT_ROOT / llm_cfg.concept_cache_dir / "concepts.jsonl"),
    }
    return llm_resources


def get_cv_resources():
    global cv_resources
    if cv_resources is not None:
        return cv_resources
    unload_llm_resources()
    model = load_backbone(CLIPModel, cv_cfg.model_name)
    cv_resources = {
        "model": model,
        "processor": CLIPProcessor.from_pretrained(cv_cfg.model_name),
        "sae": load_sae(cv_cfg.router_checkpoint, cv_cfg.router_k, cv_cfg.router_dim, cv_cfg.score_chunk_size),
        "concepts": load_concepts_by_id(PROJECT_ROOT / cv_cfg.concept_cache_dir / "concepts.jsonl"),
    }
    return cv_resources


def patch_location(patch_index: int, patch_grid: int, image_width: int, image_height: int):
    left, top, right, bottom = patch_box_from_index(image_width, image_height, patch_index, patch_grid)
    return {
        "patch_index": patch_index,
        "patch_row": patch_index // patch_grid,
        "patch_col": patch_index % patch_grid,
        "patch_box": {
            "left": round(left / image_width * 100, 2),
            "top": round(top / image_height * 100, 2),
            "width": round((right - left) / image_width * 100, 2),
            "height": round((bottom - top) / image_height * 100, 2),
        },
    }


def concept_payload(
    concepts: dict[int, dict],
    expert_id: int,
    activation: float,
    effect: float,
    patch_index: int | None = None,
    patch_grid: int | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    token_index: int | None = None,
    token_text: str | None = None,
):
    concept = concepts.get(expert_id, {})
    evidence = [item.get("token_text", "").strip() for item in concept.get("top_examples", [])[:4] if item.get("token_text")]
    result = {
        "expert_id": expert_id,
        "label": concept.get("short_label", f"Unlabeled expert {expert_id}"),
        "description": concept.get("description", "No saved concept description."),
        "activation": activation,
        "causal_effect": effect,
        "mean_active_activation": concept.get("activation_stats", {}).get("mean_active_activation"),
        "max_training_activation": concept.get("activation_stats", {}).get("max_activation"),
        "active_count": concept.get("activation_stats", {}).get("active_count"),
        "evidence": evidence,
    }
    if patch_index is not None and patch_grid is not None and image_width is not None and image_height is not None:
        result.update(patch_location(patch_index, patch_grid, image_width, image_height))
    if token_index is not None:
        result["token_index"] = token_index
    if token_text is not None:
        result["token_text"] = token_text
    return result


def response_summary(
    modality: str,
    prediction: str,
    score: float,
    confidence: float,
    experts: list[dict],
    active_expert_count: int,
    layer_id: int,
    tokens: list[dict] | None = None,
    prediction_position: int | None = None,
    patch_grid: int | None = None,
    generated_text: str | None = None,
    intervention: dict | None = None,
):
    positive_effect = sum(max(item["causal_effect"], 0) for item in experts)
    top = max(experts, key=lambda item: abs(item["causal_effect"]), default=None)
    payload = {
        "modality": modality,
        "prediction": prediction,
        "score": score,
        "confidence": confidence,
        "active_expert_count": active_expert_count,
        "layer_id": layer_id,
        "experts": experts,
        "summary": {
            "positive_causal_effect": positive_effect,
            "top_driver": top["label"] if top else "No active expert",
            "top_driver_effect": top["causal_effect"] if top else 0.0,
        },
    }
    if tokens is not None:
        payload["tokens"] = tokens
    if prediction_position is not None:
        payload["prediction_position"] = prediction_position
    if patch_grid is not None:
        payload["patch_grid"] = patch_grid
    if generated_text is not None:
        payload["generated_text"] = generated_text
    if intervention is not None:
        payload["intervention"] = intervention
    return payload


def expert_boost_patch(
    sae: RoutedSparseAutoencoder,
    expert_id: int,
    multiplier: float,
    dtype_name: str,
    start_index: int = 0,
):
    if expert_id >= sae.n_latents:
        raise HTTPException(status_code=422, detail="The selected expert is outside this SAE's latent range.")
    decoder_vector = sae.decoder[expert_id].detach()

    def patch_fn(layer_output: torch.Tensor) -> torch.Tensor:
        patched = layer_output.clone()
        active_hidden = patched[:, start_index:, :]
        flat_hidden = active_hidden.reshape(-1, active_hidden.shape[-1])
        with autocast_context(device, dtype_name):
            sae_out = sae(flat_hidden)
        matched_values = sae_out["values"] * (sae_out["indices"] == expert_id).to(sae_out["values"].dtype)
        activations = matched_values.sum(dim=-1, keepdim=True)
        contribution = activations * decoder_vector.to(device=active_hidden.device, dtype=active_hidden.dtype)
        patched[:, start_index:, :] = active_hidden + (multiplier - 1.0) * contribution.reshape_as(active_hidden)
        return patched

    return patch_fn


@app.get("/api/health")
def health():
    return {"status": "ready", "device": str(device)}


@app.get("/api/demo/{modality}")
def demo(modality: str):
    filename = "llm_explanation.json" if modality == "text" else "cv_explanation.json"
    path = PROJECT_ROOT / "artifacts" / "explanations" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="No saved explanation is available.")
    saved = json.loads(path.read_text(encoding="utf-8"))
    experts = []
    for item in saved["top_concept_activations"]:
        effect = item.get("delta_logprob_when_ablated", item.get("delta_clip_score_when_ablated", 0.0))
        expert = {
            "expert_id": item["expert_id"],
            "label": item.get("short_label", f"Unlabeled expert {item['expert_id']}"),
            "description": item.get("description", "No saved concept description."),
            "activation": item["activation"],
            "causal_effect": effect,
            "mean_active_activation": None,
            "max_training_activation": None,
            "active_count": None,
            "evidence": [],
        }
        if "patch_index" in item:
            patch_index = int(item["patch_index"])
            patch_grid = int(saved.get("patch_grid", 14))
            expert.update(patch_location(patch_index, patch_grid, 224, 224))
        if "token_index" in item:
            expert["token_index"] = item["token_index"]
        if "token_text" in item:
            expert["token_text"] = item["token_text"]
        experts.append(expert)
    prediction = saved.get("target_token", saved.get("predicted_label", "")).strip()
    score = saved.get("original_logprob", saved.get("original_score", 0.0))
    confidence = float(torch.exp(torch.tensor(score)).item()) if modality == "text" else 1.0
    tokens = saved.get("tokens")
    prediction_position = saved.get("prediction_position")
    patch_grid = saved.get("patch_grid", 14 if modality == "image" else None)
    if modality == "text" and tokens is None and saved.get("prompt"):
        tokenizer = get_llm_tokenizer()
        encoded = tokenizer(saved["prompt"], return_tensors="pt")
        input_ids = encoded["input_ids"][0].tolist()
        mask = encoded["attention_mask"][0].tolist()
        tokens = [
            {"index": index, "text": tokenizer.decode([token_id])}
            for index, (token_id, active) in enumerate(zip(input_ids, mask))
            if active
        ]
        if prediction_position is None:
            prediction_position = int(last_nonpad_position(encoded["attention_mask"])[0])
    return response_summary(
        modality,
        prediction,
        score,
        confidence,
        experts,
        len(experts),
        saved["layer_id"],
        tokens=tokens,
        prediction_position=prediction_position,
        patch_grid=patch_grid,
    )


@app.post("/api/infer/text")
async def infer_text(request: TextRequest):
    return await run_on_gpu(_infer_text, request)


def _infer_text(request: TextRequest):
    resources = get_llm_resources()
    tokenizer = resources["tokenizer"]
    model = resources["model"]
    sae = resources["sae"]
    encoded = tokenizer(request.prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    module = get_transformer_layers(model)[llm_cfg.default_layer_id]
    with torch.no_grad():
        with ActivationCapture(module) as capture:
            original_logits = model(**encoded).logits
        position = int(last_nonpad_position(encoded["attention_mask"])[0])
        if request.target_token:
            token_ids = tokenizer.encode(request.target_token, add_special_tokens=False)
            if not token_ids:
                raise HTTPException(status_code=422, detail="The requested target token cannot be encoded.")
            target_id = int(token_ids[0])
        else:
            target_id = int(original_logits[0, position].argmax())
        original_logprob = F.log_softmax(original_logits[0, position].float(), dim=-1)[target_id]
        generation_options = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
        }
        input_length = encoded["input_ids"].shape[1]
        baseline_generated_ids = model.generate(**encoded, **generation_options)[0, input_length:]
        baseline_generated_text = tokenizer.decode(baseline_generated_ids, skip_special_tokens=True)
        output_logits = original_logits
        intervention_payload = None
        if request.intervention is not None:
            boost_patch = expert_boost_patch(
                sae,
                request.intervention.expert_id,
                request.intervention.multiplier,
                llm_cfg.dtype,
            )
            with patch_layer_output(module, boost_patch):
                output_logits = model(**encoded).logits
            with patch_layer_output(module, boost_patch):
                generated_ids = model.generate(**encoded, **generation_options)[0, input_length:]
            prediction_id = int(output_logits[0, position].argmax())
            intervention_payload = {
                "expert_id": request.intervention.expert_id,
                "multiplier": request.intervention.multiplier,
                "baseline_prediction": tokenizer.decode([int(original_logits[0, position].argmax())]).strip(),
                "baseline_generated_text": baseline_generated_text,
                "prediction_changed": prediction_id != int(original_logits[0, position].argmax()),
                "scope": "All active token positions during inference and generation",
            }
        else:
            generated_ids = baseline_generated_ids
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        prediction_id = int(output_logits[0, position].argmax())
        prediction_logprob = F.log_softmax(output_logits[0, position].float(), dim=-1)[prediction_id]
        hidden = capture.value.detach().to(device)
        token_count = hidden.shape[1]
        with autocast_context(device, llm_cfg.dtype):
            sae_out = sae(hidden.reshape(-1, hidden.shape[-1]))
        best_by_expert = {}
        indices = sae_out["indices"].detach().cpu()
        values = sae_out["values"].detach().cpu()
        for row in range(indices.shape[0]):
            token_index = row % token_count
            for slot in range(indices.shape[1]):
                expert_id = int(indices[row, slot])
                activation = float(values[row, slot])
                if activation <= 0:
                    continue
                existing = best_by_expert.get(expert_id)
                if existing is None or activation > existing["activation"]:
                    best_by_expert[expert_id] = {"activation": activation, "token_index": token_index}
        selected = sorted(best_by_expert.items(), key=lambda item: item[1]["activation"], reverse=True)[:request.max_experts]
        input_ids = encoded["input_ids"][0].tolist()
        mask = encoded["attention_mask"][0].tolist()
        tokens = [
            {"index": index, "text": tokenizer.decode([token_id])}
            for index, (token_id, active) in enumerate(zip(input_ids, mask))
            if active
        ]
        experts = []
        for expert_id, item in selected:
            token_index = item["token_index"]
            activation = item["activation"]
            token_text = tokenizer.decode([input_ids[token_index]])
            decoder_vector = sae.decoder[expert_id].detach() * activation
            def patch_fn(layer_output, decoder_vector=decoder_vector, token_index=token_index):
                patched = layer_output.clone()
                patched[0, token_index, :] = patched[0, token_index, :] - decoder_vector.to(device=layer_output.device, dtype=layer_output.dtype)
                return patched
            with patch_layer_output(module, patch_fn):
                patched_logits = model(**encoded).logits
            patched_logprob = F.log_softmax(patched_logits[0, position].float(), dim=-1)[target_id]
            experts.append(
                concept_payload(
                    resources["concepts"],
                    expert_id,
                    activation,
                    float((original_logprob - patched_logprob).item()),
                    token_index=token_index,
                    token_text=token_text,
                )
            )
    return response_summary(
        "text",
        tokenizer.decode([prediction_id]).strip(),
        float(prediction_logprob.item()),
        float(torch.exp(prediction_logprob).item()),
        experts,
        len(best_by_expert),
        llm_cfg.default_layer_id,
        tokens=tokens,
        prediction_position=position,
        generated_text=generated_text,
        intervention=intervention_payload,
    )


@app.post("/api/infer/image")
async def infer_image(request: ImageRequest):
    return await run_on_gpu(_infer_image, request)


def _infer_image(request: ImageRequest):
    resources = get_cv_resources()
    if not request.candidate_labels:
        raise HTTPException(status_code=422, detail="At least one candidate label is required.")
    encoded_image = request.image_data.split(",", 1)[-1]
    image = Image.open(BytesIO(b64decode(encoded_image))).convert("RGB")
    prompts = [f"a photo of a {label}" for label in request.candidate_labels]
    inputs = resources["processor"](text=prompts, images=image, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model = resources["model"]
    sae = resources["sae"]
    module = get_clip_vision_layers(model)[cv_cfg.default_layer_id]
    with torch.no_grad():
        with ActivationCapture(module) as capture:
            original = model(**inputs)
        logits = original.logits_per_image[0].float()
        baseline_prediction_index = int(logits.argmax())
        original_score = logits[baseline_prediction_index]
        output_logits = logits
        intervention_payload = None
        if request.intervention is not None:
            boost_patch = expert_boost_patch(
                sae,
                request.intervention.expert_id,
                request.intervention.multiplier,
                cv_cfg.dtype,
                start_index=1,
            )
            with patch_vision_layer_output(module, boost_patch):
                boosted = model(**inputs)
            output_logits = boosted.logits_per_image[0].float()
            boosted_prediction_index = int(output_logits.argmax())
            intervention_payload = {
                "expert_id": request.intervention.expert_id,
                "multiplier": request.intervention.multiplier,
                "baseline_prediction": request.candidate_labels[baseline_prediction_index],
                "baseline_score": float(original_score.item()),
                "prediction_changed": boosted_prediction_index != baseline_prediction_index,
                "scope": "All active image patches",
            }
        hidden = capture.value[:, 1:, :]
        patch_count = hidden.shape[1]
        patch_grid = int(patch_count ** 0.5)
        image_width, image_height = image.size
        with autocast_context(device, cv_cfg.dtype):
            sae_out = sae(hidden.reshape(-1, hidden.shape[-1]).to(device))
        best_by_expert = {}
        indices = sae_out["indices"].detach().cpu()
        values = sae_out["values"].detach().cpu()
        for row in range(indices.shape[0]):
            patch_index = row % patch_count
            for slot in range(indices.shape[1]):
                expert_id = int(indices[row, slot])
                activation = float(values[row, slot])
                if activation <= 0:
                    continue
                existing = best_by_expert.get(expert_id)
                if existing is None or activation > existing["activation"]:
                    best_by_expert[expert_id] = {"activation": activation, "patch_index": patch_index}
        selected = sorted(best_by_expert.items(), key=lambda item: item[1]["activation"], reverse=True)[:request.max_experts]
        experts = []
        for expert_id, item in selected:
            decoder_vector = sae.decoder[expert_id].detach() * item["activation"]
            def patch_fn(layer_output, decoder_vector=decoder_vector, patch_index=item["patch_index"]):
                patched = layer_output.clone()
                patched[0, patch_index + 1, :] = patched[0, patch_index + 1, :] - decoder_vector.to(device=layer_output.device, dtype=layer_output.dtype)
                return patched
            with patch_vision_layer_output(module, patch_fn):
                patched = model(**inputs)
            patched_score = patched.logits_per_image[0].float()[baseline_prediction_index]
            experts.append(
                concept_payload(
                    resources["concepts"],
                    expert_id,
                    item["activation"],
                    float((original_score - patched_score).item()),
                    patch_index=item["patch_index"],
                    patch_grid=patch_grid,
                    image_width=image_width,
                    image_height=image_height,
                )
            )
        prediction_index = int(output_logits.argmax())
        prediction_score = output_logits[prediction_index]
        confidence = float(F.softmax(output_logits, dim=-1)[prediction_index].item())
    return response_summary(
        "image",
        request.candidate_labels[prediction_index],
        float(prediction_score.item()),
        confidence,
        experts,
        len(best_by_expert),
        cv_cfg.default_layer_id,
        patch_grid=patch_grid,
        intervention=intervention_payload,
    )


DIST_DIR = Path(__file__).resolve().parent / "dist"
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="ui")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8787)
