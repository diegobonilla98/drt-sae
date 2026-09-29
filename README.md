# DRT-SAE: distilled routed Top-K sparse autoencoders

This repository is an experimental implementation of a routed sparse autoencoder (SAE) for inspecting activations in language and vision models. It combines a dense Top-K SAE teacher, a learned router that tries to reproduce the teacher's selected features, LLM-assisted feature labels, and inference-time feature ablations. A small Neuron Atlas web UI can inspect saved explanations or run local inference when the required model and SAE checkpoints are available.

The code explores an idea; it does not establish that the learned features are reliable explanations. The current saved examples are mixed, and the vision ablation is especially weak. See [What the current results say](#what-the-current-results-say) before interpreting the demos.

## Current scope

- **Language:** Gemma 2 2B residual-stream activations at layer 13, with Pythia 410M available as a config fallback. The configured training corpus is OpenWebText.
- **Vision:** CLIP ViT-B/16 patch-token activations at vision layer 9. The configured cache uses a local COCO train2017 image folder.
- **Sparse model:** the teacher chooses its top 64 (LLM) or top 32 (vision) activations. The routed model selects 32 (LLM) or 16 (vision) experts before evaluating only those encoder vectors.
- **Feature labels:** top examples are passed to a chat model (`gpt-5-mini` by default) or retained unlabeled when API labeling is disabled. These labels are hypotheses generated from selected examples, not human-verified concepts.
- **Interventions:** subtract a feature's decoded contribution at a token or patch and measure the change in a target-token log probability or CLIP candidate score.
- **Web UI:** React/Vite frontend with a FastAPI backend for exploring token/patch activations and applying feature-amplification interventions.

## What the current results say

The development machine has two saved inference examples under `artifacts/explanations/`; generated artifacts are intentionally excluded from Git. They are useful for understanding the output format, not as a benchmark.

| Saved example | Result | Interpretation |
| --- | --- | --- |
| Gemma prompt `The Eiffel Tower is located in the city of` | The predicted token was ` Paris` (baseline log probability `-0.0554`). Among 12 selected experts, ablating 8 lowered the target log probability, while ablating 4 raised it. The largest decrease was `0.0276`; the largest increase was `0.0121`. | Some features affect this one prediction in the expected direction, while others have an opposite or confounding effect. Several generated labels look lexical or syntactic rather than a clean semantic explanation. One prompt cannot establish generality or faithfulness. |
| CLIP image `examples/cow.jpg` | CLIP ranked `cow` highest among eight supplied labels (score `31.0`). Of the 12 selected experts, 10 had exactly zero measured score change; ablating the other two increased the cow score by `0.125`. | This example provides no evidence that the selected experts positively explain the winning class. A negative delta means the score increased after ablation. |

The CV value is a raw CLIP similarity logit for the chosen candidate list, not a probability. The LLM values are next-token log probabilities. The probes inspect only selected active features at one token or one patch, and do not compare against random-feature controls, alternative prompts/images, or a causal benchmark. Concept names are model-generated and may be inaccurate. No general claim about interpretability, routing quality, compute savings, or predictive performance should be inferred from these demos.

The repository does not include a published training/evaluation report with teacher-vs-router reconstruction metrics, routing recall, load balance, latency, or controlled ablation statistics. The training scripts log some reconstruction and load metrics during a run, but the available demo JSON files are not a substitute for those evaluations. Treat the method and results as a work in progress.

## Repository map

```text
configs/       LLM and CLIP model, layer, cache, and training settings
models/        dense Top-K teacher and routed sparse autoencoder
training/      activation caching and teacher/router training scripts
concepts/      top-example collection, contact sheets, labels, and stores
inference/     token and image score-ablation demonstrations
data_sources/  local text/image loaders and activation shard storage
utils/         hooks, checkpoint helpers, metrics, losses, and device helpers
webui/         React/Vite Neuron Atlas and FastAPI inference API
examples/      small sample image used by the saved vision demo
tests/         synthetic model smoke check and UI intervention checks
```

## Setup

Use Python 3.10 or newer. Install a PyTorch build that matches your operating system and CUDA version using the [official PyTorch install selector](https://pytorch.org/get-started/locally/). Then install the Python packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS, activate with `source .venv/bin/activate` instead. For GPU runs, verify `torch.cuda.is_available()` in the environment before caching or training. Model and dataset downloads may require access approval from their respective providers; Gemma 2 access is gated by Hugging Face.

Run the inexpensive synthetic checks from the repository root:

```powershell
python tests/smoke_test.py
python tests/webui_test.py
```

The smoke test exercises tensor shapes, a forward pass, loss construction, backpropagation, and the router-bias update on random data. It does not train the SAE or validate explanation quality.

## Training workflow

Configuration is edited directly in `configs/llm_gemma2_2b.py` and `configs/cv_clip_vit_b16.py`. Scripts run their configured workloads on import, so review the config before running any of them. The defaults are substantial GPU jobs, not smoke-test settings.

### Language model

1. Set the model, layer, data source, batch size, and step counts in `configs/llm_gemma2_2b.py`.
2. Cache residual-stream activations:

   ```powershell
   python training/cache_llm_activations.py
   ```

3. Train the dense Top-K teacher, then distill and fine-tune the router:

   ```powershell
   python training/train_llm_teacher.py
   python training/train_llm_router.py
   ```

4. Extract feature examples/labels and run the saved next-token probe:

   ```powershell
   python concepts/extract_llm_concepts.py
   python inference/explain_llm_prediction.py
   ```

The default activation cache, checkpoints, concepts, and explanation paths are under `artifacts/`; those files are local outputs and are not part of the Git repository.

### Vision model

Set `image_dataset_root` in `configs/cv_clip_vit_b16.py` to a local image directory. Defaults point to `/mnt/hdd/coco2017/train2017`, which is not portable to Windows. Then run:

```powershell
python training/cache_cv_activations.py
python training/train_cv_teacher.py
python training/train_cv_router.py
python concepts/extract_cv_concepts.py
python inference/explain_cv_prediction.py
```

The default cache is limited to 25,000 images and five hours; rerunning resumes from its progress file. The concept extraction stage may make paid OpenAI Batch API calls if labeling is enabled. Keep `OPENAI_API_KEY` in the process environment; never put it in source or commit it. The saved vision demonstration uses the small local `examples/cow.jpg` image and does not require downloading COCO just to inspect the example's source image.

Both pipelines expect activation caches and teacher checkpoints from their earlier stages. They do not run end-to-end from an empty machine without model/dataset downloads, a suitable device, and the configured artifacts.

## Neuron Atlas web UI

Install the frontend dependencies once:

```powershell
Set-Location webui
npm ci
Set-Location ..
```

Start the API and Vite frontend in separate terminals from the repository root:

```powershell
python webui/server.py
```

```powershell
Set-Location webui
npm run dev
```

Open the Vite URL printed by npm (the configured default is `http://localhost:4321`). Vite proxies `/api` to the backend at `http://localhost:8787` by default; configure `webui/vite.config.js` and the server settings together if changing ports. The API can also serve a previously built frontend:

```powershell
Set-Location webui
npm ci
npm run build
Set-Location ..
python webui/server.py
```

The backend's default port is `8787`. Live text/image inference requires the configured Hugging Face models, compatible local routed-SAE checkpoints, and concept JSONL files. Without those artifacts, the UI can only show a saved demo when one is available. Amplification is an intervention on the decoded contribution; a changed output does not by itself establish a faithful explanation.

## Method outline

For input activation `x`, the teacher applies input normalization, computes dense nonnegative activations, retains its top `K_teacher`, and decodes their weighted feature vectors. The routed model instead computes a router query and scores expert keys:

```text
h = LayerNorm(x)
q = Wq h
score_i = q · key_i / sqrt(router_dim) + bias_i
S = TopK(score, K_router)
a_i = ReLU(encoder_i · h + encoder_bias_i), for i in S
x_hat = decoder_bias + sum(i in S) a_i decoder_i
```

Router scores are evaluated in chunks when the latent dictionary is larger than `score_chunk_size`. The router is initialized from a dense teacher and trained with reconstruction loss, sampled route distillation, a load-balancing term, and a coefficient penalty. A separate bias update adjusts expert routing frequency. See `models/`, `utils/losses.py`, `utils/load_balance.py`, and `training/` for the implementation details.

The inference scripts identify the active features at a target token or image patch, subtract each feature's activation-scaled decoder vector from that position, rerun the original model, and compare the same target's score. For language, `delta_logprob_when_ablated = original_logprob - patched_logprob`; for vision, `delta_clip_score_when_ablated = original_score - patched_score`. A positive delta means ablation lowered the target score. A negative delta means ablation raised it. Zero means no measurable difference at the stored precision for that probe.

## Data, checkpoints, and privacy

Raw datasets, activation shards, progress files, model checkpoints, generated concepts, API batch payloads/results, and inference outputs belong in local `artifacts/` or `data/` directories and are ignored by Git. Do not commit model weights, private datasets, API keys, or batch payloads. Use public/small reproducible examples for reports, and review any concept evidence before sharing it: contexts or images can contain sensitive material.

`requirements.txt` covers Python libraries, while the separate `webui/package-lock.json` pins the frontend toolchain. The frontend `webui/dist/`, `webui/node_modules/`, local logs, Python caches, and virtual environments are generated/local files and are ignored.
