from pathlib import Path
import sys
import torch
from tqdm.auto import trange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.cv_clip_vit_b16 import cfg
from data_sources.activation_store import iterate_activation_batches
from models.teacher_sae import TopKSAETeacher
from models.routed_sae import RoutedSparseAutoencoder
from utils.checkpoints import load_checkpoint, save_checkpoint
from utils.device import resolve_device, autocast_context
from utils.metrics import mse_loss, r2_score, cosine_recon, selected_l0
from utils.losses import coefficient_loss, sampled_route_distillation_loss
from utils.load_balance import load_loss, normalized_load_entropy, update_router_bias_


device = resolve_device(cfg.device)
batches = iterate_activation_batches(cfg.activation_cache_dir, cfg.train_batch_tokens, str(device))
first_batch = next(batches)
teacher_payload = load_checkpoint(cfg.teacher_checkpoint, map_location=device)
d_model = teacher_payload.get("extra", {}).get("d_model", cfg.d_model or first_batch.shape[-1])
n_latents = teacher_payload.get("extra", {}).get("n_latents", cfg.n_latents_factor * d_model)
teacher = TopKSAETeacher(d_model=d_model, n_latents=n_latents, k=cfg.teacher_k).to(device)
teacher.load_state_dict(teacher_payload["state_dict"])
teacher.eval()
model = RoutedSparseAutoencoder(
    d_model=d_model,
    n_latents=n_latents,
    k=cfg.router_k,
    router_dim=cfg.router_dim,
    score_chunk_size=cfg.score_chunk_size,
).to(device)
model.initialize_from_teacher(teacher)
optimizer = torch.optim.AdamW(
    [
        {"params": list(model.query.parameters()) + [model.router_keys, model.encoder_vectors, model.encoder_bias], "lr": cfg.lr_router},
        {"params": [model.decoder, model.decoder_bias], "lr": cfg.lr_decoder},
    ],
    weight_decay=cfg.weight_decay,
)
resume_payload = load_checkpoint(cfg.router_checkpoint, map_location=device) if Path(cfg.router_checkpoint).exists() else None
start_step = 0
if resume_payload is not None:
    model.load_state_dict(resume_payload["state_dict"])
    start_step = int(resume_payload.get("step", 0))
    optimizer_state = resume_payload.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    print(f"resuming {cfg.router_checkpoint} from step {start_step}")

if start_step >= cfg.router_steps:
    print(f"{cfg.router_checkpoint} is already at step {start_step}; target is {cfg.router_steps}")
    raise SystemExit(0)

progress = trange(start_step + 1, cfg.router_steps + 1, desc="Training CV router", unit="step", dynamic_ncols=True)
for step in progress:
    x = first_batch if step == start_step + 1 else next(batches)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, cfg.dtype):
        with torch.no_grad():
            teacher_out = teacher(x)
        out = model(x)
        rec = mse_loss(x, out["recon"])
        route = sampled_route_distillation_loss(model, x, teacher_out["indices"], teacher_out["values"], out["indices"], cfg.route_negative_count)
        load = load_loss(out["indices"], model.n_latents)
        coef = coefficient_loss(out["values"])
        loss = rec + cfg.route_loss_weight * route + cfg.load_loss_weight * load + cfg.coef_loss_weight * coef
    loss.backward()
    optimizer.step()
    model.normalize_decoder_()
    model.normalize_router_keys_()
    update_router_bias_(model, out["indices"].detach(), cfg.router_bias_rate)
    if step % cfg.log_every == 0:
        with torch.no_grad():
            score = r2_score(x, out["recon"]).item()
            cos = cosine_recon(x, out["recon"]).item()
            l0 = selected_l0(out["values"]).item()
            entropy = normalized_load_entropy(out["indices"], model.n_latents).item()
        progress.set_postfix(loss=f"{loss.item():.6f}", rec=f"{rec.item():.6f}", route=f"{route.item():.6f}", load=f"{load.item():.6f}", r2=f"{score:.4f}", entropy=f"{entropy:.4f}")
        progress.write(f"step={step} loss={loss.item():.6f} rec={rec.item():.6f} route={route.item():.6f} load={load.item():.6f} r2={score:.4f} cos={cos:.4f} l0={l0:.2f} load_entropy={entropy:.4f}")
    if step % cfg.save_every == 0:
        save_checkpoint(cfg.router_checkpoint, model, cfg, step, {"d_model": d_model, "n_latents": n_latents}, optimizer)

save_checkpoint(cfg.router_checkpoint, model, cfg, cfg.router_steps, {"d_model": d_model, "n_latents": n_latents}, optimizer)
print(f"saved {cfg.router_checkpoint}")
