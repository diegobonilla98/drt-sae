from pathlib import Path
import sys
import torch
from tqdm.auto import trange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.cv_clip_vit_b16 import cfg
from data_sources.activation_store import iterate_activation_batches
from models.teacher_sae import TopKSAETeacher
from utils.checkpoints import load_checkpoint, save_checkpoint
from utils.device import resolve_device, autocast_context
from utils.metrics import mse_loss, r2_score, cosine_recon, selected_l0
from utils.losses import coefficient_loss


device = resolve_device(cfg.device)
batches = iterate_activation_batches(cfg.activation_cache_dir, cfg.train_batch_tokens, str(device))
first_batch = next(batches)
resume_payload = load_checkpoint(cfg.teacher_checkpoint, map_location=device) if Path(cfg.teacher_checkpoint).exists() else None
d_model = (resume_payload or {}).get("extra", {}).get("d_model", cfg.d_model or first_batch.shape[-1])
n_latents = (resume_payload or {}).get("extra", {}).get("n_latents", cfg.n_latents_factor * d_model)
model = TopKSAETeacher(d_model=d_model, n_latents=n_latents, k=cfg.teacher_k).to(device)
optimizer = torch.optim.AdamW(
    [
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_teacher},
        {"params": [model.decoder, model.decoder_bias], "lr": cfg.lr_decoder},
    ],
    weight_decay=cfg.weight_decay,
)
start_step = 0
if resume_payload is not None:
    model.load_state_dict(resume_payload["state_dict"])
    start_step = int(resume_payload.get("step", 0))
    optimizer_state = resume_payload.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    print(f"resuming {cfg.teacher_checkpoint} from step {start_step}")

if start_step >= cfg.teacher_steps:
    print(f"{cfg.teacher_checkpoint} is already at step {start_step}; target is {cfg.teacher_steps}")
    raise SystemExit(0)

progress = trange(start_step + 1, cfg.teacher_steps + 1, desc="Training CV teacher", unit="step", dynamic_ncols=True)
for step in progress:
    x = first_batch if step == start_step + 1 else next(batches)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, cfg.dtype):
        out = model(x)
        rec = mse_loss(x, out["recon"])
        coef = coefficient_loss(out["values"])
        loss = rec + cfg.coef_loss_weight * coef
    loss.backward()
    optimizer.step()
    model.normalize_decoder_()
    if step % cfg.log_every == 0:
        with torch.no_grad():
            score = r2_score(x, out["recon"]).item()
            cos = cosine_recon(x, out["recon"]).item()
            l0 = selected_l0(out["values"]).item()
        progress.set_postfix(loss=f"{loss.item():.6f}", rec=f"{rec.item():.6f}", r2=f"{score:.4f}", cos=f"{cos:.4f}", l0=f"{l0:.2f}")
        progress.write(f"step={step} loss={loss.item():.6f} rec={rec.item():.6f} r2={score:.4f} cos={cos:.4f} l0={l0:.2f}")
    if step % cfg.save_every == 0:
        save_checkpoint(cfg.teacher_checkpoint, model, cfg, step, {"d_model": d_model, "n_latents": n_latents}, optimizer)

save_checkpoint(cfg.teacher_checkpoint, model, cfg, cfg.teacher_steps, {"d_model": d_model, "n_latents": n_latents}, optimizer)
print(f"saved {cfg.teacher_checkpoint}")
