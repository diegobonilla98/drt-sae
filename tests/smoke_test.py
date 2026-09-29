from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.teacher_sae import TopKSAETeacher
from models.routed_sae import RoutedSparseAutoencoder
from utils.metrics import mse_loss
from utils.losses import sampled_route_distillation_loss
from utils.load_balance import load_loss, update_router_bias_


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 32
    n_latents = 128
    teacher_k = 8
    router_k = 4
    x = torch.randn(64, d_model, device=device)
    teacher = TopKSAETeacher(d_model, n_latents, teacher_k).to(device)
    routed = RoutedSparseAutoencoder(d_model, n_latents, router_k, router_dim=16, score_chunk_size=64).to(device)
    routed.initialize_from_teacher(teacher)
    teacher_out = teacher(x)
    routed_out = routed(x)
    rec = mse_loss(x, routed_out["recon"])
    route = sampled_route_distillation_loss(routed, x, teacher_out["indices"], teacher_out["values"], routed_out["indices"], 16)
    balance = load_loss(routed_out["indices"], routed.n_latents)
    loss = rec + 0.1 * route + 0.01 * balance
    loss.backward()
    update_router_bias_(routed, routed_out["indices"], 1e-3)
    print({"loss": float(loss.item()), "recon_shape": list(routed_out["recon"].shape), "indices_shape": list(routed_out["indices"].shape)})


if __name__ == "__main__":
    main()
