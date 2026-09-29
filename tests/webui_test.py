from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webui.server import ExpertIntervention, ImageRequest, TextRequest, expert_boost_patch


class FixedSae(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_latents = 3
        self.decoder = nn.Parameter(torch.tensor([[2.0, -1.0], [0.5, 0.5], [-1.0, 1.0]]))

    def forward(self, activations: torch.Tensor) -> dict[str, torch.Tensor]:
        rows = activations.shape[0]
        return {
            "indices": torch.zeros((rows, 1), dtype=torch.long, device=activations.device),
            "values": torch.ones((rows, 1), dtype=activations.dtype, device=activations.device),
        }


def main() -> None:
    sae = FixedSae()
    hidden = torch.zeros((1, 3, 2))
    boosted = expert_boost_patch(sae, expert_id=0, multiplier=2.0, dtype_name="float32")(hidden)
    assert torch.allclose(boosted, torch.tensor([[[2.0, -1.0], [2.0, -1.0], [2.0, -1.0]]]))
    patch_only = expert_boost_patch(sae, expert_id=0, multiplier=2.0, dtype_name="float32", start_index=1)(hidden)
    assert torch.equal(patch_only[:, :1], hidden[:, :1])
    assert torch.allclose(patch_only[:, 1:], boosted[:, 1:])
    text_request = TextRequest(prompt="test", max_new_tokens=6, intervention=ExpertIntervention(expert_id=0, multiplier=1.5))
    image_request = ImageRequest(image_data="x" * 32, intervention=ExpertIntervention(expert_id=0, multiplier=1.5))
    assert text_request.max_new_tokens == 6
    assert image_request.intervention.expert_id == 0
    print("webui intervention checks passed")


if __name__ == "__main__":
    main()
