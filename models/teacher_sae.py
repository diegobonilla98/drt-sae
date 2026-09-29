import torch
from torch import nn
import torch.nn.functional as F

from models.common import decode_sparse, layer_norm_input, normalize_decoder_with_encoder_


class TopKSAETeacher(nn.Module):
    def __init__(self, d_model: int, n_latents: int, k: int, input_layer_norm: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.k = k
        self.input_layer_norm = input_layer_norm
        self.encoder = nn.Linear(d_model, n_latents)
        self.decoder = nn.Parameter(torch.empty(n_latents, d_model))
        self.decoder_bias = nn.Parameter(torch.zeros(d_model))
        nn.init.kaiming_uniform_(self.encoder.weight, a=5 ** 0.5)
        nn.init.normal_(self.decoder, std=d_model ** -0.5)
        self.normalize_decoder_()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm_input(x, self.input_layer_norm)

    def encode_dense(self, x: torch.Tensor) -> torch.Tensor:
        h = self.preprocess(x)
        return F.relu(self.encoder(h))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        acts = self.encode_dense(x)
        values, indices = torch.topk(acts, k=self.k, dim=-1)
        recon = decode_sparse(self.decoder, self.decoder_bias, indices, values)
        return {"recon": recon, "dense_acts": acts, "indices": indices, "values": values}

    def normalize_decoder_(self) -> None:
        normalize_decoder_with_encoder_(self.decoder, self.encoder.weight, self.encoder.bias)
