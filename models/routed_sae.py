import math
import torch
from torch import nn
import torch.nn.functional as F

from models.common import batch_gather_rows, decode_sparse, layer_norm_input, normalize_decoder_with_encoder_
from models.teacher_sae import TopKSAETeacher


class RoutedSparseAutoencoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_latents: int,
        k: int,
        router_dim: int,
        score_chunk_size: int = 8192,
        input_layer_norm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.k = k
        self.router_dim = router_dim
        self.score_chunk_size = score_chunk_size
        self.input_layer_norm = input_layer_norm
        self.query = nn.Linear(d_model, router_dim, bias=False)
        self.router_keys = nn.Parameter(torch.empty(n_latents, router_dim))
        self.router_bias = nn.Parameter(torch.zeros(n_latents), requires_grad=False)
        self.encoder_vectors = nn.Parameter(torch.empty(n_latents, d_model))
        self.encoder_bias = nn.Parameter(torch.zeros(n_latents))
        self.decoder = nn.Parameter(torch.empty(n_latents, d_model))
        self.decoder_bias = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.query.weight, std=d_model ** -0.5)
        nn.init.normal_(self.router_keys, std=router_dim ** -0.5)
        nn.init.normal_(self.encoder_vectors, std=d_model ** -0.5)
        nn.init.normal_(self.decoder, std=d_model ** -0.5)
        self.normalize_decoder_()
        self.normalize_router_keys_()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm_input(x, self.input_layer_norm)

    def router_query(self, x: torch.Tensor) -> torch.Tensor:
        h = self.preprocess(x)
        return self.query(h)

    def score_candidates_from_query(self, q: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        keys = batch_gather_rows(self.router_keys, candidates)
        bias = self.router_bias.index_select(0, candidates.reshape(-1)).reshape(candidates.shape)
        return torch.einsum("br,bcr->bc", q, keys) / math.sqrt(self.router_dim) + bias

    def score_candidates(self, x: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        q = self.router_query(x)
        return self.score_candidates_from_query(q, candidates)

    def route(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        q = self.router_query(x)
        return self.route_from_query(q)

    def route_from_query(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.score_chunk_size is None or self.score_chunk_size >= self.n_latents:
            scores = q @ self.router_keys.t() / math.sqrt(self.router_dim) + self.router_bias
            values, indices = torch.topk(scores, k=self.k, dim=-1)
            return {"indices": indices, "router_values": values, "query": q, "scores": scores}
        best_values = None
        best_indices = None
        for start in range(0, self.n_latents, self.score_chunk_size):
            end = min(start + self.score_chunk_size, self.n_latents)
            keys = self.router_keys[start:end]
            bias = self.router_bias[start:end]
            scores = q @ keys.t() / math.sqrt(self.router_dim) + bias
            local_k = min(self.k, end - start)
            values, indices = torch.topk(scores, k=local_k, dim=-1)
            indices = indices + start
            if best_values is None:
                best_values = values
                best_indices = indices
            else:
                merged_values = torch.cat([best_values, values], dim=-1)
                merged_indices = torch.cat([best_indices, indices], dim=-1)
                best_values, positions = torch.topk(merged_values, k=self.k, dim=-1)
                best_indices = torch.gather(merged_indices, dim=-1, index=positions)
        return {"indices": best_indices, "router_values": best_values, "query": q}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.preprocess(x)
        q = self.query(h)
        routed = self.route_from_query(q)
        indices = routed["indices"]
        encoder_vectors = batch_gather_rows(self.encoder_vectors, indices)
        encoder_bias = self.encoder_bias.index_select(0, indices.reshape(-1)).reshape(indices.shape)
        preacts = torch.einsum("bd,bkd->bk", h, encoder_vectors) + encoder_bias
        values = F.relu(preacts)
        recon = decode_sparse(self.decoder, self.decoder_bias, indices, values)
        result = {"recon": recon, "indices": indices, "values": values, "router_values": routed["router_values"], "query": q}
        if "scores" in routed:
            result["scores"] = routed["scores"]
        return result

    def initialize_from_teacher(self, teacher: TopKSAETeacher) -> None:
        with torch.no_grad():
            self.encoder_vectors.copy_(teacher.encoder.weight)
            self.encoder_bias.copy_(teacher.encoder.bias)
            self.decoder.copy_(teacher.decoder)
            self.decoder_bias.copy_(teacher.decoder_bias)
        self.initialize_router_keys_from_encoder_()
        self.normalize_decoder_()
        self.normalize_router_keys_()

    def initialize_router_keys_from_encoder_(self) -> None:
        with torch.no_grad():
            projection = torch.randn(self.d_model, self.router_dim, device=self.encoder_vectors.device, dtype=self.encoder_vectors.dtype)
            projection = projection / math.sqrt(self.d_model)
            self.router_keys.copy_(self.encoder_vectors @ projection)

    def normalize_decoder_(self) -> None:
        normalize_decoder_with_encoder_(self.decoder, self.encoder_vectors, self.encoder_bias)

    def normalize_router_keys_(self) -> None:
        with torch.no_grad():
            self.router_keys.div_(self.router_keys.norm(dim=-1, keepdim=True).clamp_min(1e-6))
