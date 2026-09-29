import torch
import torch.nn.functional as F


def batch_gather_rows(matrix: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    flat = indices.reshape(-1)
    gathered = matrix.index_select(0, flat)
    return gathered.reshape(*indices.shape, matrix.shape[-1])


def decode_sparse(decoder: torch.Tensor, decoder_bias: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    selected_decoder = batch_gather_rows(decoder, indices)
    recon = torch.einsum("bk,bkd->bd", values, selected_decoder)
    return recon + decoder_bias


def layer_norm_input(x: torch.Tensor, enabled: bool = True) -> torch.Tensor:
    if enabled:
        return F.layer_norm(x, (x.shape[-1],))
    return x


def normalize_decoder_with_encoder_(decoder: torch.Tensor, encoder_vectors: torch.Tensor | None = None, encoder_bias: torch.Tensor | None = None) -> None:
    with torch.no_grad():
        norms = decoder.norm(dim=-1).clamp_min(1e-6)
        decoder.div_(norms[:, None])
        if encoder_vectors is not None:
            encoder_vectors.mul_(norms[:, None])
        if encoder_bias is not None:
            encoder_bias.mul_(norms)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")
