from contextlib import contextmanager
import torch


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Could not find transformer layers on this model")


def tensor_from_output(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_tensor_in_output(output, hidden):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


class ActivationCapture:
    def __init__(self, module):
        self.module = module
        self.value = None
        self.handle = None

    def __enter__(self):
        def hook(module, inputs, output):
            self.value = tensor_from_output(output).detach()
        self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.handle.remove()


@contextmanager
def patch_layer_output(module, patch_fn):
    def hook(module, inputs, output):
        hidden = tensor_from_output(output)
        patched = patch_fn(hidden)
        return replace_tensor_in_output(output, patched)
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def extract_llm_activations(model, tokenizer, texts, layer_id: int, max_length: int, device: torch.device, context_radius: int = 24):
    layers = get_transformer_layers(model)
    module = layers[layer_id]
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    model.eval()
    with torch.no_grad():
        with ActivationCapture(module) as capture:
            model(**encoded)
    hidden = capture.value
    attention_mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))
    rows = []
    metadata = []
    input_ids = encoded["input_ids"].detach().cpu()
    mask_cpu = attention_mask.detach().cpu()
    for batch_index in range(input_ids.shape[0]):
        valid_positions = torch.where(mask_cpu[batch_index] > 0)[0].tolist()
        for token_index in valid_positions:
            token_id = int(input_ids[batch_index, token_index])
            token_text = tokenizer.decode([token_id])
            start = max(0, token_index - context_radius)
            end = min(input_ids.shape[1], token_index + context_radius + 1)
            context_ids = input_ids[batch_index, start:end].tolist()
            context = tokenizer.decode(context_ids)
            rows.append(hidden[batch_index, token_index].detach().cpu())
            metadata.append({"sample_index": batch_index, "token_index": token_index, "token_id": token_id, "token_text": token_text, "context": context})
    if not rows:
        return torch.empty(0, hidden.shape[-1]), []
    return torch.stack(rows, dim=0), metadata


def last_nonpad_position(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.long().sum(dim=-1).clamp_min(1) - 1
