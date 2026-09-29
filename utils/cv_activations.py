from contextlib import contextmanager
import math
import torch

from utils.llm_activations import ActivationCapture, tensor_from_output, replace_tensor_in_output


def get_clip_vision_model(model):
    if hasattr(model, "vision_model"):
        return model.vision_model
    return model


def get_clip_vision_layers(model):
    vision_model = get_clip_vision_model(model)
    if hasattr(vision_model, "encoder") and hasattr(vision_model.encoder, "layers"):
        return vision_model.encoder.layers
    if hasattr(vision_model, "vision_model") and hasattr(vision_model.vision_model.encoder, "layers"):
        return vision_model.vision_model.encoder.layers
    raise AttributeError("Could not find CLIP vision transformer layers")


@contextmanager
def patch_vision_layer_output(module, patch_fn):
    def hook(module, inputs, output):
        hidden = tensor_from_output(output)
        patched = patch_fn(hidden)
        return replace_tensor_in_output(output, patched)
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def extract_clip_patch_activations(model, processor, images, layer_id: int, device: torch.device):
    layers = get_clip_vision_layers(model)
    module = layers[layer_id]
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    vision_model = get_clip_vision_model(model)
    model.eval()
    with torch.no_grad():
        with ActivationCapture(module) as capture:
            vision_model(pixel_values=pixel_values)
    hidden = capture.value
    patch_hidden = hidden[:, 1:, :].detach().cpu()
    rows = patch_hidden.reshape(-1, patch_hidden.shape[-1])
    patch_count = patch_hidden.shape[1]
    grid = int(math.sqrt(patch_count))
    metadata = []
    for batch_index in range(patch_hidden.shape[0]):
        for patch_index in range(patch_count):
            metadata.append({"sample_index": batch_index, "patch_index": patch_index, "patch_row": patch_index // grid, "patch_col": patch_index % grid, "patch_grid": grid})
    return rows, metadata


def patch_box_from_index(image_width: int, image_height: int, patch_index: int, grid: int):
    row = patch_index // grid
    col = patch_index % grid
    left = int(col * image_width / grid)
    right = int((col + 1) * image_width / grid)
    top = int(row * image_height / grid)
    bottom = int((row + 1) * image_height / grid)
    return left, top, right, bottom
