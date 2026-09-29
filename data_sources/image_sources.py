from pathlib import Path
from itertools import islice
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_image_files(root: str, limit: int | None = None):
    paths = [p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    paths = sorted(paths)
    iterator = paths if limit is None else islice(paths, limit)
    for path in iterator:
        yield str(path)


def load_rgb_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def batched_items(iterator, batch_size: int):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_hf_images(dataset_name: str, split: str, image_column: str, limit: int | None = None):
    from datasets import load_dataset
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    iterator = dataset if limit is None else islice(dataset, limit)
    for row in iterator:
        image = row.get(image_column)
        if image is not None:
            yield image.convert("RGB")
