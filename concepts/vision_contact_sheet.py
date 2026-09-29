from pathlib import Path
from PIL import Image, ImageDraw

from data_sources.image_sources import load_rgb_image
from utils.cv_activations import patch_box_from_index


def make_contact_sheet(examples: list[dict], output_path: str, tile_size: int = 160, columns: int = 4, quality: int = 92) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    rows = max(1, (len(examples) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), "white")
    for i, example in enumerate(examples):
        image = load_rgb_image(example["image_path"])
        grid = int(example.get("patch_grid", 14))
        box = patch_box_from_index(image.width, image.height, int(example["patch_index"]), grid)
        crop = image.crop(box)
        crop.thumbnail((tile_size, tile_size))
        tile = Image.new("RGB", (tile_size, tile_size), "white")
        left = (tile_size - crop.width) // 2
        top = (tile_size - crop.height) // 2
        tile.paste(crop, (left, top))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((2, 2, tile_size - 3, tile_size - 3), outline="red", width=3)
        x = (i % columns) * tile_size
        y = (i // columns) * tile_size
        sheet.paste(tile, (x, y))
    sheet.save(output_path, quality=quality, optimize=True)
    return output_path
