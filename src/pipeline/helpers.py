import base64
import json
import re
from pathlib import Path

# encode image to base64
def encode_image_b64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# read json file safely
def load_json(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# write json file with standardized formatting
def save_json(file_path: Path, data: dict | list) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# retrieve all image files from dir in natural alphanumerical order
def get_sorted_images(dir_path: Path) -> list[Path]:
    extensions = ("*.jpg", "*.jpeg", "*.png")
    images = []
    for ext in extensions:
        images.extend(dir_path.glob(ext))

    def natural_sort_key(path: Path):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", path.name)
        ]

    return sorted(images, key=natural_sort_key)