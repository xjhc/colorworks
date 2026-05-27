from __future__ import annotations

import hashlib
import io
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from colorworks.recipe import Recipe, load_recipe, save_recipe


SLUG_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class AssetRecord:
    id: str
    checksum: str
    original_filename: str
    path: Path
    width: int
    height: int
    mode: str

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "checksum": self.checksum,
            "original_filename": self.original_filename,
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "AssetRecord":
        return cls(
            id=str(data["id"]),
            checksum=str(data["checksum"]),
            original_filename=str(data["original_filename"]),
            path=Path(str(data["path"])),
            width=int(data["width"]),
            height=int(data["height"]),
            mode=str(data["mode"]),
        )


class LocalStore:
    def __init__(self, root: Path):
        self.root = root
        self.assets_dir = root / "assets"
        self.outputs_dir = root / "outputs"
        self.recipes_dir = root / "recipes"
        for directory in (self.assets_dir, self.outputs_dir, self.recipes_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def save_asset(self, *, filename: str, content: bytes) -> AssetRecord:
        checksum = hashlib.sha256(content).hexdigest()
        width, height, mode, extension = _inspect_raster(content, filename)
        asset_id = checksum[:16]
        image_path = self.assets_dir / f"{asset_id}{extension}"
        image_path.write_bytes(content)

        record = AssetRecord(
            id=asset_id,
            checksum=checksum,
            original_filename=filename or f"{asset_id}{extension}",
            path=image_path,
            width=width,
            height=height,
            mode=mode,
        )
        self._asset_record_path(asset_id).write_text(
            json.dumps(record.to_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return record

    def get_asset(self, asset_id: str) -> AssetRecord:
        path = self._asset_record_path(asset_id)
        if not path.exists():
            raise KeyError(f"unknown asset_id: {asset_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return AssetRecord.from_json(data)

    def save_output(self, png_bytes: bytes) -> tuple[str, Path]:
        checksum = hashlib.sha256(png_bytes).hexdigest()
        path = self.outputs_dir / f"{checksum}.png"
        if not path.exists():
            path.write_bytes(png_bytes)
        return checksum, path

    def output_path(self, checksum: str) -> Path:
        path = self.outputs_dir / f"{checksum}.png"
        if not path.exists():
            raise KeyError(f"unknown output checksum: {checksum}")
        return path

    def save_recipe(self, recipe: Recipe) -> tuple[str, Path]:
        recipe_id = f"{_slug(recipe.name)}-{uuid.uuid4().hex[:8]}"
        path = self.recipes_dir / f"{recipe_id}.json"
        save_recipe(path, recipe)
        return recipe_id, path

    def list_recipes(self) -> list[tuple[str, Recipe]]:
        recipes: list[tuple[str, Recipe]] = []
        for path in sorted(self.recipes_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
            try:
                recipes.append((path.stem, load_recipe(path)))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return recipes

    def get_recipe(self, recipe_id: str) -> Recipe:
        path = self.recipes_dir / f"{recipe_id}.json"
        if not path.exists():
            raise KeyError(f"unknown recipe_id: {recipe_id}")
        return load_recipe(path)

    def _asset_record_path(self, asset_id: str) -> Path:
        return self.assets_dir / f"{asset_id}.json"


def _safe_extension(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}:
        return suffix
    return ".png"


def _inspect_raster(content: bytes, filename: str) -> tuple[int, int, str, str]:
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
            mode = image.mode
            extension = _extension_for_format(image.format, filename)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("upload must be a supported raster image") from exc
    return width, height, mode, extension


def _extension_for_format(image_format: str | None, filename: str) -> str:
    match image_format:
        case "PNG":
            return ".png"
        case "JPEG":
            return ".jpg"
        case "WEBP":
            return ".webp"
        case "BMP":
            return ".bmp"
        case "GIF":
            return ".gif"
        case "TIFF":
            return ".tiff"
        case _:
            return _safe_extension(filename)


def _slug(value: str) -> str:
    slug = SLUG_RE.sub("-", value.strip().lower()).strip("-._")
    return slug[:48] or "recipe"
