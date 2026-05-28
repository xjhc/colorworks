from __future__ import annotations

import hashlib
import io
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from PIL import Image, UnidentifiedImageError

from colorworks.recipe import Recipe, load_recipe, save_recipe
from colorworks.domain import ParameterDef


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
        self.presets_dir = root / "presets"
        self.artifacts_dir = root / "artifacts"
        self.runs_dir = root / "runs"
        self.calibration_assets_dir = root / "calibration_assets"
        self.index_path = self.artifacts_dir / "index.json"

        for directory in (self.assets_dir, self.outputs_dir, self.recipes_dir,
                          self.presets_dir, self.artifacts_dir, self.runs_dir,
                          self.calibration_assets_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self._artifacts_index = {}
        self._load_artifacts_index()

    def _load_artifacts_index(self) -> None:
        self._artifacts_index = {}
        if self.index_path.exists():
            try:
                self._artifacts_index = json.loads(self.index_path.read_text(encoding="utf-8"))
            except Exception:
                pass

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
            # Check in artifacts
            artifact_path = self.artifacts_dir / f"{checksum}.png"
            if artifact_path.exists():
                return artifact_path
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

    # Preset CRUD
    def save_preset(self, data: dict[str, Any]) -> str:
        preset_id = data.get("id")
        if preset_id:
            # Reject non-slug IDs
            if not re.match(r"^[a-z0-9_-]+$", preset_id):
                raise ValueError("Preset ID must only contain lowercase alphanumeric characters, dashes, and underscores")
        else:
            name = data.get("name", "untitled_preset")
            # Generate a slug from name, replacing any non-alphanumeric/underscore/dash with dash
            slugified_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
            if not slugified_name:
                slugified_name = "preset"
            preset_id = f"{slugified_name}-{uuid.uuid4().hex[:8]}"

        from colorworks.presets import BUILTIN_PRESETS
        builtins = {p["id"] for p in BUILTIN_PRESETS}
        if preset_id in builtins:
            raise ValueError("Cannot overwrite a built-in preset")

        path = (self.presets_dir / f"{preset_id}.json").resolve()
        presets_dir_resolved = self.presets_dir.resolve()
        if not path.is_relative_to(presets_dir_resolved) or path == presets_dir_resolved:
            raise ValueError("Invalid preset ID path traversal attempt")

        data["id"] = preset_id
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return preset_id

    def list_presets(self) -> list[dict[str, Any]]:
        from colorworks.presets import BUILTIN_PRESETS
        presets = []
        for p in BUILTIN_PRESETS:
            presets.append(dict(p))

        for path in sorted(self.presets_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
            try:
                preset_data = json.loads(path.read_text(encoding="utf-8"))
                preset_data["id"] = path.stem
                preset_data["is_builtin"] = False
                presets.append(preset_data)
            except Exception:
                continue
        return presets

    def get_preset(self, preset_id: str) -> dict[str, Any]:
        if not re.match(r"^[a-z0-9_-]+$", preset_id):
            raise KeyError(f"unknown preset_id: {preset_id}")
        from colorworks.presets import BUILTIN_PRESETS
        for p in BUILTIN_PRESETS:
            if p["id"] == preset_id:
                return dict(p)

        path = (self.presets_dir / f"{preset_id}.json").resolve()
        presets_dir_resolved = self.presets_dir.resolve()
        if not path.is_relative_to(presets_dir_resolved) or path == presets_dir_resolved or not path.exists():
            raise KeyError(f"unknown preset_id: {preset_id}")
        preset_data = json.loads(path.read_text(encoding="utf-8"))
        preset_data["id"] = preset_id
        preset_data["is_builtin"] = False
        return preset_data

    def delete_preset(self, preset_id: str) -> None:
        if not re.match(r"^[a-z0-9_-]+$", preset_id):
            raise KeyError(f"unknown preset_id: {preset_id}")
        from colorworks.presets import BUILTIN_PRESETS
        builtins = {p["id"] for p in BUILTIN_PRESETS}
        if preset_id in builtins:
            raise ValueError("Cannot delete a built-in preset")

        path = (self.presets_dir / f"{preset_id}.json").resolve()
        presets_dir_resolved = self.presets_dir.resolve()
        if not path.is_relative_to(presets_dir_resolved) or path == presets_dir_resolved or not path.exists():
            raise KeyError(f"unknown preset_id: {preset_id}")
        path.unlink()

    def _asset_record_path(self, asset_id: str) -> Path:
        return self.assets_dir / f"{asset_id}.json"

    # Calibration assets storage operations
    def has_calibration_asset(self, checksum: str) -> bool:
        return (
            (self.calibration_assets_dir / f"{checksum}.json").exists()
            and (self.calibration_assets_dir / f"{checksum}.npy").exists()
        )

    def save_calibration_asset(self, checksum: str, data: np.ndarray, metadata: dict[str, Any]) -> None:
        # Enforce content-address integrity
        data_canonical = np.asarray(data, dtype="<f4")
        computed_checksum = hashlib.sha256(data_canonical.tobytes()).hexdigest()
        if computed_checksum != checksum:
            raise ValueError(f"Calibration data checksum mismatch: computed {computed_checksum} but got {checksum}")
        if metadata.get("checksum") != checksum:
            raise ValueError(f"Calibration metadata checksum mismatch: metadata checksum {metadata.get('checksum')} != expected {checksum}")

        meta_path = self.calibration_assets_dir / f"{checksum}.json"
        npy_path = self.calibration_assets_dir / f"{checksum}.npy"
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        np.save(npy_path, data_canonical)

    def get_calibration_path(self, checksum: str) -> Path:
        path = self.calibration_assets_dir / f"{checksum}.npy"
        if not path.exists():
            raise KeyError(f"unknown calibration asset data: {checksum}")
        return path

    def get_calibration_metadata(self, checksum: str) -> dict[str, Any]:
        path = self.calibration_assets_dir / f"{checksum}.json"
        if not path.exists():
            raise KeyError(f"unknown calibration asset metadata: {checksum}")
        return json.loads(path.read_text(encoding="utf-8"))

    def get_calibration_data(self, checksum: str) -> np.ndarray:
        path = self.get_calibration_path(checksum)
        return np.load(path)

    # Per-artifact cache helpers
    def get_artifact_cache_key(
        self,
        algo_id: str,
        algo_version: str,
        artifact_name: str,
        asset_checksum: str,
        params: dict[str, Any],
        parameters_def: list[ParameterDef],
        calibration_assets_checksum: str | None = None,
    ) -> str:
        dep_params = {}
        for p in parameters_def:
            if artifact_name in p.invalidates:
                dep_params[p.key] = params.get(p.key, p.default)
        dep_str = ",".join(f"{k}={dep_params[k]}" for k in sorted(dep_params.keys()))
        s = f"{algo_id}:{algo_version}:{artifact_name}:{asset_checksum}:{dep_str}"
        if calibration_assets_checksum:
            s += f":calibration={calibration_assets_checksum}"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def get_tone_map_cache_key(self, asset_checksum: str, contrast: float, midpoint: float) -> str:
        from colorworks.algorithms import registry
        try:
            algo = registry.get("tonal_analyzer")
            return self.get_artifact_cache_key(
                algo_id=algo.definition.id,
                algo_version=algo.definition.version,
                artifact_name="tone_map",
                asset_checksum=asset_checksum,
                params={"contrast": contrast, "midpoint": midpoint},
                parameters_def=algo.definition.parameters,
            )
        except Exception:
            # Fallback
            s = f"tone_map:{asset_checksum}:contrast={contrast:.4f}:midpoint={midpoint:.4f}"
            return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def get_edge_mask_cache_key(self, asset_checksum: str, preserve_edges: bool, edge_threshold: float) -> str:
        from colorworks.algorithms import registry
        try:
            algo = registry.get("tonal_analyzer")
            return self.get_artifact_cache_key(
                algo_id=algo.definition.id,
                algo_version=algo.definition.version,
                artifact_name="edge_mask",
                asset_checksum=asset_checksum,
                params={"preserve_edges": preserve_edges, "edge_threshold": edge_threshold},
                parameters_def=algo.definition.parameters,
            )
        except Exception:
            # Fallback
            s = f"edge_mask:{asset_checksum}:preserve={preserve_edges}:threshold={edge_threshold:.4f}"
            return hashlib.sha256(s.encode("utf-8")).hexdigest()
    def get_final_raster_cache_key(self, tone_map_checksum: str, edge_mask_checksum: str | None, composition_json: str) -> str:
        s = f"final_raster:tone={tone_map_checksum}:edge={edge_mask_checksum or ''}:{composition_json}"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def get_cached_artifact(self, cache_key: str) -> tuple[np.ndarray | Path, dict[str, Any]] | None:
        meta_path = self.artifacts_dir / f"{cache_key}.json"
        npy_path = self.artifacts_dir / f"{cache_key}.npy"
        png_path = self.artifacts_dir / f"{cache_key}.png"

        if not meta_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            art_id = meta.get("id")
            if art_id and art_id not in self._artifacts_index:
                self._artifacts_index[art_id] = {
                    "cache_key": cache_key,
                    "name": meta.get("name"),
                    "type": meta.get("type"),
                    "checksum": meta.get("checksum"),
                }
                self.index_path.write_text(
                    json.dumps(self._artifacts_index, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

            type_name = meta.get("type")
            if type_name in ("scalar_field", "binary_mask", "vector_field_2d", "structure_tensor_field") and npy_path.exists():
                import numpy as np
                arr = np.load(npy_path)
                return arr, meta
            elif png_path.exists():
                return png_path, meta
        except Exception:
            return None
        return None

    def save_cached_artifact(self, cache_key: str, data: np.ndarray | bytes | Image.Image, metadata: dict[str, Any]) -> None:
        import numpy as np
        meta_path = self.artifacts_dir / f"{cache_key}.json"
        png_path = self.artifacts_dir / f"{cache_key}.png"
        npy_path = self.artifacts_dir / f"{cache_key}.npy"

        # Save metadata file
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if isinstance(data, np.ndarray):
            np.save(npy_path, data)
            if metadata.get("type") == "vector_field_2d":
                from colorworks.domain import VectorField2D, render_vector_hsv, render_vector_glyphs, RasterGrid
                grid = RasterGrid(width=data.shape[1], height=data.shape[0])
                field = VectorField2D(grid, data, is_bidirectional=metadata.get("is_bidirectional", False))
                hsv_img = render_vector_hsv(field)
                hsv_img.save(png_path, format="PNG")
                hsv_img.save(self.artifacts_dir / f"{cache_key}_hsv.png", format="PNG")
                glyphs_img = render_vector_glyphs(field)
                glyphs_img.save(self.artifacts_dir / f"{cache_key}_glyphs.png", format="PNG")
            elif metadata.get("type") == "structure_tensor_field":
                from colorworks.domain import StructureTensorField, render_tensor_anisotropy, RasterGrid
                grid = RasterGrid(width=data.shape[1], height=data.shape[0])
                field = StructureTensorField(grid, data)
                aniso_img = render_tensor_anisotropy(field)
                aniso_img.save(png_path, format="PNG")
            else:
                if data.dtype == bool:
                    arr_uint8 = np.where(data, 255, 0).astype(np.uint8)
                    img = Image.fromarray(arr_uint8, mode="L")
                else:
                    arr_uint8 = np.clip(data * 255.0, 0.0, 255.0).astype(np.uint8)
                    img = Image.fromarray(arr_uint8, mode="L")
                img.save(png_path, format="PNG")
        elif isinstance(data, Image.Image):
            data.save(png_path, format="PNG")
        elif isinstance(data, bytes):
            png_path.write_bytes(data)

        # Update central index
        art_id = metadata.get("id")
        if art_id:
            self._artifacts_index[art_id] = {
                "cache_key": cache_key,
                "name": metadata.get("name"),
                "type": metadata.get("type"),
                "checksum": metadata.get("checksum"),
            }
            try:
                self.index_path.write_text(
                    json.dumps(self._artifacts_index, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            except Exception:
                pass

    def get_artifact_path_by_id(self, artifact_id: str) -> Path | None:
        record = self._artifacts_index.get(artifact_id)
        if record:
            cache_key = record.get("cache_key")
            if cache_key:
                png_path = self.artifacts_dir / f"{cache_key}.png"
                if png_path.exists():
                    return png_path
        # Fallback: ArtifactStore.save_preview() writes {artifact_id}.png directly
        # (used by iterative runs that bypass the cache-key layer)
        direct_path = self.artifacts_dir / f"{artifact_id}.png"
        if direct_path.exists():
            return direct_path
        return None


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
