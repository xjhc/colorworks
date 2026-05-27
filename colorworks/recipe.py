from __future__ import annotations

import json
from dataclasses import dataclass, is_dataclass, asdict
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from colorworks.renderers.bayer import BayerParams


RECIPE_SCHEMA_VERSION = 1
RENDERER_ID = "ordered_bayer"
RENDERER_VERSION = "0.1.0"


@dataclass(frozen=True)
class Recipe:
    name: str
    asset_id: str
    asset_checksum: str
    params: BayerParams | dict[str, Any]
    created_at: str
    composition: Any | None = None
    renderer_id: str = RENDERER_ID
    renderer_version: str = RENDERER_VERSION
    schema_version: int = RECIPE_SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "name": self.name,
            "renderer_id": self.renderer_id,
            "renderer_version": self.renderer_version,
            "asset": {
                "id": self.asset_id,
                "checksum": self.asset_checksum,
            },
            "params": self.params.to_json() if hasattr(self.params, "to_json") else self.params,
            "created_at": self.created_at,
        }
        if self.composition is not None:
            data["composition"] = self._dataclass_to_dict(self.composition)
        return data

    def _dataclass_to_dict(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return {k: self._dataclass_to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [self._dataclass_to_dict(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: self._dataclass_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, Enum):
            return obj.value
        return obj

    @classmethod
    def create(
        cls,
        *,
        name: str,
        asset_id: str,
        asset_checksum: str,
        params: BayerParams | dict[str, Any],
        composition: Any | None = None,
        renderer_id: str = RENDERER_ID,
    ) -> "Recipe":
        return cls(
            name=name.strip() or "Untitled recipe",
            asset_id=asset_id,
            asset_checksum=asset_checksum,
            params=params.normalized() if hasattr(params, "normalized") else params,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            composition=composition,
            renderer_id=renderer_id,
        )

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Recipe":
        # Keep schema check flexible, but error if version differs
        if data.get("schema_version") != RECIPE_SCHEMA_VERSION:
            raise ValueError("unsupported recipe schema_version")

        renderer_id = str(data.get("renderer_id", RENDERER_ID))

        asset = data.get("asset")
        if not isinstance(asset, dict):
            raise ValueError("recipe asset must be an object")

        params = data.get("params")
        if not isinstance(params, dict):
            raise ValueError("recipe params must be an object")

        if renderer_id == RENDERER_ID:
            parsed_params = BayerParams.from_json(params)
        else:
            parsed_params = params

        return cls(
            name=str(data.get("name") or "Untitled recipe"),
            renderer_id=renderer_id,
            renderer_version=str(data.get("renderer_version") or RENDERER_VERSION),
            asset_id=str(asset["id"]),
            asset_checksum=str(asset["checksum"]),
            params=parsed_params,
            created_at=str(data.get("created_at") or ""),
            schema_version=RECIPE_SCHEMA_VERSION,
            composition=data.get("composition"),
        )


def load_recipe(path: Path) -> Recipe:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("recipe JSON must be an object")
    return Recipe.from_json(data)


def save_recipe(path: Path, recipe: Recipe) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(recipe.to_json(), handle, indent=2, sort_keys=True)
        handle.write("\n")
