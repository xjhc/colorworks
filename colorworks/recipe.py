from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
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
    params: BayerParams
    created_at: str
    renderer_id: str = RENDERER_ID
    renderer_version: str = RENDERER_VERSION
    schema_version: int = RECIPE_SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "renderer_id": self.renderer_id,
            "renderer_version": self.renderer_version,
            "asset": {
                "id": self.asset_id,
                "checksum": self.asset_checksum,
            },
            "params": self.params.to_json(),
            "created_at": self.created_at,
        }

    @classmethod
    def create(
        cls,
        *,
        name: str,
        asset_id: str,
        asset_checksum: str,
        params: BayerParams,
    ) -> "Recipe":
        return cls(
            name=name.strip() or "Untitled recipe",
            asset_id=asset_id,
            asset_checksum=asset_checksum,
            params=params.normalized(),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Recipe":
        if data.get("schema_version") != RECIPE_SCHEMA_VERSION:
            raise ValueError("unsupported recipe schema_version")
        if data.get("renderer_id") != RENDERER_ID:
            raise ValueError("unsupported renderer_id")

        asset = data.get("asset")
        if not isinstance(asset, dict):
            raise ValueError("recipe asset must be an object")

        params = data.get("params")
        if not isinstance(params, dict):
            raise ValueError("recipe params must be an object")

        return cls(
            name=str(data.get("name") or "Untitled recipe"),
            renderer_id=str(data["renderer_id"]),
            renderer_version=str(data.get("renderer_version") or RENDERER_VERSION),
            asset_id=str(asset["id"]),
            asset_checksum=str(asset["checksum"]),
            params=BayerParams.from_json(params),
            created_at=str(data.get("created_at") or ""),
            schema_version=RECIPE_SCHEMA_VERSION,
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
