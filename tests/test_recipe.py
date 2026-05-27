from __future__ import annotations

import hashlib

from PIL import Image

from colorworks.recipe import Recipe, load_recipe, save_recipe
from colorworks.renderers.bayer import BayerParams, ordered_dither


def test_recipe_round_trip_reproduces_output_checksum(tmp_path) -> None:
    source = Image.linear_gradient("L").resize((96, 96))
    params = BayerParams(matrix_size=8, threshold=0.12, contrast=1.4)
    before = hashlib.sha256(ordered_dither(source, params).tobytes()).hexdigest()

    recipe = Recipe.create(
        name="Round trip",
        asset_id="asset-test",
        asset_checksum="abc123",
        params=params,
    )
    path = tmp_path / "recipe.json"
    save_recipe(path, recipe)

    loaded = load_recipe(path)
    after = hashlib.sha256(ordered_dither(source, loaded.params).tobytes()).hexdigest()

    assert loaded.name == "Round trip"
    assert loaded.asset_id == "asset-test"
    assert before == after
