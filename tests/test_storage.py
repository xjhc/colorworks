from __future__ import annotations

import io

import pytest
from PIL import Image

from colorworks.storage import LocalStore


def test_save_asset_validates_raster_before_writing(tmp_path) -> None:
    store = LocalStore(tmp_path)

    with pytest.raises(ValueError, match="supported raster image"):
        store.save_asset(filename="not-an-image.png", content=b"not an image")

    assert list(store.assets_dir.iterdir()) == []


def test_save_asset_uses_detected_image_format_for_path(tmp_path) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), "white").save(buffer, format="PNG")
    store = LocalStore(tmp_path)

    record = store.save_asset(filename="misleading.jpg", content=buffer.getvalue())

    assert record.path.suffix == ".png"
    assert record.width == 4
    assert record.height == 3


def test_save_asset_transcodes_heic_to_png(tmp_path) -> None:
    pytest.importorskip("pillow_heif")
    buffer = io.BytesIO()
    Image.new("RGB", (5, 7), "white").save(buffer, format="HEIF")
    store = LocalStore(tmp_path)

    record = store.save_asset(filename="iphone.heic", content=buffer.getvalue())

    # Browsers can't render HEIC, so it's stored (and previewed) as PNG...
    assert record.path.suffix == ".png"
    assert record.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    # ...while the original filename is kept for display and dimensions survive.
    assert record.original_filename == "iphone.heic"
    assert (record.width, record.height) == (5, 7)
