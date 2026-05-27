from __future__ import annotations

import argparse
import io
import json
import mimetypes
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from PIL import Image

from colorworks.recipe import Recipe
from colorworks.renderers.bayer import BayerParams, ordered_dither
from colorworks.storage import LocalStore


STATIC_DIR = Path(__file__).with_name("static")


class ColorworksHandler(BaseHTTPRequestHandler):
    server: "ColorworksServer"

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        try:
            route = urlsplit(self.path).path
            if route == "/" or route == "/index.html":
                self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            elif route.startswith("/static/"):
                rel = route.removeprefix("/static/")
                self._send_file(STATIC_DIR / rel)
            elif route.startswith("/api/assets/") and route.endswith("/image"):
                asset_id = route.split("/")[-2]
                asset = self.server.store.get_asset(asset_id)
                self._send_file(asset.path)
            elif route.startswith("/api/assets/"):
                asset_id = route.rsplit("/", 1)[-1]
                asset = self.server.store.get_asset(asset_id)
                self._send_json({"asset": asset.to_json()})
            elif route == "/api/recipes":
                self._send_json(
                    {
                        "recipes": [
                            {"id": recipe_id, **recipe.to_json()}
                            for recipe_id, recipe in self.server.store.list_recipes()
                        ]
                    }
                )
            elif route.startswith("/api/recipes/"):
                recipe_id = route.rsplit("/", 1)[-1]
                recipe = self.server.store.get_recipe(recipe_id)
                self._send_json({"id": recipe_id, **recipe.to_json()})
            elif route.startswith("/api/outputs/"):
                checksum = route.rsplit("/", 1)[-1].removesuffix(".png")
                self._send_file(self.server.store.output_path(checksum), "image/png")
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "not found")
        except KeyError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        try:
            if self.path == "/api/assets":
                self._handle_asset_upload()
            elif self.path == "/api/render":
                self._handle_render()
            elif self.path == "/api/recipes":
                self._handle_recipe_save()
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "not found")
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except KeyError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _handle_asset_upload(self) -> None:
        content = self._read_body(max_bytes=64 * 1024 * 1024)
        if not content:
            raise ValueError("empty upload")
        filename_header = self.headers.get("X-Filename", "source.png")
        filename = unquote(filename_header)
        record = self.server.store.save_asset(filename=filename, content=content)
        self._send_json({"asset": record.to_json()})

    def _handle_render(self) -> None:
        payload = self._read_json()
        asset_id = str(payload["asset_id"])
        params_payload = payload.get("params")
        if not isinstance(params_payload, dict):
            raise ValueError("params must be an object")

        record = self.server.store.get_asset(asset_id)
        params = BayerParams.from_json(params_payload)
        started = time.perf_counter()
        with Image.open(record.path) as image:
            rendered = ordered_dither(image, params)
        buffer = io.BytesIO()
        rendered.save(buffer, format="PNG", optimize=False)
        png_bytes = buffer.getvalue()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        checksum, _ = self.server.store.save_output(png_bytes)

        self._send_json(
            {
                "output": {
                    "checksum": checksum,
                    "url": f"/api/outputs/{checksum}.png",
                    "width": rendered.width,
                    "height": rendered.height,
                    "render_ms": round(elapsed_ms, 2),
                }
            }
        )

    def _handle_recipe_save(self) -> None:
        payload = self._read_json()
        params_payload = payload.get("params")
        if not isinstance(params_payload, dict):
            raise ValueError("params must be an object")

        asset_id = str(payload["asset_id"])
        asset = self.server.store.get_asset(asset_id)
        recipe = Recipe.create(
            name=str(payload.get("name") or "Untitled recipe"),
            asset_id=asset.id,
            asset_checksum=asset.checksum,
            params=BayerParams.from_json(params_payload),
        )
        recipe_id, path = self.server.store.save_recipe(recipe)
        self._send_json({"id": recipe_id, "path": str(path), **recipe.to_json()})

    def _read_body(self, *, max_bytes: int = 1024 * 1024) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length > max_bytes:
            raise ValueError("request body too large")
        return self.rfile.read(length)

    def _read_json(self) -> dict[str, Any]:
        body = self._read_body()
        data = json.loads(body.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str | None = None) -> None:
        base = STATIC_DIR.resolve()
        resolved = path.resolve()
        if path.is_relative_to(STATIC_DIR) and not resolved.is_relative_to(base):
            self._send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        if not path.exists() or not path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "not found")
            return

        body = path.read_bytes()
        guessed_type = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", guessed_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status)


class ColorworksServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], store: LocalStore):
        super().__init__(server_address, ColorworksHandler)
        self.store = store


def run(host: str, port: int, data_dir: Path) -> None:
    store = LocalStore(data_dir)
    server = ColorworksServer((host, port), store)
    print(f"Colorworks running at http://{host}:{server.server_port}")
    print(f"Data directory: {data_dir.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Colorworks.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Colorworks Phase 0 local web tool.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--data-dir", type=Path, default=Path("colorworks_data"))
    args = parser.parse_args()
    run(args.host, args.port, args.data_dir)


if __name__ == "__main__":
    main()
