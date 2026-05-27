from __future__ import annotations

import argparse
import io
import json
import mimetypes
import time
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from PIL import Image

from colorworks.recipe import Recipe
from colorworks.renderers.bayer import BayerParams, ordered_dither
from colorworks.storage import LocalStore

# Import registry and domain elements
from colorworks.domain import (
    ParameterDef,
    ParameterType,
    AlgorithmRole,
    AlgorithmDefinition,
    PatternKindDef,
    ScalarField,
    BinaryMask,
    Substrate,
    RasterGrid,
    ArtifactStore,
    Composition,
    InkLayerSpec,
    PaletteColor,
    PatternSpec,
    PatternCoordinateSpec,
)
from colorworks.algorithms import registry, MediaAsset, RenderContext
from colorworks.compositor import Compositor
# Import to populate registry
import colorworks.algorithms.tonal_analyzer
import colorworks.algorithms.pattern_catalog
import colorworks.algorithms.floyd_steinberg
import colorworks.algorithms.structure_analyzer

STATIC_DIR = Path(__file__).with_name("static")


def parameter_to_dict(p: ParameterDef) -> dict[str, Any]:
    res = {
        "key": p.key,
        "label": p.label,
        "type": p.type.value,
        "default": p.default,
        "group": p.group,
        "description": p.description,
    }
    if p.min is not None:
        res["min"] = p.min
    if p.max is not None:
        res["max"] = p.max
    if p.step is not None:
        res["step"] = p.step
    if p.options is not None:
        res["options"] = [
            {"value": option.value, "label": option.label}
            for option in p.options
        ]
    if p.ui_hint is not None:
        res["ui_hint"] = p.ui_hint
    if p.visible_when is not None and hasattr(p.visible_when, "param_key"):
        res["visible_when"] = {
            "param_key": p.visible_when.param_key,
            "value": p.visible_when.value
        }
    return res


def algorithm_to_dict(algo: Any) -> dict[str, Any]:
    defn = algo.definition
    return {
        "id": defn.id,
        "name": defn.name,
        "description": defn.description,
        "version": defn.version,
        "role": defn.role.value,
        "artifact_kinds": [
            {
                "name": artifact.name,
                "type": artifact.type,
                "label": artifact.label,
                "suitable_as": artifact.suitable_as,
            }
            for artifact in defn.artifact_kinds
        ],
        "parameters": [parameter_to_dict(p) for p in defn.parameters],
    }


def pattern_to_dict(pat: PatternKindDef) -> dict[str, Any]:
    return {
        "kind": pat.kind,
        "name": pat.name,
        "description": pat.description,
        "requires_orientation": pat.requires_orientation,
        "accepts_orientation": pat.accepts_orientation,
        "parameters": [parameter_to_dict(p) for p in pat.parameters],
    }


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
            elif route == "/api/schemas":
                algos = [algorithm_to_dict(a) for a in registry.list_algorithms()]
                patterns = [pattern_to_dict(p) for p in registry.list_patterns()]
                self._send_json({"algorithms": algos, "patterns": patterns})
            elif route.startswith("/api/artifacts/") and route.endswith("/meta"):
                artifact_id = route.split("/")[-2]
                path = self.server.store.get_artifact_path_by_id(artifact_id)
                if path is None:
                    self._send_error(HTTPStatus.NOT_FOUND, "artifact not found")
                else:
                    meta_path = path.with_suffix(".json")
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        self._send_json(meta)
                    else:
                        self._send_error(HTTPStatus.NOT_FOUND, "artifact meta not found")
            elif route.startswith("/api/artifacts/"):
                artifact_id = route.rsplit("/", 1)[-1]
                query = urlsplit(self.path).query
                view_mode = "default"
                if "view=" in query:
                    parts = query.split("&")
                    for p in parts:
                        if p.startswith("view="):
                            view_mode = p.split("=")[1]

                path = self.server.store.get_artifact_path_by_id(artifact_id)
                if path is None or not path.exists():
                    self._send_error(HTTPStatus.NOT_FOUND, f"artifact file {artifact_id} not found")
                else:
                    if view_mode == "orientation_hsv":
                        view_path = path.parent / f"{path.stem}_hsv.png"
                    elif view_mode == "glyph_field":
                        view_path = path.parent / f"{path.stem}_glyphs.png"
                    else:
                        view_path = path

                    if view_path.exists():
                        self._send_file(view_path, "image/png")
                    else:
                        self._send_file(path, "image/png")
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
            elif route == "/api/presets":
                self._send_json({"presets": self.server.store.list_presets()})
            elif route.startswith("/api/presets/"):
                preset_id = route.rsplit("/", 1)[-1]
                preset = self.server.store.get_preset(preset_id)
                self._send_json(preset)
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
            elif self.path == "/api/presets":
                self._handle_preset_save()
            elif self.path == "/api/export/svg":
                self._handle_export_svg()
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "not found")
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except KeyError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_DELETE(self) -> None:
        try:
            route = urlsplit(self.path).path
            if route.startswith("/api/presets/"):
                preset_id = route.rsplit("/", 1)[-1]
                self.server.store.delete_preset(preset_id)
                self._send_json({"status": "deleted"})
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "not found")
        except KeyError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _handle_preset_save(self) -> None:
        payload = self._read_json()
        preset_id = self.server.store.save_preset(payload)
        preset = self.server.store.get_preset(preset_id)
        self._send_json(preset)

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
        renderer_id = str(payload.get("renderer_id", "ordered_bayer"))

        record = self.server.store.get_asset(asset_id)
        started = time.perf_counter()

        if renderer_id == "ordered_bayer":
            params_payload = payload.get("params")
            if not isinstance(params_payload, dict):
                raise ValueError("params must be an object")
            params = BayerParams.from_json(params_payload)
            with Image.open(record.path) as image:
                rendered = ordered_dither(image, params)
            buffer = io.BytesIO()
            rendered.save(buffer, format="PNG", optimize=False)
            png_bytes = buffer.getvalue()
            checksum, _ = self.server.store.save_output(png_bytes)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

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
            return

        # Check if algorithm is in registry
        algo = None
        try:
            algo = registry.get(renderer_id)
        except KeyError:
            pass

        if algo is not None:
            # 1. Use the shared pipeline execution helper
            ctx, comp_obj, algo, enabled_artifacts = self._execute_pipeline(payload)

            if algo.definition.role == AlgorithmRole.RENDERER:
                # Renderer direct execution path (bypasses compositor)
                final_key = self.server.store.get_artifact_cache_key(
                    algo_id=algo.definition.id,
                    algo_version=algo.definition.version,
                    artifact_name="final_raster",
                    asset_checksum=record.checksum,
                    params=ctx.params,
                    parameters_def=algo.definition.parameters,
                )
                final_cache = self.server.store.get_cached_artifact(final_key)

                if final_cache is not None:
                    print(f"[CACHE] final_raster (renderer): HIT")
                    final_png_path, final_meta = final_cache
                    final_checksum = final_meta["checksum"]
                    final_bytes = final_png_path.read_bytes()
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                else:
                    print(f"[CACHE] final_raster (renderer): MISS")
                    algo.analyze(ctx)
                    res_result = algo.compose(ctx)
                    primary_art_id = res_result.algorithm_primary_artifact_id
                    art = ctx.store.get(primary_art_id)

                    if isinstance(art.value, Image.Image):
                        final_img = art.value
                    elif hasattr(art.value, "data"):
                        final_img = Image.fromarray(art.value.data)
                    else:
                        raise ValueError("primary artifact for renderer must be an image")

                    buf = io.BytesIO()
                    final_img.save(buf, format="PNG")
                    final_bytes = buf.getvalue()
                    final_checksum = hashlib.sha256(final_bytes).hexdigest()

                    # Save to cache
                    self.server.store.save_cached_artifact(final_key, final_bytes, {
                        "id": f"final_raster_{final_checksum[:16]}",
                        "name": "final_raster",
                        "type": "raster_image",
                        "checksum": final_checksum
                    })
                    # Save to normal outputs dir
                    self.server.store.save_output(final_bytes)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0

                self._send_json(
                    {
                        "output": {
                            "checksum": final_checksum,
                            "url": f"/api/outputs/{final_checksum}.png",
                            "width": record.width,
                            "height": record.height,
                            "render_ms": round(elapsed_ms, 2),
                        },
                        "artifacts": {
                            "final_raster": {
                                "id": f"final_raster_{final_checksum[:16]}",
                                "url": f"/api/outputs/{final_checksum}.png"
                            }
                        }
                    }
                )
                return

            # Analyzer path (runs Compositor)
            # Resolve referenced artifact checksums for the final raster cache key
            referenced_checksums = {}
            for layer in comp_obj.layers:
                for src in (
                    layer.density_source,
                    layer.pattern.mask_source,
                    layer.pattern.orientation_source,
                    layer.pattern.field_source,
                    layer.pattern.warp_source,
                ):
                    if src:
                        try:
                            art = ctx.store.get_by_name(src)
                            referenced_checksums[src] = art.checksum
                        except KeyError:
                            pass

            # Canonical snapshot
            comp_snapshot = {
                "paper_color": {"hex": comp_obj.paper_color.hex},
                "layers": [
                    {
                        "name": l.name,
                        "color": {"hex": l.color.hex},
                        "role": l.role,
                        "density_source": l.density_source,
                        "density_source_checksum": referenced_checksums.get(l.density_source, ""),
                        "pattern": {
                            "kind": l.pattern.kind,
                            "params": l.pattern.params,
                            "mask_source": l.pattern.mask_source,
                            "mask_source_checksum": referenced_checksums.get(l.pattern.mask_source, "") if l.pattern.mask_source else "",
                            "orientation_source": l.pattern.orientation_source,
                            "orientation_source_checksum": referenced_checksums.get(l.pattern.orientation_source, "") if l.pattern.orientation_source else "",
                            "field_source": l.pattern.field_source,
                            "field_source_checksum": referenced_checksums.get(l.pattern.field_source, "") if l.pattern.field_source else "",
                            "warp_source": l.pattern.warp_source,
                            "warp_source_checksum": referenced_checksums.get(l.pattern.warp_source, "") if l.pattern.warp_source else "",
                            "coordinates": {
                                "space": l.pattern.coordinates.space,
                                "origin": l.pattern.coordinates.origin,
                                "scale": l.pattern.coordinates.scale,
                                "rotation_deg": l.pattern.coordinates.rotation_deg,
                                "seed": l.pattern.coordinates.seed if l.pattern.coordinates.seed is not None else ctx.seed,
                            }
                        },
                        "threshold": l.threshold,
                        "blend_mode": l.blend_mode,
                        "opacity": l.opacity,
                        "priority": l.priority,
                    }
                    for l in comp_obj.layers
                ],
                "width": record.width,
                "height": record.height,
            }
            comp_json = json.dumps(comp_snapshot, sort_keys=True)
            final_key = self.server.store.get_final_raster_cache_key(
                referenced_checksums.get("tone_map", ""),
                referenced_checksums.get("edge_mask"),
                comp_json
            )
            final_cache = self.server.store.get_cached_artifact(final_key)

            if final_cache is not None:
                print(f"[CACHE] final_raster: HIT")
                final_png_path, final_meta = final_cache
                final_checksum = final_meta["checksum"]
                final_bytes = final_png_path.read_bytes()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
            else:
                print(f"[CACHE] final_raster: MISS")
                compositor = Compositor(ctx.store)
                final_img = compositor.composite(comp_obj, record.width, record.height, ctx.seed)

                buf = io.BytesIO()
                final_img.save(buf, format="PNG")
                final_bytes = buf.getvalue()
                final_checksum = hashlib.sha256(final_bytes).hexdigest()

                # Save to cache
                self.server.store.save_cached_artifact(final_key, final_bytes, {
                    "id": f"final_raster_{final_checksum[:16]}",
                    "name": "final_raster",
                    "type": "raster_image",
                    "checksum": final_checksum
                })
                # Save to normal outputs dir as well
                self.server.store.save_output(final_bytes)
                elapsed_ms = (time.perf_counter() - started) * 1000.0

            # Construct dynamic artifact descriptions for the response
            artifact_responses = {}
            for name in enabled_artifacts:
                try:
                    art = ctx.store.get_by_name(name)
                    artifact_responses[name] = {
                        "id": art.id,
                        "url": f"/api/artifacts/{art.id}"
                    }
                except KeyError:
                    pass

            self._send_json(
                {
                    "output": {
                        "checksum": final_checksum,
                        "url": f"/api/outputs/{final_checksum}.png",
                        "width": record.width,
                        "height": record.height,
                        "render_ms": round(elapsed_ms, 2),
                    },
                    "artifacts": artifact_responses
                }
            )
            return

        else:
            raise ValueError(f"unsupported renderer_id: {renderer_id}")

    def _execute_pipeline(self, payload: dict[str, Any]) -> tuple[RenderContext, Composition | None, Any, list[str]]:
        asset_id = str(payload["asset_id"])
        renderer_id = str(payload.get("renderer_id", "ordered_bayer"))

        record = self.server.store.get_asset(asset_id)

        algo = registry.get(renderer_id)

        params_dict = payload.get("params", {})
        comp_dict = payload.get("composition", {})

        # Normalize parameters dynamically
        params = {}
        for param_def in algo.definition.parameters:
            val = params_dict.get(param_def.key)
            if val is None:
                val = param_def.default
            if param_def.type == ParameterType.FLOAT:
                val = float(val)
            elif param_def.type == ParameterType.INT:
                val = int(val)
            elif param_def.type == ParameterType.BOOL:
                val = bool(val)
            elif param_def.type == ParameterType.STR:
                val = str(val)
            params[param_def.key] = val

        # Load the source asset image
        substrate = RasterGrid(record.width, record.height)
        with Image.open(record.path) as img:
            img_loaded = img.copy()
        asset = MediaAsset(id=record.id, checksum=record.checksum, image=img_loaded, substrate=substrate)

        # Setup active run ArtifactStore
        active_store = ArtifactStore(output_dir=None)
        ctx = RenderContext(
            input=asset,
            params=params,
            composition=None,
            seed=int(payload.get("seed", 42)),
            store=active_store,
        )

        if algo.definition.role == AlgorithmRole.RENDERER:
            return ctx, None, algo, []

        # Analyzer path
        enabled_artifacts = [
            name for name in algo.produced_in_analyze
            if algo.is_artifact_enabled(name, params)
        ]

        # Compute cache keys for the intermediate artifacts dynamically
        analyze_cache_keys = {}
        for name in enabled_artifacts:
            key = self.server.store.get_artifact_cache_key(
                algo_id=algo.definition.id,
                algo_version=algo.definition.version,
                artifact_name=name,
                asset_checksum=record.checksum,
                params=params,
                parameters_def=algo.definition.parameters
            )
            analyze_cache_keys[name] = key

        # Check cache hits for analyze stage
        analyze_hit = True
        cached_data = {}
        cached_meta = {}
        for name in enabled_artifacts:
            key = analyze_cache_keys[name]
            res = self.server.store.get_cached_artifact(key)
            if res is None:
                analyze_hit = False
                break
            else:
                cached_data[name] = res[0]
                cached_meta[name] = res[1]

        # Run analyze stage (with authoritative caching checks)
        if analyze_hit:
            print(f"[CACHE] analyze stage: HIT")
            published_ids = {}
            for name in enabled_artifacts:
                arr = cached_data[name]
                meta = cached_meta[name]
                type_name = meta.get("type")
                if type_name == "scalar_field":
                    field = ScalarField(substrate, arr, "float32")
                    pub_id = active_store.publish(name, field)
                elif type_name == "binary_mask":
                    mask = BinaryMask(substrate, arr)
                    pub_id = active_store.publish(name, mask)
                elif type_name == "vector_field_2d":
                    from colorworks.domain import VectorField2D
                    field = VectorField2D(substrate, arr, is_bidirectional=meta.get("is_bidirectional", False))
                    pub_id = active_store.publish(name, field)
                elif type_name == "structure_tensor_field":
                    from colorworks.domain import StructureTensorField
                    field = StructureTensorField(substrate, arr)
                    pub_id = active_store.publish(name, field)
                else:
                    pub_id = active_store.publish(name, arr)
                published_ids[name] = pub_id
            algo.load_from_cache(ctx, published_ids)
        else:
            print(f"[CACHE] analyze stage: MISS (or partial hit)")
            published_ids = {}
            for name in enabled_artifacts:
                key = analyze_cache_keys[name]
                res = self.server.store.get_cached_artifact(key)
                if res is not None:
                    arr, meta = res
                    type_name = meta.get("type")
                    if type_name == "scalar_field":
                        field = ScalarField(substrate, arr, "float32")
                        pub_id = active_store.publish(name, field)
                    elif type_name == "binary_mask":
                        mask = BinaryMask(substrate, arr)
                        pub_id = active_store.publish(name, mask)
                    elif type_name == "vector_field_2d":
                        from colorworks.domain import VectorField2D
                        field = VectorField2D(substrate, arr, is_bidirectional=meta.get("is_bidirectional", False))
                        pub_id = active_store.publish(name, field)
                    elif type_name == "structure_tensor_field":
                        from colorworks.domain import StructureTensorField
                        field = StructureTensorField(substrate, arr)
                        pub_id = active_store.publish(name, field)
                    else:
                        pub_id = active_store.publish(name, arr)
                    published_ids[name] = pub_id
            algo.load_from_cache(ctx, published_ids)

            algo.analyze(ctx)

            # Save newly generated artifacts to cache
            for name in enabled_artifacts:
                try:
                    art = active_store.get_by_name(name)
                    key = analyze_cache_keys[name]
                    from colorworks.domain import VectorField2D, StructureTensorField
                    if isinstance(art.value, (ScalarField, BinaryMask, VectorField2D, StructureTensorField)):
                        raw_data = art.value.data
                    else:
                        raw_data = art.value

                    meta_dict = {
                        "id": art.id,
                        "name": name,
                        "type": art.type,
                        "checksum": art.checksum
                    }
                    if isinstance(art.value, VectorField2D):
                        meta_dict["is_bidirectional"] = art.value.is_bidirectional

                    self.server.store.save_cached_artifact(key, raw_data, meta_dict)
                except KeyError:
                    pass

        # Run compose stage
        res_result = algo.compose(ctx)

        # Setup composition spec
        if not comp_dict:
            comp_obj = res_result.default_composition
            if comp_obj is None:
                raise ValueError(f"algorithm {renderer_id} did not provide a composition")
        else:
            paper_hex = comp_dict.get("paper_color", {}).get("hex", "#f4ebd9")
            layers_list = []
            for l in comp_dict.get("layers", []):
                pat_dict = l.get("pattern", {})
                pat_spec = PatternSpec(
                    kind=pat_dict.get("kind", "wave"),
                    params=pat_dict.get("params", {}),
                    field_source=pat_dict.get("field_source"),
                    orientation_source=pat_dict.get("orientation_source"),
                    warp_source=pat_dict.get("warp_source"),
                    mask_source=pat_dict.get("mask_source"),
                    coordinates=PatternCoordinateSpec(
                        space=pat_dict.get("coordinates", {}).get("space", "image_px"),
                        origin=tuple(pat_dict.get("coordinates", {}).get("origin", [0.0, 0.0])),
                        scale=float(pat_dict.get("coordinates", {}).get("scale", 1.0)),
                        rotation_deg=float(pat_dict.get("coordinates", {}).get("rotation_deg", 0.0)),
                        seed=pat_dict.get("coordinates", {}).get("seed"),
                    )
                )
                layers_list.append(InkLayerSpec(
                    name=l.get("name", "ink"),
                    color=PaletteColor(l.get("color", {}).get("hex", "#1a1a1a")),
                    role=l.get("role", "shadow"),
                    density_source=l.get("density_source", "tone_map"),
                    pattern=pat_spec,
                    threshold=l.get("threshold"),
                    blend_mode=l.get("blend_mode", "normal"),
                    opacity=float(l.get("opacity", 1.0)),
                    priority=int(l.get("priority", 0)),
                ))
            comp_obj = Composition(
                paper_color=PaletteColor(paper_hex),
                layers=layers_list,
            )

        return ctx, comp_obj, algo, enabled_artifacts

    def _handle_export_svg(self) -> None:
        payload = self._read_json()
        ctx, comp_obj, algo, enabled_artifacts = self._execute_pipeline(payload)
        record = self.server.store.get_asset(payload["asset_id"])

        has_stroke_layer = False
        for l in comp_obj.layers:
            if l.pattern.kind in ("hatch", "crosshatch"):
                has_stroke_layer = True
                break

        if not has_stroke_layer:
            self._send_error(HTTPStatus.BAD_REQUEST, "Composition has no hatch/crosshatch stroke layers for SVG export")
            return

        compositor = Compositor(ctx.store)
        stroke_sets = compositor.build_stroke_set(comp_obj, record.width, record.height, ctx.seed)

        svg_str = self._serialize_to_svg(comp_obj, stroke_sets, record.width, record.height)

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/svg+xml")
        self.send_header("Content-Length", str(len(svg_str)))
        self.send_header("Content-Disposition", f"attachment; filename=colorworks-{record.id[:8]}.svg")
        self.end_headers()
        self.wfile.write(svg_str.encode("utf-8"))

    def _serialize_to_svg(self, composition: Composition, stroke_sets: list[tuple[InkLayerSpec, StrokeSet]], width: int, height: int) -> str:
        paper_hex = composition.paper_color.hex

        lines = []
        lines.append(f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
        lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
        lines.append(f'  <rect width="100%" height="100%" fill="{paper_hex}" />')

        for layer, stroke_set in stroke_sets:
            ink_hex = layer.color.hex
            opacity = layer.opacity
            safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in layer.name)
            lines.append(f'  <g id="layer_{safe_name}" stroke="{ink_hex}" opacity="{opacity}" stroke-linecap="round" fill="none">')

            for stroke in stroke_set.strokes:
                pts = stroke.path.points
                if len(pts) < 2:
                    continue
                if stroke.width_profile is not None:
                    for i in range(len(pts) - 1):
                        p1 = pts[i]
                        p2 = pts[i + 1]
                        w = float((stroke.width_profile[i] + stroke.width_profile[i + 1]) / 2.0)
                        lines.append(f'    <line x1="{p1[0]:.2f}" y1="{p1[1]:.2f}" x2="{p2[0]:.2f}" y2="{p2[1]:.2f}" stroke-width="{w:.2f}" />')
                else:
                    d_path = "M " + " L ".join(f"{p[0]:.2f} {p[1]:.2f}" for p in pts)
                    lines.append(f'    <path d="{d_path}" stroke-width="1.0" />')
            lines.append('  </g>')

        lines.append('</svg>')
        return "\n".join(lines)

    def _handle_recipe_save(self) -> None:
        payload = self._read_json()
        renderer_id = str(payload.get("renderer_id", "ordered_bayer"))
        asset_id = str(payload["asset_id"])
        asset = self.server.store.get_asset(asset_id)

        if renderer_id == "ordered_bayer":
            params_payload = payload.get("params")
            recipe = Recipe.create(
                name=str(payload.get("name") or "Untitled recipe"),
                asset_id=asset.id,
                asset_checksum=asset.checksum,
                params=BayerParams.from_json(params_payload),
            )
        else:
            algo = registry.get(renderer_id)
            params = payload.get("params", {})
            composition = None
            if algo.definition.role != AlgorithmRole.RENDERER:
                composition = payload.get("composition", {})
            recipe = Recipe.create(
                name=str(payload.get("name") or "Untitled recipe"),
                asset_id=asset.id,
                asset_checksum=asset.checksum,
                params=params,
                composition=composition,
                renderer_id=renderer_id,
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
        # Resolve path safely
        resolved = path.resolve()

        # Determine valid base directories
        static_base = STATIC_DIR.resolve()
        assets_base = self.server.store.assets_dir.resolve()
        outputs_base = self.server.store.outputs_dir.resolve()
        artifacts_base = self.server.store.artifacts_dir.resolve()

        # Verify that path belongs to one of allowed directories
        is_safe = (
            resolved.is_relative_to(static_base) or
            resolved.is_relative_to(assets_base) or
            resolved.is_relative_to(outputs_base) or
            resolved.is_relative_to(artifacts_base)
        )

        if not is_safe:
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
    parser = argparse.ArgumentParser(description="Run the Colorworks local web tool.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--data-dir", type=Path, default=Path("colorworks_data"))
    args = parser.parse_args()
    run(args.host, args.port, args.data_dir)


if __name__ == "__main__":
    main()
