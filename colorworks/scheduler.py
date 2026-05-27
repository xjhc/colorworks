"""
In-process asyncio scheduler for Phase 3 preview/render runs.

One asyncio event loop runs in a background thread.
HTTP handlers submit coroutines to it and receive events via thread-safe queues.
"""
from __future__ import annotations

import asyncio
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from colorworks.domain import (
    CancelToken,
    PreviewRun,
    RenderRun,
    RenderResult,
    RunStatus,
    WarmStartState,
)


class RunScheduler:
    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir
        self._runs_dir.mkdir(parents=True, exist_ok=True)

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self._lock = threading.Lock()
        # run_id → PreviewRun | RenderRun
        self._runs: dict[str, Any] = {}
        # run_id → list of subscriber queues
        self._subscribers: dict[str, list[Queue]] = {}
        # run_id → ordered list of event dicts (for late subscribers)
        self._history: dict[str, list[dict]] = {}
        # (session_id, asset_id, algorithm_id) → WarmStartState
        self._warm_states: dict[tuple[str, str, str], WarmStartState] = {}
        # run_id → CancelToken
        self._cancel_tokens: dict[str, CancelToken] = {}
        # run_id → (ctx, artifact checksums) for promotion
        self._preview_contexts: dict[str, Any] = {}

        self._load_render_runs()

    # ── asyncio loop ──────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    # ── public API ────────────────────────────────────────────────────────────

    def submit_preview(
        self,
        run: PreviewRun,
        algo: Any,
        ctx: Any,          # RenderContext
        session_id: str,
    ) -> None:
        """Submit an iterative preview run. Supersedes other running previews in the session."""
        with self._lock:
            # Supersede currently-running previews for this session
            for rid, r in self._runs.items():
                if (
                    isinstance(r, PreviewRun)
                    and r.session_id == session_id
                    and r.status == RunStatus.RUNNING
                ):
                    tok = self._cancel_tokens.get(rid)
                    if tok:
                        tok.cancel()
                    r.superseded_at = datetime.now(timezone.utc)

            token = CancelToken()
            ctx.cancel = token
            self._runs[run.id] = run
            self._cancel_tokens[run.id] = token
            self._subscribers[run.id] = []
            self._history[run.id] = []

        asyncio.run_coroutine_threadsafe(
            self._execute(run, algo, ctx, session_id),
            self._loop,
        )

    def submit_render(
        self,
        run: RenderRun,
        algo: Any,
        ctx: Any,          # RenderContext
    ) -> None:
        """Submit a durable render run."""
        with self._lock:
            token = CancelToken()
            ctx.cancel = token
            self._runs[run.id] = run
            self._cancel_tokens[run.id] = token
            self._subscribers[run.id] = []
            self._history[run.id] = []

        asyncio.run_coroutine_threadsafe(
            self._execute(run, algo, ctx, None),
            self._loop,
        )

    def subscribe(self, run_id: str) -> Queue:
        """Return a queue that receives event dicts, then a None sentinel when done."""
        q: Queue = Queue()
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                q.put(None)
                return q
            history = list(self._history.get(run_id, []))
            terminal = run.status in (RunStatus.COMPLETED, RunStatus.CANCELLED, RunStatus.FAILED)
            if run_id not in self._subscribers:
                self._subscribers[run_id] = []
            self._subscribers[run_id].append(q)

        # Replay buffered events
        for event in history:
            q.put(event)
        if terminal:
            q.put(None)
        return q

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            tok = self._cancel_tokens.get(run_id)
        if tok:
            tok.cancel()
            return True
        return False

    def get_run(self, run_id: str) -> dict | None:
        with self._lock:
            r = self._runs.get(run_id)
        if r is None:
            return None
        return r.to_dict()

    def get_warm_state(
        self, session_id: str, asset_id: str, algorithm_id: str
    ) -> WarmStartState | None:
        with self._lock:
            return self._warm_states.get((session_id, asset_id, algorithm_id))

    def is_exportable(self, run_id: str) -> bool:
        """True only if the run completed non-partially (safe to export)."""
        with self._lock:
            r = self._runs.get(run_id)
        if r is None:
            return False
        return r.status == RunStatus.COMPLETED and not getattr(r, "_partial", False)

    def get_preview_context(self, run_id: str) -> Any | None:
        with self._lock:
            return self._preview_contexts.get(run_id)

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop the asyncio loop and background thread cleanly."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=timeout)

    # ── execution coroutine ───────────────────────────────────────────────────

    async def _execute(
        self,
        run: Any,
        algo: Any,
        ctx: Any,
        session_id: str | None,
    ) -> None:
        run.status = RunStatus.RUNNING
        try:
            async for progress in algo.render(ctx):
                event = _progress_to_dict(progress, run.id)
                self._broadcast(run.id, event)

                if progress.kind == "completed":
                    run.status = RunStatus.COMPLETED
                    run._partial = False
                    if progress.result:
                        run.primary_artifact_id = progress.result.algorithm_primary_artifact_id
                        run.final_artifact_id = progress.result.final_artifact_id
                    # For preview runs, stash ctx for promote
                    if isinstance(run, PreviewRun):
                        with self._lock:
                            self._preview_contexts[run.id] = ctx
                    # For render runs, persist to disk
                    if isinstance(run, RenderRun):
                        run.completed_at = datetime.now(timezone.utc)
                        run.artifact_ids = list(ctx.store.list())
                        self._persist_render_run(run, ctx)

                elif progress.kind == "cancelled":
                    run.status = RunStatus.CANCELLED
                    run._partial = True
                    if session_id and progress.result and progress.result.warm_state:
                        with self._lock:
                            key = (session_id, run.asset_id, run.algorithm_id)
                            self._warm_states[key] = progress.result.warm_state

                elif progress.kind == "failed":
                    run.status = RunStatus.FAILED
                    run._partial = True
                    if progress.result:
                        run.error = str(progress.result)

        except Exception as exc:
            run.status = RunStatus.FAILED
            run._partial = True
            run.error = str(exc)
            self._broadcast(run.id, {"kind": "failed", "run_id": run.id, "error": str(exc)})
        finally:
            self._broadcast(run.id, None)   # sentinel

    # ── helpers ───────────────────────────────────────────────────────────────

    def _broadcast(self, run_id: str, event: dict | None) -> None:
        with self._lock:
            if event is not None:
                hist = self._history.setdefault(run_id, [])
                hist.append(event)
            queues = list(self._subscribers.get(run_id, []))
        for q in queues:
            q.put(event)

    def _persist_render_run(self, run: RenderRun, ctx: Any) -> None:
        """Save run metadata + artifact blobs to disk so they survive restart."""
        run_path = self._runs_dir / f"{run.id}.json"
        meta = run.to_dict()
        meta["artifact_checksums"] = {}

        from colorworks.domain import (
            ScalarField, BinaryMask, VectorField2D, StructureTensorField,
            StrokeSet, PointSet,
        )
        import numpy as np
        from PIL import Image
        import io as _io

        for art_id in ctx.store.list():
            try:
                art = ctx.store.get(art_id)
            except KeyError:
                continue
            art_path = self._runs_dir / f"{run.id}_{art_id}.npy"
            try:
                val = art.value
                if isinstance(val, (ScalarField, BinaryMask, VectorField2D,
                                     StructureTensorField)):
                    np.save(str(art_path), val.data)
                elif isinstance(val, PointSet):
                    np.save(str(art_path), val.coords)
                elif isinstance(val, Image.Image):
                    img_path = self._runs_dir / f"{run.id}_{art_id}.png"
                    val.save(str(img_path), format="PNG")
                    art_path = img_path
                meta["artifact_checksums"][art_id] = art.checksum
            except Exception:
                pass

        run_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def _load_render_runs(self) -> None:
        """Restore completed RenderRun metadata from disk on startup."""
        for path in self._runs_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                run = RenderRun(
                    id=data["id"],
                    asset_id=data["asset_id"],
                    algorithm_id=data["algorithm_id"],
                    algorithm_version=data.get("algorithm_version", "1.0.0"),
                    params=data.get("params", {}),
                    composition=None,
                    seed=data.get("seed", 0),
                    quality=data.get("quality", "full"),
                    status=RunStatus(data.get("status", "completed")),
                    primary_artifact_id=data.get("primary_artifact_id"),
                    final_artifact_id=data.get("final_artifact_id"),
                    artifact_ids=data.get("artifact_ids", []),
                    promoted_from_preview_id=data.get("promoted_from_preview_id"),
                )
                run._partial = False
                with self._lock:
                    self._runs[run.id] = run
                    self._history[run.id] = []
            except Exception:
                pass


# ── helpers ───────────────────────────────────────────────────────────────────

def _progress_to_dict(progress: Any, run_id: str) -> dict:
    from colorworks.algorithms import RenderProgress
    event: dict[str, Any] = {"kind": progress.kind, "run_id": run_id}
    if progress.iteration is not None:
        event["iteration"] = progress.iteration
    if progress.total_iterations is not None:
        event["total_iterations"] = progress.total_iterations
    if progress.energy is not None:
        event["energy"] = progress.energy
    if progress.delta is not None:
        event["delta"] = progress.delta
    if progress.preview_artifact_id is not None:
        event["preview_artifact_id"] = progress.preview_artifact_id
    if progress.artifact_kind is not None:
        event["artifact_kind"] = progress.artifact_kind
    if progress.artifact_id is not None:
        event["artifact_id"] = progress.artifact_id
    if progress.result is not None:
        if progress.result.algorithm_primary_artifact_id:
            event["primary_artifact_id"] = progress.result.algorithm_primary_artifact_id
        if progress.result.final_artifact_id:
            event["final_artifact_id"] = progress.result.final_artifact_id
        if progress.result.warm_state is not None:
            event["warm_start_available"] = True
    return event
