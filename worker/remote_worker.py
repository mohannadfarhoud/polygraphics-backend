"""Remote GPU worker: WebSocket job notifications + REST assignment, progress, and upload.

Environment (see ``.env.worker.example``):

* ``POLYGRAPH_API_BASE`` — HTTP API root, e.g. ``https://host/polygraph``
* ``POLYGRAPH_WORKER_TOKEN`` — must match ``APP_WORKER_TOKEN`` on the API server
* ``POLYGRAPH_USE_WEBSOCKET`` — ``1``/``true`` to subscribe to ``wss://.../internal/worker/ws`` (default on)
* ``POLYGRAPH_WEBSOCKET_URL`` — optional full ``wss://host/...`` WebSocket path if auto URL returns 404 behind nginx
* ``POLYGRAPH_WS_TRY_STRIPPED`` — ``1`` (default) also try ``wss://host/internal/worker/ws`` when the prefixed URL 404s
* ``POLYGRAPH_WS_PING_INTERVAL`` / ``POLYGRAPH_WS_PING_TIMEOUT`` — WebSocket keepalive seconds (defaults ``30`` / ``600``) while **waiting** for work only; the socket is **closed after each job_assigned** and stays disconnected until the job finishes (progress uses REST ``POST .../progress``), then reconnects — avoids 1011 ping timeouts during long GPU runs
* ``POLYGRAPH_OVERRIDE_DEVICE`` — optional ``cuda`` / ``cpu`` / ``auto``; if unset, worker uses ``cuda`` when ``torch.cuda.is_available()`` else keeps API ``device``
* ``POLYGRAPH_POLL_SECONDS`` — fallback polling interval for ``GET /internal/worker/next`` (default ``30``)
* ``POLYGRAPH_POLL_WHILE_WEBSOCKET`` — when ``1``/``true``, keep that interval even if WebSocket is on; default is **off** so REST polling is slowed (min ~180s) while WS carries ``job_assigned`` — fewer duplicate wakes / log spam
* ``POLYGRAPH_PROGRESS_INTERVAL_SECONDS`` — min seconds between ``POST .../progress`` calls (default ``5``)
* ``POLYGRAPH_REQUIRE_CUDA`` — ``1``/``true`` to exit immediately if ``torch.cuda.is_available()`` is false
* ``POLYGRAPH_WORKER_PRESERVE_WORKDIR`` — ``1``/``true`` keeps per-job temp folders on the worker for inspection
* ``POLYGRAPH_WORKER_WORKDIR_ROOT`` — optional parent dir for those per-job folders (default OS temp dir)
* ``expose_masked_views`` (API ``PUT /settings``): when true, worker uploads SAM/precut ``masked_*`` images to ``POST /internal/worker/jobs/{id}/masked-views`` — browse via GET ``uploads/{job_id}/masked_views/``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

import httpx


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


def _make_worker_job_dir(job_id: str) -> Path:
    """Create a per-job workspace under OS temp or ``POLYGRAPH_WORKER_WORKDIR_ROOT``."""
    root_raw = os.getenv("POLYGRAPH_WORKER_WORKDIR_ROOT", "").strip()
    kwargs: dict[str, str] = {"prefix": f"polyjob-{job_id}-"}
    if root_raw:
        root = Path(root_raw).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        kwargs["dir"] = str(root)
    return Path(tempfile.mkdtemp(**kwargs))


def _apply_local_overrides(settings_dict: dict) -> dict:
    mapping = (
        ("sam_checkpoint_path", "POLYGRAPH_OVERRIDE_SAM_CHECKPOINT"),
        ("mapanything_pretrained_id", "POLYGRAPH_OVERRIDE_MAPANYTHING_MODEL"),
        ("dust3r_checkpoint_path", "POLYGRAPH_OVERRIDE_DUST3R_CHECKPOINT"),
        ("gs_repo_path", "POLYGRAPH_OVERRIDE_GS_REPO"),
        ("ai_prior_command", "POLYGRAPH_OVERRIDE_AI_PRIOR_COMMAND"),
        ("ai_prior_triposr_repo_path", "POLYGRAPH_OVERRIDE_TRIPOSR_REPO"),
        ("ai_prior_triposr_python_executable", "POLYGRAPH_OVERRIDE_TRIPOSR_PYTHON"),
    )
    out = dict(settings_dict)
    for key, env in mapping:
        v = os.getenv(env, "").strip()
        if v:
            out[key] = v
    dev_raw = os.getenv("POLYGRAPH_OVERRIDE_DEVICE", "").strip()
    if dev_raw:
        d = dev_raw.lower()
        if d in ("auto", "cpu", "cuda"):
            out["device"] = d
    else:
        # Prefer GPU on this machine; API ``device`` is for in-process API runs, not the worker.
        try:
            import torch

            if torch.cuda.is_available():
                out["device"] = "cuda"
        except ImportError:
            pass

    return out


def _fetch_latest_settings(client: httpx.Client, base: str, token: str) -> dict | None:
    """Fetch latest runtime settings from API right before running a job.

    Falls back to assignment payload settings when API read fails.
    """
    try:
        r = client.get(
            f"{base.rstrip('/')}/settings",
            headers=_headers(token),
            timeout=60.0,
        )
        if r.status_code != 200:
            body = (r.text or "")[:240]
            print(
                f"[polygraph-worker] warning: GET /settings -> {r.status_code}; "
                f"using assignment settings ({body})",
                flush=True,
            )
            return None
        data = r.json()
        if isinstance(data, dict):
            return data
        print("[polygraph-worker] warning: GET /settings returned non-object; using assignment settings", flush=True)
        return None
    except Exception as exc:
        print(f"[polygraph-worker] warning: GET /settings failed; using assignment settings ({exc})", flush=True)
        return None


def iter_worker_websocket_uris(http_base: str, token: str) -> list[str]:
    """Candidate ``wss://`` URLs to try when connecting (proxy path / nginx quirks)."""
    seen: set[str] = set()
    ordered: list[str] = []

    def add(uri: str) -> None:
        if uri not in seen:
            seen.add(uri)
            ordered.append(uri)

    raw = os.getenv("POLYGRAPH_WEBSOCKET_URL", "").strip()
    if raw:
        u = urlparse(raw)
        scheme = u.scheme or "wss"
        path = (u.path or "/internal/worker/ws").rstrip("/") or "/internal/worker/ws"
        q = urlencode({"token": token})
        add(urlunparse((scheme, u.netloc, path, "", q, "")))

    u = urlparse(http_base.strip().rstrip("/"))
    scheme = "wss" if u.scheme == "https" else "ws"
    q = urlencode({"token": token})
    base_path = u.path.rstrip("/")
    try_stripped = os.getenv("POLYGRAPH_WS_TRY_STRIPPED", "1").strip().lower() in ("1", "true", "yes")

    if base_path:
        add(urlunparse((scheme, u.netloc, f"{base_path}/internal/worker/ws", "", q, "")))
    if try_stripped or not base_path:
        add(urlunparse((scheme, u.netloc, "/internal/worker/ws", "", q, "")))

    return ordered


def build_worker_websocket_uri(http_base: str, token: str) -> str:
    """Build ``wss://...`` URI; optional ``POLYGRAPH_WEBSOCKET_URL`` overrides path/host."""
    uris = iter_worker_websocket_uris(http_base, token)
    return uris[0] if uris else ""


def _mask_ws_uri(uri: str) -> str:
    if "token=" in uri:
        return uri.split("token=", 1)[0] + "token=***"
    return uri


def _require_cuda_if_configured() -> None:
    raw = os.getenv("POLYGRAPH_REQUIRE_CUDA", "").strip().lower()
    if raw not in ("1", "true", "yes"):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("POLYGRAPH_REQUIRE_CUDA is set but torch.cuda.is_available() is False")


class ApiReportingJobRepository:
    """Forwards pipeline ``PROCESSING`` stage/progress to the API (throttled)."""

    def __init__(self, client: httpx.Client, base: str, token: str, job_id: str) -> None:
        self._client = client
        self._base = base.rstrip("/")
        self._token = token
        self._job_id = job_id
        self._last_post = float("-inf")
        self._last_stage: str | None = None
        self._interval = float(os.environ.get("POLYGRAPH_PROGRESS_INTERVAL_SECONDS", "5"))

    def set_status(
        self,
        job_id: str,
        status: Any,
        *,
        stage: str | None = None,
        progress: int | None = None,
        model_url: str | None = None,
        model_format: str | None = None,
        error: str | None = None,
    ) -> None:
        from app.interfaces import JobStatus

        if job_id != self._job_id:
            return
        if status != JobStatus.PROCESSING:
            return
        if stage is None and progress is None:
            return
        now = time.monotonic()
        stage_changed = stage is not None and stage != self._last_stage
        if not stage_changed and now - self._last_post < self._interval:
            return
        try:
            self._last_post = now
            if stage is not None:
                self._last_stage = stage
            r = self._client.post(
                f"{self._base}/internal/worker/jobs/{job_id}/progress",
                headers={"X-Worker-Token": self._token},
                json={
                    "stage": stage or "processing",
                    "progress": int(progress) if progress is not None else 0,
                },
                timeout=60.0,
            )
            if r.status_code >= 400:
                print(
                    f"[polygraph-worker] progress POST failed job={job_id} "
                    f"HTTP {r.status_code}: {(r.text or '')[:240]}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[polygraph-worker] progress POST error job={job_id}: {exc}", flush=True)


def _run_one_job(base: str, token: str, payload: dict, client: httpx.Client | None = None) -> None:
    job_id = payload["job_id"]
    payload_settings = dict(payload.get("settings") or {})
    image_urls: list[str] = list(payload.get("image_urls") or [])

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=600.0)

    work = _make_worker_job_dir(job_id)
    settings = None
    try:
        from app.config import PipelineConfig
        from app.interfaces import NoopWebSocketNotifier
        from app.pipeline import ReconstructionPipeline
        from app.pipeline_ready import assert_pipeline_ready
        from app.reconstruction import Dust3RReconstructor
        from app.runtime_settings import RuntimeSettings, effective_reconstruction_backend, minimum_input_images
        from app.segmentation import SamSegmenter

        from app.cuda_memory import bootstrap_worker_cuda

        bootstrap_worker_cuda()

        refresh_settings = os.getenv("POLYGRAPH_WORKER_REFRESH_SETTINGS_BEFORE_JOB", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        settings_source = "assignment"
        settings_dict = payload_settings
        if refresh_settings:
            latest = _fetch_latest_settings(client, base, token)
            if latest:
                settings_dict = latest
                settings_source = "api"
        settings_dict = _apply_local_overrides(settings_dict)
        settings = RuntimeSettings.model_validate(settings_dict)
        print(f"[polygraph-worker] job {job_id}: settings source={settings_source} (local overrides applied)", flush=True)
        _require_cuda_if_configured()
        print(f"[polygraph-worker] job {job_id}: validating checkpoints and backends...", flush=True)
        assert_pipeline_ready(settings)

        min_images = int(minimum_input_images(settings))
        if len(image_urls) < min_images:
            raise RuntimeError(
                f"Assignment lists {len(image_urls)} image URL(s); need at least {min_images}. "
                "Confirm uploads finished and POST /jobs/{job_id}/start ran on the API."
            )

        upload_dir = work / "uploads" / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        print(f"[polygraph-worker] job {job_id}: downloading {len(image_urls)} images, then running 3D pipeline...", flush=True)
        for i, url in enumerate(image_urls):
            full = url if url.startswith("http") else f"{base.rstrip('/')}{url}"
            resp = client.get(full)
            try:
                resp.raise_for_status()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to fetch image {i + 1}/{len(image_urls)} ({full}): {exc}"
                ) from exc
            suffix = Path(url).suffix or ".jpg"
            out_path = upload_dir / f"input_{i:03d}{suffix}"
            data = resp.content
            out_path.write_bytes(data)
            print(
                f"[polygraph-worker] job {job_id}: saved {out_path.name} ({len(data)} bytes)",
                flush=True,
            )

        cfg = PipelineConfig(
            root_dir=work,
            output_dir_name=settings.output_dir_name,
            masked_dir_name=settings.masked_dir_name,
            masks_dir_name=settings.masks_dir_name,
            nb_neighbors=settings.nb_neighbors,
            std_ratio=settings.std_ratio,
            poisson_depth=settings.poisson_depth,
            poisson_density_quantile=settings.poisson_density_quantile,
            decimation_target_triangles=settings.decimation_target_triangles,
            cdn_base_url=settings.cdn_base_url,
        )

        job_repo = ApiReportingJobRepository(client, base, token, job_id)
        pipe = ReconstructionPipeline(
            cfg,
            settings,
            segmenter=SamSegmenter(settings),
            reconstructor=Dust3RReconstructor(settings),
            job_repo=job_repo,
            notifier=NoopWebSocketNotifier(),
        )
        paths = sorted(upload_dir.glob("input_*"))
        print(f"[polygraph-worker] job {job_id}: reconstructing 3D model from {len(paths)} local images...", flush=True)
        pipe.process_3d_job(job_id, paths)
        if getattr(settings, "expose_masked_views", False):
            masked_job_dir = work / settings.masked_dir_name / job_id
            _upload_masked_views_to_api(client, base, token, job_id, masked_job_dir)

        out_dir = work / settings.output_dir_name
        if (
            effective_reconstruction_backend(settings) == "gaussian_splatting"
            and settings.compare_mesh_preview_with_gs
        ):
            stem = f"{job_id}_compare_mesh"
            sidecar = out_dir / f"{stem}.glb"
            if not sidecar.is_file():
                print(
                    f"[polygraph-worker] job {job_id}: no comparison GLB at {sidecar.name} (skipping upload)",
                    flush=True,
                )
            else:
                body = sidecar.read_bytes()
                ur = client.post(
                    f"{base}/internal/worker/jobs/{job_id}/comparison-glb",
                    headers={"X-Worker-Token": token},
                    data={"variant": "mesh"},
                    files={"file": (f"{stem}.glb", body, "model/gltf-binary")},
                )
                ur.raise_for_status()
                print(
                    f"[polygraph-worker] job {job_id}: uploaded preview comparison GLB ({len(body)} bytes)",
                    flush=True,
                )

        glb = out_dir / f"{job_id}.glb"
        ply = out_dir / f"{job_id}.ply"
        if ply.is_file():
            ext, data = "ply", ply.read_bytes()
        elif glb.is_file():
            ext, data = "glb", glb.read_bytes()
        else:
            raise RuntimeError(f"No output .glb or .ply under {out_dir}")

        r = client.post(
            f"{base}/internal/worker/jobs/{job_id}/complete",
            headers={"X-Worker-Token": token},
            data={"model_format": ext},
            files={"file": (f"{job_id}.{ext}", data, "application/octet-stream")},
        )
        r.raise_for_status()
    finally:
        try:
            from app.cuda_memory import purge_torch_cuda

            purge_torch_cuda()
        except Exception:
            pass
        if _env_truthy("POLYGRAPH_WORKER_PRESERVE_WORKDIR"):
            msg = f"[polygraph-worker] keeping work dir for inspection: {work}"
            if settings is not None:
                masked_dir = work / settings.masked_dir_name / job_id
                masks_dir = work / settings.masks_dir_name / job_id
                msg += f" | masked={masked_dir} | masks={masks_dir}"
            print(msg, flush=True)
        else:
            shutil.rmtree(work, ignore_errors=True)
        if own_client and client is not None:
            client.close()


def _headers(token: str) -> dict[str, str]:
    return {"X-Worker-Token": token}


def _upload_masked_views_to_api(
    client: httpx.Client,
    base: str,
    token: str,
    job_id: str,
    masked_job_dir: Path,
) -> None:
    """POST ``masked_*`` crops to API ``uploads/{job_id}/masked_views/`` when ``expose_masked_views``."""
    paths = sorted(masked_job_dir.glob("masked_*"))
    paths = [
        p
        for p in paths
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    ]
    if not paths:
        print(f"[polygraph-worker] job {job_id}: expose_masked_views: no masked_* files under {masked_job_dir}", flush=True)
        return
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    multipart: list[tuple[str, tuple[str, bytes, str]]] = []
    for p in paths:
        suf = p.suffix.lower()
        multipart.append(("files", (p.name, p.read_bytes(), mime.get(suf, "application/octet-stream"))))
    url = f"{base.rstrip('/')}/internal/worker/jobs/{job_id}/masked-views"
    r = client.post(url, headers=_headers(token), files=multipart, timeout=300.0)
    if r.status_code >= 400:
        print(
            f"[polygraph-worker] job {job_id}: masked-views POST failed HTTP {r.status_code}: {(r.text or '')[:240]}",
            flush=True,
        )
    else:
        print(f"[polygraph-worker] job {job_id}: uploaded {len(paths)} isolated-view file(s)", flush=True)


def _get_assignment(client: httpx.Client, base: str, token: str, job_id: str) -> httpx.Response:
    return client.get(
        f"{base.rstrip('/')}/internal/worker/jobs/{job_id}/assignment",
        headers=_headers(token),
        timeout=120.0,
    )


def _effective_poll_seconds(*, use_websocket: bool, configured_poll: float) -> float:
    """When WebSocket carries ``job_assigned``, slow down REST polling to reduce duplicate wakeups/logs.

    Set ``POLYGRAPH_POLL_WHILE_WEBSOCKET=1`` to keep polling at ``POLYGRAPH_POLL_SECONDS`` while WS is on.
    """
    if not use_websocket:
        return configured_poll
    if os.getenv("POLYGRAPH_POLL_WHILE_WEBSOCKET", "").strip().lower() in ("1", "true", "yes"):
        return configured_poll
    return max(float(configured_poll), 180.0)


def _poll_next(client: httpx.Client, base: str, token: str) -> httpx.Response:
    return client.get(
        f"{base.rstrip('/')}/internal/worker/next",
        headers=_headers(token),
        timeout=120.0,
    )


async def _consume_payloads(
    base: str,
    token: str,
    queue: asyncio.Queue,
    client: httpx.Client,
    websocket_may_connect: asyncio.Event,
    active_ws: _ActiveWorkerWebSocket,
) -> None:
    """Process jobs from the queue; always ``websocket_may_connect.set()`` when a run ends so WS can reconnect."""
    while True:
        payload = await queue.get()
        websocket_may_connect.clear()
        await active_ws.close_if_open()
        job_id = payload.get("job_id")
        try:
            print(f"[polygraph-worker] running job {job_id}", flush=True)
            await asyncio.to_thread(_run_one_job, base, token, payload, client)
            print(f"[polygraph-worker] finished job {job_id}", flush=True)
        except Exception as exc:
            print(f"[polygraph-worker] job error: {exc}", flush=True)
            if job_id:
                try:
                    await asyncio.to_thread(
                        lambda: client.post(
                            f"{base.rstrip('/')}/internal/worker/jobs/{job_id}/fail",
                            headers=_headers(token),
                            json={"error": str(exc)},
                            timeout=60.0,
                        )
                    )
                except Exception:
                    pass
        finally:
            websocket_may_connect.set()


def _websocket_invalid_status_code(exc: BaseException) -> int | None:
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if isinstance(code, int):
            return code
    return None


class _ActiveWorkerWebSocket:
    """Lets the job consumer close the idle WebSocket when any job starts (poll or WS assignment)."""

    def __init__(self) -> None:
        self._ws: Any = None

    def register(self, ws: Any) -> None:
        self._ws = ws

    def clear(self) -> None:
        self._ws = None

    async def close_if_open(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is None:
            return
        try:
            closed = getattr(ws, "closed", False)
        except Exception:
            closed = True
        if closed:
            return
        try:
            await ws.close()
        except Exception:
            pass


async def _ws_feed(
    base: str,
    token: str,
    queue: asyncio.Queue,
    client: httpx.Client,
    websocket_may_connect: asyncio.Event,
    active_ws: _ActiveWorkerWebSocket,
) -> None:
    """Subscribe for ``job_assigned`` only while idle. Close after each assignment; reconnect after the job ends.

    Long GPU work does not hold a WebSocket (avoids keepalive ping timeouts); progress uses REST.
    """
    import websockets

    try:
        from websockets.exceptions import InvalidStatus
    except ImportError:
        from websockets.exceptions import InvalidStatusCode as InvalidStatus  # type: ignore[misc,no-redef]

    while True:
        await websocket_may_connect.wait()

        candidates = iter_worker_websocket_uris(base, token)
        if not candidates:
            print("[polygraph-worker] WebSocket: no URI candidates; retrying in 5s", flush=True)
            await asyncio.sleep(5)
            continue

        for idx, uri in enumerate(candidates):
            print(
                f"[polygraph-worker] WebSocket trying {_mask_ws_uri(uri)} ({idx + 1}/{len(candidates)})",
                flush=True,
            )
            try:
                ping_interval = float(os.environ.get("POLYGRAPH_WS_PING_INTERVAL", "30"))
                ping_timeout = float(os.environ.get("POLYGRAPH_WS_PING_TIMEOUT", "600"))
                async with websockets.connect(
                    uri,
                    ping_interval=ping_interval,
                    ping_timeout=ping_timeout,
                ) as ws:
                    active_ws.register(ws)
                    print("[polygraph-worker] WebSocket connected (idle wait for assignments)", flush=True)
                    async for raw in ws:
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if data.get("type") != "job_assigned":
                            continue
                        jid = data.get("job_id")
                        if not jid:
                            continue
                        r = await asyncio.to_thread(_get_assignment, client, base, token, jid)
                        if r.status_code == 409:
                            continue
                        if r.status_code != 200:
                            body = (r.text or "")[:300]
                            print(f"[polygraph-worker] assignment {jid} -> {r.status_code} {body}", flush=True)
                            continue
                        await queue.put(r.json())
                        websocket_may_connect.clear()
                        print(
                            "[polygraph-worker] WebSocket: job queued; disconnecting until job finishes "
                            "(progress via REST /internal/worker/jobs/.../progress)",
                            flush=True,
                        )
                        break
                active_ws.clear()
                break
            except InvalidStatus as exc:
                code = _websocket_invalid_status_code(exc)
                if code == 404 and idx + 1 < len(candidates):
                    continue
                print(f"[polygraph-worker] WebSocket error: {exc}; retrying in 5s", flush=True)
                await asyncio.sleep(5)
                break
            except Exception as exc:
                print(f"[polygraph-worker] WebSocket error: {exc}; retrying in 5s", flush=True)
                await asyncio.sleep(5)
                break

        # No sleep here: after a job handoff we block on ``websocket_may_connect`` until work finishes.


async def _poll_feed(base: str, token: str, queue: asyncio.Queue, client: httpx.Client, interval: float) -> None:
    while True:
        try:
            r = await asyncio.to_thread(_poll_next, client, base, token)
            if r.status_code == 204:
                pass
            elif r.status_code != 200:
                body = (r.text or "")[:300]
                print(f"[polygraph-worker] poll /next -> {r.status_code} {body}", flush=True)
            else:
                await queue.put(r.json())
        except Exception as exc:
            print(f"[polygraph-worker] poll error: {exc}", flush=True)
        await asyncio.sleep(interval)


def main() -> None:
    import warnings

    for _m in ("weights_only", "torch.cuda.amp.autocast"):
        warnings.filterwarnings("ignore", message=_m, category=FutureWarning)

    base = os.environ.get("POLYGRAPH_API_BASE", "").strip().rstrip("/")
    token = os.environ.get("POLYGRAPH_WORKER_TOKEN", "").strip()
    if not base or not token:
        raise SystemExit("Set POLYGRAPH_API_BASE and POLYGRAPH_WORKER_TOKEN (see .env.worker.example)")

    poll_interval = float(os.environ.get("POLYGRAPH_POLL_SECONDS", "30"))
    use_ws = os.getenv("POLYGRAPH_USE_WEBSOCKET", "1").strip().lower() in ("1", "true", "yes")
    poll_eff = _effective_poll_seconds(use_websocket=use_ws, configured_poll=poll_interval)

    try:
        import torch

        print(f"[polygraph-worker] torch cuda_available={torch.cuda.is_available()}", flush=True)
    except ImportError:
        pass

    poll_note = ""
    if use_ws and poll_eff > poll_interval + 1e-6:
        poll_note = " (REST slowed while WS on; POLYGRAPH_POLL_WHILE_WEBSOCKET=1 to poll at POLYGRAPH_POLL_SECONDS)"
    print(
        f"[polygraph-worker] API {base} | ws={'on' if use_ws else 'off'} | fallback poll {poll_eff}s{poll_note}",
        flush=True,
    )

    asyncio.run(_run_async(base, token, poll_eff, use_ws))


async def _run_async(base: str, token: str, poll_interval: float, use_ws: bool) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    client = httpx.Client(timeout=600.0)
    # Cleared when WS hands off a job (socket closed); set again in consumer ``finally`` so WS reconnects idle.
    websocket_may_connect = asyncio.Event()
    websocket_may_connect.set()
    active_ws = _ActiveWorkerWebSocket()
    try:
        consumer = asyncio.create_task(
            _consume_payloads(base, token, queue, client, websocket_may_connect, active_ws),
        )
        tasks = [consumer]
        if use_ws:
            tasks.append(
                asyncio.create_task(_ws_feed(base, token, queue, client, websocket_may_connect, active_ws)),
            )
        tasks.append(asyncio.create_task(_poll_feed(base, token, queue, client, poll_interval)))
        await asyncio.gather(*tasks)
    finally:
        client.close()


if __name__ == "__main__":
    main()
