"""Remote GPU worker: WebSocket job notifications + REST assignment, progress, and upload.

Environment (see ``.env.worker.example``):

* ``POLYGRAPH_API_BASE`` — HTTP API root, e.g. ``https://host/polygraph``
* ``POLYGRAPH_WORKER_TOKEN`` — must match ``APP_WORKER_TOKEN`` on the API server
* ``POLYGRAPH_USE_WEBSOCKET`` — ``1``/``true`` to subscribe to ``wss://.../internal/worker/ws`` (default on)
* ``POLYGRAPH_POLL_SECONDS`` — fallback polling interval for ``GET /internal/worker/next`` (default ``30``)
* ``POLYGRAPH_PROGRESS_INTERVAL_SECONDS`` — min seconds between ``POST .../progress`` calls (default ``5``)
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


def _apply_local_overrides(settings_dict: dict) -> dict:
    mapping = (
        ("sam_checkpoint_path", "POLYGRAPH_OVERRIDE_SAM_CHECKPOINT"),
        ("dust3r_checkpoint_path", "POLYGRAPH_OVERRIDE_DUST3R_CHECKPOINT"),
        ("colmap_binary_path", "POLYGRAPH_OVERRIDE_COLMAP_PATH"),
        ("dust3r_repo_path", "POLYGRAPH_OVERRIDE_DUST3R_REPO"),
        ("gs_repo_path", "POLYGRAPH_OVERRIDE_GS_REPO"),
    )
    out = dict(settings_dict)
    for key, env in mapping:
        v = os.getenv(env, "").strip()
        if v:
            out[key] = v
    return out


def build_worker_websocket_uri(http_base: str, token: str) -> str:
    u = urlparse(http_base.strip().rstrip("/"))
    scheme = "wss" if u.scheme == "https" else "ws"
    path = u.path.rstrip("/") + "/internal/worker/ws"
    query = urlencode({"token": token})
    return urlunparse((scheme, u.netloc, path, "", query, ""))


class ApiReportingJobRepository:
    """Forwards pipeline ``PROCESSING`` stage/progress to the API (throttled)."""

    def __init__(self, client: httpx.Client, base: str, token: str, job_id: str) -> None:
        self._client = client
        self._base = base.rstrip("/")
        self._token = token
        self._job_id = job_id
        self._last_post = float("-inf")
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
        if now - self._last_post < self._interval:
            return
        self._last_post = now
        try:
            self._client.post(
                f"{self._base}/internal/worker/jobs/{job_id}/progress",
                headers={"X-Worker-Token": self._token},
                json={
                    "stage": stage or "processing",
                    "progress": int(progress) if progress is not None else 0,
                },
                timeout=60.0,
            )
        except Exception:
            pass


def _run_one_job(base: str, token: str, payload: dict, client: httpx.Client | None = None) -> None:
    job_id = payload["job_id"]
    settings_dict = _apply_local_overrides(payload["settings"])
    image_urls: list[str] = payload["image_urls"]

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=600.0)

    work = Path(tempfile.mkdtemp(prefix=f"polyjob-{job_id}-"))
    try:
        upload_dir = work / "uploads" / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        for i, url in enumerate(image_urls):
            full = url if url.startswith("http") else f"{base.rstrip('/')}{url}"
            resp = client.get(full)
            resp.raise_for_status()
            suffix = Path(url).suffix or ".jpg"
            (upload_dir / f"input_{i:03d}{suffix}").write_bytes(resp.content)

        from app.config import PipelineConfig
        from app.pipeline import ReconstructionPipeline
        from app.reconstruction import Dust3RReconstructor
        from app.runtime_settings import RuntimeSettings
        from app.segmentation import SamSegmenter

        settings = RuntimeSettings.model_validate(settings_dict)
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
        from app.interfaces import NoopWebSocketNotifier

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
        pipe.process_3d_job(job_id, paths)

        out_dir = work / settings.output_dir_name
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
        shutil.rmtree(work, ignore_errors=True)
        if own_client and client is not None:
            client.close()


def _headers(token: str) -> dict[str, str]:
    return {"X-Worker-Token": token}


def _get_assignment(client: httpx.Client, base: str, token: str, job_id: str) -> httpx.Response:
    return client.get(
        f"{base.rstrip('/')}/internal/worker/jobs/{job_id}/assignment",
        headers=_headers(token),
        timeout=120.0,
    )


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
) -> None:
    while True:
        payload = await queue.get()
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


async def _ws_feed(base: str, token: str, queue: asyncio.Queue, client: httpx.Client) -> None:
    import websockets

    uri = build_worker_websocket_uri(base, token)
    print(f"[polygraph-worker] WebSocket: {uri.split('token=')[0]}token=***", flush=True)
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=120) as ws:
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
        except Exception as exc:
            print(f"[polygraph-worker] WebSocket error: {exc}; reconnecting in 5s", flush=True)
            await asyncio.sleep(5)


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
    base = os.environ.get("POLYGRAPH_API_BASE", "").strip().rstrip("/")
    token = os.environ.get("POLYGRAPH_WORKER_TOKEN", "").strip()
    if not base or not token:
        raise SystemExit("Set POLYGRAPH_API_BASE and POLYGRAPH_WORKER_TOKEN (see .env.worker.example)")

    poll_interval = float(os.environ.get("POLYGRAPH_POLL_SECONDS", "30"))
    use_ws = os.getenv("POLYGRAPH_USE_WEBSOCKET", "1").strip().lower() in ("1", "true", "yes")

    print(
        f"[polygraph-worker] API {base} | ws={'on' if use_ws else 'off'} | fallback poll {poll_interval}s",
        flush=True,
    )

    asyncio.run(_run_async(base, token, poll_interval, use_ws))


async def _run_async(base: str, token: str, poll_interval: float, use_ws: bool) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    client = httpx.Client(timeout=600.0)
    try:
        consumer = asyncio.create_task(_consume_payloads(base, token, queue, client))
        tasks = [consumer]
        if use_ws:
            tasks.append(asyncio.create_task(_ws_feed(base, token, queue, client)))
        tasks.append(asyncio.create_task(_poll_feed(base, token, queue, client, poll_interval)))
        await asyncio.gather(*tasks)
    finally:
        client.close()


if __name__ == "__main__":
    main()
