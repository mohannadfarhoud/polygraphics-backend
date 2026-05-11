"""Remote GPU worker: poll the public API for ``QUEUED`` jobs and run the pipeline locally.

Environment (see ``.env.worker.example``):

* ``POLYGRAPH_API_BASE`` — API root, e.g. ``https://agentmanager.easymediasuitecloud.com/polygraph``
* ``POLYGRAPH_WORKER_TOKEN`` — must match ``APP_WORKER_TOKEN`` on the API server
* ``POLYGRAPH_POLL_SECONDS`` — optional (default ``5``)

Optional path overrides (GPU machine paths often differ from the server):

* ``POLYGRAPH_OVERRIDE_SAM_CHECKPOINT``
* ``POLYGRAPH_OVERRIDE_DUST3R_CHECKPOINT``
* ``POLYGRAPH_OVERRIDE_COLMAP_PATH``
* ``POLYGRAPH_OVERRIDE_DUST3R_REPO``
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

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


def _run_one_job(
    base: str,
    token: str,
    payload: dict,
) -> None:
    job_id = payload["job_id"]
    settings_dict = _apply_local_overrides(payload["settings"])
    image_urls: list[str] = payload["image_urls"]

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    work = Path(tempfile.mkdtemp(prefix=f"polyjob-{job_id}-"))
    try:
        upload_dir = work / "uploads" / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=600.0) as client:
            for i, url in enumerate(image_urls):
                full = url if url.startswith("http") else f"{base.rstrip('/')}{url}"
                resp = client.get(full)
                resp.raise_for_status()
                suffix = Path(url).suffix or ".jpg"
                (upload_dir / f"input_{i:03d}{suffix}").write_bytes(resp.content)

        from app.config import PipelineConfig
        from app.interfaces import NoopJobRepository, NoopWebSocketNotifier
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
        pipe = ReconstructionPipeline(
            cfg,
            settings,
            segmenter=SamSegmenter(settings),
            reconstructor=Dust3RReconstructor(settings),
            job_repo=NoopJobRepository(),
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

        with httpx.Client(timeout=600.0) as client:
            r = client.post(
                f"{base}/internal/worker/jobs/{job_id}/complete",
                headers={"X-Worker-Token": token},
                data={"model_format": ext},
                files={"file": (f"{job_id}.{ext}", data, "application/octet-stream")},
            )
            r.raise_for_status()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    base = os.environ.get("POLYGRAPH_API_BASE", "").strip().rstrip("/")
    token = os.environ.get("POLYGRAPH_WORKER_TOKEN", "").strip()
    if not base or not token:
        raise SystemExit("Set POLYGRAPH_API_BASE and POLYGRAPH_WORKER_TOKEN (see .env.worker.example)")
    poll = float(os.environ.get("POLYGRAPH_POLL_SECONDS", "5"))

    print(
        f"[polygraph-worker] polling {base}/internal/worker/next every {poll}s "
        "(idle is silent; errors print below)",
        flush=True,
    )

    with httpx.Client(timeout=120.0) as client:
        while True:
            job_id: str | None = None
            try:
                r = client.get(f"{base}/internal/worker/next", headers={"X-Worker-Token": token})
                if r.status_code == 204:
                    time.sleep(poll)
                    continue
                r.raise_for_status()
                payload = r.json()
                job_id = payload.get("job_id")
                print(f"[polygraph-worker] claimed job {job_id}", flush=True)
                _run_one_job(base, token, payload)
            except httpx.HTTPStatusError as exc:
                body = (exc.response.text or "")[:400].replace("\n", " ")
                print(
                    f"[polygraph-worker] HTTP {exc.response.status_code} on next: {body}",
                    flush=True,
                )
                time.sleep(poll)
            except httpx.RequestError as exc:
                print(f"[polygraph-worker] network error: {exc}", flush=True)
                time.sleep(poll)
            except Exception as exc:
                print(f"[polygraph-worker] job error: {exc}", flush=True)
                if job_id:
                    try:
                        client.post(
                            f"{base}/internal/worker/jobs/{job_id}/fail",
                            headers={"X-Worker-Token": token},
                            json={"error": str(exc)},
                            timeout=60.0,
                        )
                    except Exception:
                        pass
                time.sleep(poll)


if __name__ == "__main__":
    main()
