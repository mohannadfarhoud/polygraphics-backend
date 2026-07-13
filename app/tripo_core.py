"""Tripo cloud API helpers (shared by worker script and API service)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int, *, min_value: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except Exception:
        return default
    return max(min_value, val)


def parse_optional_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def tripo_api_key() -> str:
    return os.getenv("TRIPO_API_KEY", "").strip() or os.getenv("AI_PRIOR_API_KEY", "").strip()


def remote_workers_enabled() -> bool:
    return os.getenv("APP_REMOTE_WORKERS", "").strip().lower() in ("1", "true", "yes")


def triposr_repo_configured() -> bool:
    return bool(
        os.getenv("TRIPO_TRIPOSR_REPO", "").strip()
        or os.getenv("POLYGRAPH_OVERRIDE_TRIPOSR_REPO", "").strip()
    )


def resolve_tripo_backend() -> str:
    """Choose generation backend: Tripo SDK, local TripoSR, GPU worker, or dev mock."""
    if tripo_api_key():
        return "tripo_sdk"
    if triposr_repo_configured():
        return "triposr_local"
    use_worker = env_bool("TRIPO_USE_WORKER", True)
    if use_worker and remote_workers_enabled():
        return "worker_triposr"
    if env_bool("TRIPO_DEV_MOCK", False):
        return "dev_mock"
    if remote_workers_enabled():
        return "worker_triposr"
    raise RuntimeError(
        "No Tripo backend available. Set TRIPO_API_KEY (Tripo SDK), "
        "TRIPO_TRIPOSR_REPO / POLYGRAPH_OVERRIDE_TRIPOSR_REPO (local TripoSR), "
        "or enable APP_REMOTE_WORKERS with a GPU worker running TripoSR."
    )


def should_use_dev_mock() -> bool:
    return resolve_tripo_backend() == "dev_mock"


def tripo_configured() -> bool:
    try:
        resolve_tripo_backend()
        return True
    except RuntimeError:
        return False


def build_image_to_model_kwargs(*, image: str) -> dict[str, Any]:
    model_version = os.getenv("TRIPO_MODEL_VERSION", "v3.1-20260211").strip() or "v3.1-20260211"
    texture_quality = os.getenv("TRIPO_TEXTURE_QUALITY", "standard").strip().lower() or "standard"
    if texture_quality not in ("standard", "detailed"):
        texture_quality = "standard"
    texture_alignment = os.getenv("TRIPO_TEXTURE_ALIGNMENT", "original_image").strip().lower() or "original_image"
    if texture_alignment not in ("original_image", "geometry"):
        texture_alignment = "original_image"
    orientation = os.getenv("TRIPO_ORIENTATION", "default").strip().lower() or "default"
    if orientation not in ("default", "align_image"):
        orientation = "default"

    request_kwargs: dict[str, Any] = {
        "image": image,
        "model_version": model_version,
        "texture": env_bool("TRIPO_TEXTURE", True),
        "pbr": env_bool("TRIPO_PBR", True),
        "texture_quality": texture_quality,
        "texture_alignment": texture_alignment,
        "auto_size": env_bool("TRIPO_AUTO_SIZE", False),
        "orientation": orientation,
        "quad": env_bool("TRIPO_QUAD", False),
        "compress": env_bool("TRIPO_COMPRESS", False),
        "generate_parts": env_bool("TRIPO_GENERATE_PARTS", False),
        "smart_low_poly": env_bool("TRIPO_SMART_LOW_POLY", False),
    }
    model_seed = parse_optional_int("TRIPO_MODEL_SEED")
    texture_seed = parse_optional_int("TRIPO_TEXTURE_SEED")
    face_limit = parse_optional_int("TRIPO_FACE_LIMIT")
    if model_seed is not None:
        request_kwargs["model_seed"] = model_seed
    if texture_seed is not None:
        request_kwargs["texture_seed"] = texture_seed
    if face_limit is not None and face_limit > 0:
        request_kwargs["face_limit"] = face_limit
    return request_kwargs


def task_status(task: Any) -> str:
    raw = getattr(task, "status", "")
    val = getattr(raw, "value", raw)
    return str(val).strip().lower()


def task_error(task: Any) -> str:
    for attr in ("error", "message", "reason"):
        v = getattr(task, attr, None)
        if v:
            return str(v)
    output = getattr(task, "output", None)
    if output is not None:
        for attr in ("error", "message", "reason"):
            v = getattr(output, attr, None)
            if v:
                return str(v)
    return "Unknown Tripo task failure."


def select_downloaded_model(downloaded: dict[str, str]) -> Path:
    for key in ("model", "pbr_model", "base_model"):
        val = downloaded.get(key)
        if val and Path(val).is_file():
            return Path(val)
    for val in downloaded.values():
        if val and Path(val).is_file():
            return Path(val)
    raise RuntimeError("Tripo SDK did not download any model file.")


async def run_image_to_model(*, image_path: Path, output_path: Path, backend: str) -> str:
    """Generate GLB using the selected backend."""
    if backend == "dev_mock":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = _minimal_glb_bytes()
        if len(data) < 256:
            data += b"\x00" * (256 - len(data))
        output_path.write_bytes(data)
        return "dev-mock-no-api-key"

    if backend == "triposr_local":
        from .tripo_triposr import run_triposr_local

        return run_triposr_local(image_path=image_path, output_path=output_path)

    if backend == "worker_triposr":
        raise RuntimeError("worker_triposr jobs are processed by the GPU worker, not on the API server.")

    api_key = tripo_api_key()
    if not api_key:
        raise RuntimeError("Missing TRIPO_API_KEY for Tripo SDK generation.")

    try:
        from tripo3d import TripoClient
    except Exception as exc:
        raise RuntimeError(
            "Python package 'tripo3d' is required. Install: pip install tripo3d"
        ) from exc

    timeout_s = env_int("TRIPO_POLL_TIMEOUT_SECONDS", 1800, min_value=60)
    verbose = env_bool("TRIPO_VERBOSE", False)
    request_kwargs = build_image_to_model_kwargs(image=str(image_path))
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    async with TripoClient(api_key=api_key) as client:
        task_id = await client.image_to_model(**request_kwargs)
        task = await client.wait_for_task(task_id, timeout=timeout_s, verbose=verbose)
        status = task_status(task)
        if status != "success":
            raise RuntimeError(f"Tripo task failed (status={status}): {task_error(task)}")
        downloaded = await client.download_task_models(task, str(output_dir))

    selected_model = select_downloaded_model(downloaded)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected_model, output_path)
    if not output_path.is_file() or output_path.stat().st_size < 1024:
        raise RuntimeError(f"Downloaded Tripo model is empty: {output_path}")
    return str(task_id)


def _minimal_glb_bytes() -> bytes:
    """Tiny valid GLB v2 for explicit dev mock only (empty scene)."""
    import json
    import struct

    json_chunk = json.dumps(
        {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": []}],
            "nodes": [],
        }
    ).encode("utf-8")
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
    bin_chunk = b"\x00\x00\x00\x00"
    body = b""
    body += struct.pack("<I", len(json_chunk))
    body += b"JSON"
    body += json_chunk
    body += struct.pack("<I", len(bin_chunk))
    body += b"BIN\x00"
    body += bin_chunk
    header = b"glTF" + struct.pack("<I", 2) + struct.pack("<I", 12 + len(body))
    return header + body
