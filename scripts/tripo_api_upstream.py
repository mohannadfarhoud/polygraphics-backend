from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, *, min_value: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except Exception:
        return default
    return max(min_value, val)


def _load_manifest(path: Path) -> list[Path]:
    if not path.is_file():
        raise RuntimeError(f"Input manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_images = payload.get("masked_images")
    if not isinstance(raw_images, list) or not raw_images:
        raise RuntimeError("Manifest must contain a non-empty list field named masked_images.")
    images = [Path(str(p)).resolve() for p in raw_images]
    existing = [p for p in images if p.is_file()]
    if not existing:
        raise RuntimeError("Manifest has masked_images but none of the files exist on disk.")
    return existing


def _foreground_pixels(path: Path) -> int:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        return 0
    if img.ndim == 2:
        return int((img > 8).sum())
    if img.shape[2] >= 4:
        alpha = img[:, :, 3]
        return int((alpha > 8).sum())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return int((gray > 8).sum())


def _pick_best_image(masked_images: list[Path]) -> Path:
    if len(masked_images) == 1:
        return masked_images[0]
    ranked = sorted(
        ((p, _foreground_pixels(p)) for p in masked_images),
        key=lambda x: x[1],
        reverse=True,
    )
    best, score = ranked[0]
    if score <= 0:
        return masked_images[0]
    return best


def _parse_optional_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _task_status(task: Any) -> str:
    raw = getattr(task, "status", "")
    val = getattr(raw, "value", raw)
    return str(val).strip().lower()


def _task_error(task: Any) -> str:
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


def _select_downloaded_model(downloaded: dict[str, str]) -> Path:
    for key in ("model", "pbr_model", "base_model"):
        val = downloaded.get(key)
        if val and Path(val).is_file():
            return Path(val)
    for val in downloaded.values():
        if val and Path(val).is_file():
            return Path(val)
    raise RuntimeError("Tripo SDK did not download any model file.")


async def _run(args: argparse.Namespace) -> None:
    try:
        from tripo3d import TripoClient
    except Exception as exc:
        raise RuntimeError(
            "Python package 'tripo3d' is required for Tripo API upstream. "
            "Install it in worker env: pip install tripo3d"
        ) from exc

    images = _load_manifest(args.input_manifest.resolve())
    selected = _pick_best_image(images)

    output_path = args.output.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("TRIPO_API_KEY", "").strip() or os.getenv("AI_PRIOR_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing TRIPO_API_KEY (or AI_PRIOR_API_KEY) in environment.")

    timeout_s = _env_int("TRIPO_POLL_TIMEOUT_SECONDS", 1800, min_value=60)
    verbose = _env_bool("TRIPO_VERBOSE", True)
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

    model_seed = _parse_optional_int("TRIPO_MODEL_SEED")
    texture_seed = _parse_optional_int("TRIPO_TEXTURE_SEED")
    face_limit = _parse_optional_int("TRIPO_FACE_LIMIT")

    request_kwargs: dict[str, Any] = {
        "image": str(selected),
        "model_version": model_version,
        "texture": _env_bool("TRIPO_TEXTURE", True),
        "pbr": _env_bool("TRIPO_PBR", True),
        "texture_quality": texture_quality,
        "texture_alignment": texture_alignment,
        "auto_size": _env_bool("TRIPO_AUTO_SIZE", False),
        "orientation": orientation,
        "quad": _env_bool("TRIPO_QUAD", False),
        "compress": _env_bool("TRIPO_COMPRESS", False),
        "generate_parts": _env_bool("TRIPO_GENERATE_PARTS", False),
        "smart_low_poly": _env_bool("TRIPO_SMART_LOW_POLY", False),
    }
    if model_seed is not None:
        request_kwargs["model_seed"] = model_seed
    if texture_seed is not None:
        request_kwargs["texture_seed"] = texture_seed
    if face_limit is not None and face_limit > 0:
        request_kwargs["face_limit"] = face_limit

    async with TripoClient(api_key=api_key) as client:
        task_id = await client.image_to_model(**request_kwargs)
        task = await client.wait_for_task(task_id, timeout=timeout_s, verbose=verbose)
        status = _task_status(task)
        if status != "success":
            raise RuntimeError(f"Tripo task failed (status={status}): {_task_error(task)}")
        downloaded = await client.download_task_models(task, str(output_dir))

    selected_model = _select_downloaded_model(downloaded)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected_model, output_path)
    if not output_path.is_file() or output_path.stat().st_size < 256:
        raise RuntimeError(f"Downloaded Tripo model is empty: {output_path}")

    report = {
        "provider": "tripo_api",
        "selected_input_image": str(selected),
        "task_id": str(task_id),
        "model_version": model_version,
        "request": request_kwargs,
        "downloaded_files": downloaded,
        "selected_model_file": str(selected_model),
        "final_output": str(output_path),
    }
    (output_dir / "tripo_api_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(output_path), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tripo API upstream adapter for PolyGraphics AI-prior command.")
    parser.add_argument("--input-manifest", required=True, type=Path, help="Manifest JSON with masked_images list.")
    parser.add_argument("--output", required=True, type=Path, help="Output mesh path expected by adapter.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for downloaded task files.")
    args = parser.parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
