from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from .meshing import export_glb
from .runtime_settings import RuntimeSettings


@dataclass
class AiPriorResult:
    mesh: o3d.geometry.TriangleMesh
    confidence: float
    provider: str
    details: dict


def _load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(f"AI prior provider returned an empty mesh: {path}")
    mesh.compute_vertex_normals()
    return mesh


def _convert_mesh_to_glb(src: Path, dst: Path) -> Path:
    mesh = _load_mesh(src)
    export_glb(mesh, dst, compressed=False)
    return dst


def _ensure_command_provider(settings: RuntimeSettings) -> Path:
    raw = (getattr(settings, "ai_prior_command", None) or "").strip()
    if not raw:
        raise RuntimeError(
            "AI prior backend selected but ai_prior_command is empty. "
            "Set runtime setting ai_prior_command to an executable/script that produces a mesh."
        )
    p = Path(raw)
    if p.is_file():
        return p
    found = shutil.which(raw)
    if found:
        return Path(found)
    raise RuntimeError(f"AI prior command not found: {raw}")


def _ensure_python_executable(raw_path: str | None) -> Path:
    raw = (raw_path or "").strip()
    if raw:
        p = Path(raw)
        if p.is_file():
            return p
        found = shutil.which(raw)
        if found:
            return Path(found)
        raise RuntimeError(f"Python executable not found: {raw}")
    return Path(sys.executable)


def _score_single_image_candidate(masked_path: Path, original_path: Path) -> float:
    """Combined quality score for selecting the best single-image AI input frame.

    Weights: 0.4 sharpness + 0.3 object_size + 0.2 center_alignment + 0.1 exposure.
    """
    try:
        import cv2
    except Exception:
        return 0.0

    msk = cv2.imread(str(masked_path), cv2.IMREAD_COLOR)
    if msk is None or msk.size == 0:
        return 0.0

    h, w = msk.shape[:2]
    fg = np.any(msk > 8, axis=2)
    fg_count = float(np.count_nonzero(fg))
    total = float(h * w)
    if fg_count == 0:
        return 0.0

    # Object size ratio (0..1)
    object_size = min(1.0, fg_count / total)

    # Center alignment: 1 = object centroid at image center, 0 = at corner
    ys, xs = np.where(fg)
    cx = float(xs.mean()) / max(1.0, float(w - 1))
    cy = float(ys.mean()) / max(1.0, float(h - 1))
    dist = float(np.hypot(cx - 0.5, cy - 0.5))
    center_alignment = max(0.0, 1.0 - dist / 0.5)

    # Sharpness + exposure from original
    src = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    if src is None or src.size == 0:
        sharpness = 0.0
        exposure_score = 0.5
    else:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = min(1.0, lap_var / 500.0)
        mean_luma = float(gray.mean())
        exposure_score = max(0.0, 1.0 - abs(mean_luma - 128.0) / 128.0)

    return (
        0.4 * sharpness
        + 0.3 * object_size
        + 0.2 * center_alignment
        + 0.1 * exposure_score
    )


def pick_best_single_image_pair(
    masked_images: list[Path],
    original_images: list[Path] | None,
) -> tuple[Path, Path]:
    """Return (best_masked, best_original) using combined quality score."""
    if not masked_images:
        raise RuntimeError("Single-image AI provider needs at least one masked image.")
    originals: list[Path] = (
        original_images
        if (original_images and len(original_images) == len(masked_images))
        else masked_images
    )
    best_masked = masked_images[0]
    best_original = originals[0]
    best_score = -1.0
    for masked, original in zip(masked_images, originals):
        score = _score_single_image_candidate(masked, original)
        if score > best_score:
            best_score = score
            best_masked = masked
            best_original = original
    return best_masked, best_original


def _write_manifest(masked_images: list[Path], manifest_path: Path) -> None:
    payload = {"masked_images": [str(p.resolve()) for p in masked_images]}
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_output_mesh(
    *,
    configured_output: Path,
    settings: RuntimeSettings,
    work_dir: Path,
    extra_search_dirs: list[Path] | None = None,
) -> Path:
    # 1) Explicit output from args template.
    candidates: list[Path] = [configured_output]
    # 2) Optional explicit fallback path from settings.
    alt = str(getattr(settings, "ai_prior_output_mesh_path", "")).strip()
    if alt:
        candidates.append(Path(alt))
    # 3) Common provider output names under work_dir.
    for name in (
        "ai_prior_mesh.glb",
        "ai_prior_mesh.obj",
        "ai_prior_mesh.ply",
        "mesh.glb",
        "mesh.obj",
        "mesh.ply",
        "output.glb",
        "output.obj",
        "output.ply",
    ):
        candidates.append(work_dir / name)
    for d in extra_search_dirs or []:
        for name in (
            "ai_prior_mesh.glb",
            "ai_prior_mesh.obj",
            "ai_prior_mesh.ply",
            "mesh.glb",
            "mesh.obj",
            "mesh.ply",
            "output.glb",
            "output.obj",
            "output.ply",
        ):
            candidates.append(d / name)

    for p in candidates:
        if p.is_file():
            suf = p.suffix.lower()
            if suf == ".glb":
                return p
            if suf in (".obj", ".ply"):
                out = work_dir / "ai_prior_mesh.glb"
                return _convert_mesh_to_glb(p, out)
            raise RuntimeError(
                f"AI prior output has unsupported extension: {p}. "
                "Expected .glb or convertible .obj/.ply."
            )
    scan_dirs = [work_dir, *(extra_search_dirs or [])]
    fallback_hits: list[Path] = []
    for d in scan_dirs:
        if not d.is_dir():
            continue
        for ext in ("*.glb", "*.obj", "*.ply"):
            fallback_hits.extend(d.rglob(ext))
    fallback_hits = [p for p in fallback_hits if p.is_file()]
    if fallback_hits:
        newest = sorted(fallback_hits, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        if newest.suffix.lower() == ".glb":
            return newest
        out = work_dir / "ai_prior_mesh.glb"
        return _convert_mesh_to_glb(newest, out)
    raise RuntimeError(
        "AI prior command succeeded but no mesh output was found. "
        "Expected --output path or ai_prior_output_mesh_path to point to .glb/.obj/.ply."
    )


def run_ai_prior_mesh(
    *,
    job_id: str,
    masked_images: list[Path],
    original_images: list[Path] | None = None,
    settings: RuntimeSettings,
    work_dir: Path,
    preferred_input_image: Path | None = None,
    provider_override: str | None = None,
) -> AiPriorResult:
    # provider_override is set by the auto-backend resolver in pipeline.py
    provider = (
        provider_override.strip().lower()
        if provider_override
        else str(getattr(settings, "ai_prior_provider", "command")).strip().lower()
    )
    confidence = float(getattr(settings, "ai_prior_default_confidence", 0.62))
    timeout_s = int(getattr(settings, "ai_prior_timeout_seconds", 600))
    work_dir.mkdir(parents=True, exist_ok=True)
    out_mesh = work_dir / "ai_prior_mesh.glb"

    if provider == "command":
        cmd_path = _ensure_command_provider(settings)
        args_template = (
            str(getattr(settings, "ai_prior_command_args_template", "")).strip()
            or "--input-manifest {input_manifest} --output {output_mesh}"
        )
        manifest = work_dir / "ai_prior_manifest.json"
        _write_manifest(masked_images, manifest)
        args = args_template.format(
            job_id=job_id,
            input_manifest=str(manifest),
            output_mesh=str(out_mesh),
            output_dir=str(work_dir),
        )
        cmd = [str(cmd_path)] + shlex.split(args, posix=False)
        env = os.environ.copy()
        key_env_name = str(getattr(settings, "ai_prior_api_key_env", "AI_PRIOR_API_KEY")).strip()
        key_required = bool(getattr(settings, "ai_prior_require_api_key", False))
        key_value = env.get(key_env_name, "").strip()
        if key_required and not key_value:
            raise RuntimeError(
                f"AI prior command requires API key but env var {key_env_name!r} is missing or empty."
            )
        if key_value:
            env["AI_PRIOR_API_KEY"] = key_value
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(30, timeout_s),
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"AI prior command timed out after {max(30, timeout_s)}s. "
                f"Command: {' '.join(cmd)}"
            ) from exc
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-30:]
            raise RuntimeError(
                "AI prior command failed.\n"
                f"Command: {' '.join(cmd)}\n"
                + "\n".join(tail)
            )
        out_mesh = _resolve_output_mesh(
            configured_output=out_mesh,
            settings=settings,
            work_dir=work_dir,
        )
        mesh = _load_mesh(out_mesh)
        return AiPriorResult(
            mesh=mesh,
            confidence=confidence,
            provider=provider,
            details={
                "output_mesh": str(out_mesh),
                "command": str(cmd_path),
                "command_args": args,
                "timeout_seconds": max(30, timeout_s),
                "api_key_env": key_env_name,
                "api_key_present": bool(key_value),
            },
        )

    if provider in ("triposr", "triposr_local"):
        raise RuntimeError(
            "ai_prior_provider=triposr_local is no longer supported. "
            "Use instantmesh_local for single-image jobs, or MapAnything/DUSt3R for 2+ images."
        )

    if provider == "mock":
        # Minimal local fallback for development/testing when a real AI model is unavailable.
        # It builds a mesh from the union of segmented silhouettes as a coarse prior.
        pts: list[np.ndarray] = []
        for i, p in enumerate(masked_images):
            img = o3d.io.read_image(str(p))
            arr = np.asarray(img)
            if arr.size == 0:
                continue
            if arr.ndim == 3:
                fg = np.any(arr > 8, axis=2)
            else:
                fg = arr > 8
            ys, xs = np.where(fg)
            if xs.size == 0:
                continue
            z = np.full(xs.shape, float(i) / max(1.0, float(len(masked_images) - 1)), dtype=np.float32)
            pts.append(np.column_stack([xs.astype(np.float32), ys.astype(np.float32), z]))
        if not pts:
            raise RuntimeError("AI prior mock provider could not find foreground points in masked images.")
        cloud = np.vstack(pts)
        cloud -= cloud.mean(axis=0, keepdims=True)
        scl = float(np.linalg.norm(cloud, axis=1).max())
        if scl > 1e-6:
            cloud /= scl
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.astype(np.float64))
        pcd.estimate_normals()
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)
        if len(mesh.vertices) == 0:
            raise RuntimeError("AI prior mock provider created an empty mesh.")
        mesh.compute_vertex_normals()
        return AiPriorResult(
            mesh=mesh,
            confidence=max(0.1, confidence - 0.25),
            provider=provider,
            details={"warning": "mock provider used"},
        )

    raise RuntimeError(
        f"Unsupported ai_prior_provider={provider!r}. "
        "Supported providers: command, instantmesh_local, mock."
    )

