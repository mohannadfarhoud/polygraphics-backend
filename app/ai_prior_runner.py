from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

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


def _write_manifest(masked_images: list[Path], manifest_path: Path) -> None:
    payload = {"masked_images": [str(p.resolve()) for p in masked_images]}
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_ai_prior_mesh(
    *,
    job_id: str,
    masked_images: list[Path],
    settings: RuntimeSettings,
    work_dir: Path,
) -> AiPriorResult:
    provider = str(getattr(settings, "ai_prior_provider", "command")).strip().lower()
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
        cmd = [str(cmd_path)] + [x for x in args.split(" ") if x]
        env = os.environ.copy()
        key_env_name = str(getattr(settings, "ai_prior_api_key_env", "AI_PRIOR_API_KEY")).strip()
        key_value = env.get(key_env_name, "").strip()
        if key_value:
            env["AI_PRIOR_API_KEY"] = key_value
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(30, timeout_s),
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-30:]
            raise RuntimeError(
                "AI prior command failed.\n"
                f"Command: {' '.join(cmd)}\n"
                + "\n".join(tail)
            )
        if not out_mesh.is_file():
            alt = str(getattr(settings, "ai_prior_output_mesh_path", "")).strip()
            if alt:
                out_mesh = Path(alt)
        if not out_mesh.is_file():
            raise RuntimeError(
                "AI prior command succeeded but no output mesh found. "
                "Provide ai_prior_output_mesh_path or ensure command writes --output file."
            )
        mesh = _load_mesh(out_mesh)
        return AiPriorResult(
            mesh=mesh,
            confidence=confidence,
            provider=provider,
            details={"output_mesh": str(out_mesh), "command": str(cmd_path)},
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
        "Supported providers: command, mock."
    )

