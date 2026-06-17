"""InstantMesh local provider for single-image 3D reconstruction.

Runs the InstantMesh repo's entry script as a subprocess so it operates
in its own Python environment with its own dependencies.
"""

from __future__ import annotations

import logging
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

_log = logging.getLogger(__name__)


@dataclass
class InstantMeshResult:
    mesh: o3d.geometry.TriangleMesh
    confidence: float
    output_mesh: Path
    selected_frame: Path
    debug_dir: Path | None
    details: dict


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(f"InstantMesh returned an empty mesh: {path}")
    mesh.compute_vertex_normals()
    return mesh


def _convert_to_glb(src: Path, dst: Path) -> Path:
    mesh = _load_mesh(src)
    export_glb(mesh, dst, compressed=False)
    return dst


def _ensure_python(raw_path: str | None) -> Path:
    raw = (raw_path or "").strip()
    if raw:
        p = Path(raw)
        if p.is_file():
            return p
        found = shutil.which(raw)
        if found:
            return Path(found)
        raise RuntimeError(f"InstantMesh Python executable not found: {raw}")
    return Path(sys.executable)


def _find_output_mesh(work_dir: Path, output_dir: Path) -> Path | None:
    """Search common output locations for a valid mesh file."""
    search_dirs = [output_dir, work_dir]
    candidates_names = (
        "mesh.glb", "mesh.obj", "mesh.ply",
        "output.glb", "output.obj", "output.ply",
        "instantmesh_output.glb", "instantmesh_output.obj",
    )
    for d in search_dirs:
        for name in candidates_names:
            p = d / name
            if p.is_file():
                return p
        # Fallback: pick newest mesh file anywhere under the directory
        hits = []
        for ext in ("*.glb", "*.obj", "*.ply"):
            hits.extend(d.rglob(ext))
        hits = [p for p in hits if p.is_file()]
        if hits:
            return max(hits, key=lambda p: p.stat().st_mtime)
    return None


def _normalise_to_glb(src: Path, work_dir: Path) -> Path:
    """Return src if already .glb, otherwise convert to .glb."""
    if src.suffix.lower() == ".glb":
        return src
    out = work_dir / "instantmesh_output.glb"
    return _convert_to_glb(src, out)


def _prepare_clean_input(
    *,
    original_path: Path,
    masked_path: Path | None,
    output_dir: Path,
    debug_dir: Path | None,
    target_size: int = 512,
    margin_ratio: float = 0.15,
) -> Path:
    """Background-remove, bbox-crop, center on square canvas, resize to target_size.

    Reuses the same logic as the single-image AI input preparer.
    """
    try:
        import cv2
    except Exception:
        return original_path

    src = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    msk = cv2.imread(str(masked_path), cv2.IMREAD_COLOR) if masked_path else None

    if src is None or src.size == 0:
        return original_path

    h, w = src.shape[:2]

    # --- Debug: save selected frame ---
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            cv2.imwrite(str(debug_dir / "selected_frame.jpg"), src, [cv2.IMWRITE_JPEG_QUALITY, 90])
        except Exception:
            pass

    # --- Background removal ---
    if msk is not None and msk.size > 0:
        if msk.shape[:2] != (h, w):
            msk = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)
        if debug_dir is not None:
            try:
                cv2.imwrite(str(debug_dir / "segmented_frame.png"), msk)
            except Exception:
                pass
        fg_mask = np.any(msk > 8, axis=2)
    else:
        # No mask: use entire frame as foreground
        fg_mask = np.ones((h, w), dtype=bool)

    clean = src.copy()
    clean[~fg_mask] = [255, 255, 255]

    # --- Bounding box from mask ---
    ys, xs = np.where(fg_mask)
    if len(ys) == 0:
        return original_path

    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    obj_w, obj_h = x1 - x0 + 1, y1 - y0 + 1

    mx = max(4, int(obj_w * margin_ratio))
    my = max(4, int(obj_h * margin_ratio))
    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(w - 1, x1 + mx)
    y1 = min(h - 1, y1 + my)

    crop = clean[y0:y1 + 1, x0:x1 + 1]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return original_path

    # --- Square canvas ---
    ch, cw = crop.shape[:2]
    canvas_side = max(ch, cw)
    canvas = np.full((canvas_side, canvas_side, 3), 255, dtype=np.uint8)
    off_y = (canvas_side - ch) // 2
    off_x = (canvas_side - cw) // 2
    canvas[off_y:off_y + ch, off_x:off_x + cw] = crop

    result = cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_AREA)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "instantmesh_input.png"
    ok = cv2.imwrite(str(out_path), result)

    if not ok:
        return original_path

    if debug_dir is not None:
        try:
            cv2.imwrite(str(debug_dir / "instantmesh_input.png"), result)
        except Exception:
            pass

    return out_path


# ---------------------------------------------------------------------------
# Post-processing (optional mesh cleanup)
# ---------------------------------------------------------------------------

def _post_process_mesh(
    mesh: o3d.geometry.TriangleMesh,
    *,
    remove_floaters: bool = True,
    decimate_target: int | None = None,
) -> o3d.geometry.TriangleMesh:
    """Light cleanup: remove isolated components, optional decimation."""
    if remove_floaters:
        try:
            from .meshing import keep_largest_mesh_component
            mesh = keep_largest_mesh_component(mesh)
        except Exception as exc:
            _log.warning("floater removal failed: %s", exc)

    if decimate_target and len(mesh.triangles) > decimate_target:
        try:
            from .meshing import decimate
            mesh = decimate(mesh, decimate_target)
        except Exception as exc:
            _log.warning("decimation failed: %s", exc)

    mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_instantmesh(
    *,
    job_id: str,
    selected_frame: Path,
    masked_frame: Path | None,
    settings: RuntimeSettings,
    work_dir: Path,
) -> InstantMeshResult:
    """Run InstantMesh on the selected (pre-processed) frame and return the result mesh.

    Parameters
    ----------
    job_id:          Job identifier (used for logging and workspace paths).
    selected_frame:  Best original frame to reconstruct from.
    masked_frame:    Corresponding SAM-segmented frame (for background removal); optional.
    settings:        Runtime settings carrying repo/executable paths.
    work_dir:        Per-job working directory (under data/ai_prior_workspace/{job_id}).
    """
    repo_raw = str(getattr(settings, "ai_prior_instantmesh_repo_path", "") or "").strip()
    if not repo_raw:
        raise RuntimeError(
            "ai_prior_provider='instantmesh_local' requires ai_prior_instantmesh_repo_path "
            "to point to a local InstantMesh repository on the worker."
        )
    repo = Path(repo_raw)
    if not repo.is_dir():
        raise RuntimeError(f"InstantMesh repo path does not exist: {repo}")

    entry_raw = (
        str(getattr(settings, "ai_prior_instantmesh_entry_script", "") or "").strip()
        or "run.py"
    )
    entry = Path(entry_raw) if Path(entry_raw).is_absolute() else repo / entry_raw
    if not entry.is_file():
        raise RuntimeError(f"InstantMesh entry script not found: {entry}")

    py = _ensure_python(getattr(settings, "ai_prior_instantmesh_python_executable", None))
    timeout_s = int(getattr(settings, "ai_prior_timeout_seconds", 600))
    confidence = float(getattr(settings, "ai_prior_default_confidence", 0.70))

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = work_dir / "instantmesh_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = work_dir / "instantmesh_inputs"
    debug_dir = work_dir / "instantmesh_debug"

    # Build clean input image
    input_image = _prepare_clean_input(
        original_path=selected_frame,
        masked_path=masked_frame,
        output_dir=inputs_dir,
        debug_dir=debug_dir,
    )

    args_template = (
        str(getattr(settings, "ai_prior_instantmesh_args_template", "") or "").strip()
        or "--input {input_image} --output-dir {output_dir}"
    )
    args = args_template.format(
        job_id=job_id,
        input_image=str(input_image),
        output_dir=str(output_dir),
        repo_path=str(repo),
    )
    cmd = [str(py), str(entry)] + shlex.split(args, posix=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

    _log.info("job=%s: running InstantMesh: %s", job_id, " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(60, timeout_s),
            check=False,
            cwd=str(repo),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"InstantMesh timed out after {max(60, timeout_s)}s for job {job_id}."
        ) from exc

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-40:]
        raise RuntimeError(
            "InstantMesh command failed.\n"
            f"Command: {' '.join(cmd)}\n"
            + "\n".join(tail)
        )

    # Locate output mesh
    raw_mesh = _find_output_mesh(work_dir, output_dir)
    if raw_mesh is None:
        raise RuntimeError(
            "InstantMesh succeeded but no mesh output found. "
            "Expected .glb/.obj/.ply under the output directory."
        )

    final_glb = _normalise_to_glb(raw_mesh, work_dir)
    mesh = _load_mesh(final_glb)

    # Optional post-processing
    post_process = bool(getattr(settings, "instantmesh_post_process", True))
    decimate_target_raw = getattr(settings, "instantmesh_decimate_target", None)
    decimate_target = int(decimate_target_raw) if decimate_target_raw else None
    if post_process:
        mesh = _post_process_mesh(mesh, remove_floaters=True, decimate_target=decimate_target)
        # Re-export post-processed mesh
        processed_path = work_dir / "instantmesh_processed.glb"
        export_glb(mesh, processed_path, compressed=False)
        final_glb = processed_path

    # Save debug: final model symlink/copy
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(final_glb, debug_dir / "final_model.glb")
        except Exception:
            pass

    _log.info("job=%s: InstantMesh completed → %s", job_id, final_glb.name)
    return InstantMeshResult(
        mesh=mesh,
        confidence=confidence,
        output_mesh=final_glb,
        selected_frame=selected_frame,
        debug_dir=debug_dir,
        details={
            "output_mesh": str(final_glb),
            "instantmesh_repo_path": str(repo),
            "instantmesh_entry_script": str(entry),
            "instantmesh_python": str(py),
            "instantmesh_input_image": str(input_image),
            "selected_frame": str(selected_frame),
            "masked_frame": str(masked_frame) if masked_frame else None,
            "debug_dir": str(debug_dir),
            "command_args": args,
            "timeout_seconds": max(60, timeout_s),
            "post_processed": post_process,
        },
    )
