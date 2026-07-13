"""Run local TripoSR (single image -> mesh) without Tripo cloud API key."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import open3d as o3d

from .meshing import export_glb


def _triposr_repo() -> Path:
    raw = (
        os.getenv("TRIPO_TRIPOSR_REPO", "").strip()
        or os.getenv("POLYGRAPH_OVERRIDE_TRIPOSR_REPO", "").strip()
    )
    if not raw:
        raise RuntimeError(
            "TripoSR is not configured. Set TRIPO_TRIPOSR_REPO on the API server "
            "or POLYGRAPH_OVERRIDE_TRIPOSR_REPO on the GPU worker."
        )
    repo = Path(raw).expanduser().resolve()
    if not repo.is_dir():
        raise RuntimeError(f"TripoSR repo path does not exist: {repo}")
    return repo


def _python_executable() -> Path:
    raw = (
        os.getenv("TRIPO_TRIPOSR_PYTHON", "").strip()
        or os.getenv("POLYGRAPH_OVERRIDE_TRIPOSR_PYTHON", "").strip()
    )
    if raw:
        p = Path(raw).expanduser()
        if p.is_file():
            return p
        found = shutil.which(raw)
        if found:
            return Path(found)
        raise RuntimeError(f"TripoSR python executable not found: {raw}")
    return Path(sys.executable)


def _entry_script(repo: Path) -> Path:
    raw = os.getenv("TRIPO_TRIPOSR_ENTRY_SCRIPT", "run.py").strip() or "run.py"
    entry = Path(raw)
    if not entry.is_absolute():
        entry = repo / entry
    if not entry.is_file():
        raise RuntimeError(f"TripoSR entry script not found: {entry}")
    return entry


def _timeout_seconds() -> int:
    raw = os.getenv("TRIPO_TRIPOSR_TIMEOUT_SECONDS", "900").strip()
    try:
        return max(60, int(raw))
    except ValueError:
        return 900


def _resolve_output_mesh(*, work_dir: Path, output_dir: Path, configured_output: Path) -> Path:
    candidates: list[Path] = [configured_output]
    for base in (work_dir, output_dir):
        if not base.is_dir():
            continue
        for name in ("mesh.glb", "mesh.obj", "mesh.ply", "output.glb", "output.obj", "output.ply"):
            candidates.append(base / name)
        for ext in ("*.glb", "*.obj", "*.ply"):
            candidates.extend(base.rglob(ext))

    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp in seen or not rp.is_file():
            continue
        seen.add(rp)
        ordered.append(rp)

    for p in ordered:
        if p.suffix.lower() == ".glb":
            return p
        if p.suffix.lower() in (".obj", ".ply"):
            out = work_dir / "triposr_converted.glb"
            mesh = o3d.io.read_triangle_mesh(str(p))
            if mesh is None or len(mesh.vertices) == 0:
                continue
            mesh.compute_vertex_normals()
            export_glb(mesh, out, compressed=False)
            return out

    raise RuntimeError(
        "TripoSR finished but no mesh (.glb/.obj/.ply) was found in the output directory."
    )


def run_triposr_local(*, image_path: Path, output_path: Path) -> str:
    """Run TripoSR subprocess and write a GLB to output_path."""
    repo = _triposr_repo()
    entry = _entry_script(repo)
    py = _python_executable()
    timeout_s = _timeout_seconds()

    work_dir = output_path.parent / "triposr_work"
    output_dir = work_dir / "out"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    args_template = (
        os.getenv("TRIPO_TRIPOSR_ARGS_TEMPLATE", "").strip()
        or "{input_image} --output-dir {output_dir}"
    )
    args = args_template.format(
        input_image=str(image_path.resolve()),
        output_dir=str(output_dir.resolve()),
        output_mesh=str(output_path.resolve()),
        repo_path=str(repo),
    )
    cmd = [str(py), str(entry)] + shlex.split(args, posix=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            cwd=str(repo),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"TripoSR timed out after {timeout_s}s") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-40:]
        raise RuntimeError("TripoSR failed.\n" + "\n".join(tail))

    resolved = _resolve_output_mesh(
        work_dir=work_dir,
        output_dir=output_dir,
        configured_output=output_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(resolved, output_path)
    if not output_path.is_file() or output_path.stat().st_size < 1024:
        raise RuntimeError(f"TripoSR output is too small or missing: {output_path}")
    return f"triposr-local-{int(time.time())}"
