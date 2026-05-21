from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import trimesh


def _convert_to_glb(src: Path, dst: Path) -> None:
    scene_or_mesh = trimesh.load(str(src), force="mesh")
    if scene_or_mesh is None:
        raise RuntimeError(f"Failed to load mesh file: {src}")
    mesh = scene_or_mesh
    if hasattr(mesh, "vertices") and len(mesh.vertices) == 0:
        raise RuntimeError(f"Mesh has zero vertices: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(dst))
    if not dst.is_file() or dst.stat().st_size < 256:
        raise RuntimeError(f"Converted GLB output is empty: {dst}")


def _run_upstream(
    *,
    upstream_cmd: str,
    input_manifest: Path,
    output_mesh: Path,
    output_dir: Path,
) -> Path:
    args_tmpl = os.getenv(
        "POLYGRAPH_AI_PRIOR_UPSTREAM_ARGS_TEMPLATE",
        "--input-manifest {input_manifest} --output {output_mesh}",
    ).strip()
    args = args_tmpl.format(
        input_manifest=str(input_manifest),
        output_mesh=str(output_mesh),
        output_dir=str(output_dir),
    )
    cmd = [upstream_cmd] + shlex.split(args, posix=False)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-40:]
        raise RuntimeError(
            "Upstream AI prior command failed.\n"
            f"Command: {' '.join(cmd)}\n"
            + "\n".join(tail)
        )

    # Preferred output path from contract.
    if output_mesh.is_file():
        return output_mesh

    # Fallback: detect common mesh names under output dir.
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
        p = output_dir / name
        if p.is_file():
            return p

    raise RuntimeError(
        "Upstream command succeeded but no mesh output was found. "
        "Ensure it writes --output path or one of ai_prior_mesh.* / mesh.* / output.*"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="PolyGraphics AI-prior command adapter")
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Final .glb path expected by polyGraphics")
    args = parser.parse_args()

    if not args.input_manifest.is_file():
        raise RuntimeError(f"Input manifest file not found: {args.input_manifest}")

    payload = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    images = payload.get("masked_images")
    if not isinstance(images, list) or not images:
        raise RuntimeError("Manifest must contain non-empty list field: masked_images")

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    upstream = os.getenv("POLYGRAPH_AI_PRIOR_UPSTREAM_CMD", "").strip()
    if not upstream:
        raise RuntimeError(
            "POLYGRAPH_AI_PRIOR_UPSTREAM_CMD is not set. "
            "Point it to your real AI prior executable/script."
        )

    produced = _run_upstream(
        upstream_cmd=upstream,
        input_manifest=args.input_manifest,
        output_mesh=args.output,
        output_dir=output_dir,
    )

    # Ensure final file is GLB for polyGraphics.
    if produced.suffix.lower() == ".glb":
        if produced.resolve() != args.output.resolve():
            args.output.write_bytes(produced.read_bytes())
    elif produced.suffix.lower() in (".obj", ".ply"):
        _convert_to_glb(produced, args.output)
    else:
        raise RuntimeError(f"Unsupported upstream mesh extension: {produced.suffix}")

    if not args.output.is_file() or args.output.stat().st_size < 256:
        raise RuntimeError(f"Final output GLB missing/empty: {args.output}")

    print(str(args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

