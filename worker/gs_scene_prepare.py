"""Subprocess entry: build GS workspace (MapAnything → COLMAP-text) then exit (releases CUDA before train.py).

Run via ``python -m worker.gs_scene_prepare`` from the repo root — see ``run_gaussian_splatting``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings-json", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--masked-json", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from app.cuda_memory import ensure_cuda_allocator_env
    from app.gaussian_splatting_runner import build_gaussian_scene_workspace
    from app.runtime_settings import RuntimeSettings

    ensure_cuda_allocator_env()

    raw = Path(args.settings_json).read_text(encoding="utf-8")
    settings = RuntimeSettings.model_validate_json(raw)
    masked = [Path(p) for p in json.loads(Path(args.masked_json).read_text(encoding="utf-8"))]
    work_dir = Path(args.work_dir)

    print("[gs_scene_prepare] building scene workspace (MapAnything)…", flush=True)
    build_gaussian_scene_workspace(
        masked,
        work_dir,
        settings,
        progress_callback=None,
    )
    print("[gs_scene_prepare] done — process exiting to free CUDA for train.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
