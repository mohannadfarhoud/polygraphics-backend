"""Subprocess entry: run SAM masking for all inputs then exit (releases CUDA before DUSt3R/GS).

Invoked from ``ReconstructionPipeline._run_segmentation`` when ``gpu_isolate_phases`` is enabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--settings-json", required=True)
    parser.add_argument("--paths-json", required=True)
    parser.add_argument("--root-dir", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from app.cuda_memory import ensure_cuda_allocator_env

    ensure_cuda_allocator_env()

    from app.runtime_settings import RuntimeSettings
    from app.segmentation import SamSegmenter

    raw = Path(args.settings_json).read_text(encoding="utf-8")
    settings = RuntimeSettings.model_validate_json(raw)
    paths = [Path(p) for p in json.loads(Path(args.paths_json).read_text(encoding="utf-8"))]
    job_id = args.job_id
    root = Path(args.root_dir)

    masked_base = root / settings.masked_dir_name / job_id
    save_masks = bool(settings.save_raw_masks)
    masks_root = (root / settings.masks_dir_name / job_id) if save_masks else None

    print(f"[gpu_phase_sam] segmenting {len(paths)} images for job {job_id}…", flush=True)
    segmenter = SamSegmenter(settings)
    total = max(1, len(paths))
    for idx, image_path in enumerate(paths):
        suffix = image_path.suffix or ".png"
        output_path = masked_base / f"masked_{idx:03d}{suffix}"
        mask_path = (masks_root / f"mask_{idx:03d}.png") if masks_root is not None else None
        segmenter.segment_file(
            image_path,
            output_path,
            mask_output_path=mask_path,
        )
        print(f"[gpu_phase_sam] {idx + 1}/{total} {output_path.name}", flush=True)

    segmenter.release_gpu_memory()
    print("[gpu_phase_sam] done — exiting to release CUDA", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
