#!/usr/bin/env python3
"""GPU/CPU worker CLI: run one isolation training job (local dataset dir or API poll)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import httpx

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.isolation_train import run_training_job  # noqa: E402


def _headers(token: str) -> dict[str, str]:
    return {"X-Worker-Token": token}


def _train_local(args: argparse.Namespace) -> int:
    metrics = run_training_job(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        base_model=args.base_model,
        epochs=args.epochs,
        val_split=args.val_split,
    )
    print(json.dumps(metrics, indent=2))
    return 0


def _train_via_api(args: argparse.Namespace) -> int:
    base = args.api_base.rstrip("/")
    token = args.worker_token
    with httpx.Client(timeout=600.0) as client:
        r = client.get(f"{base}/internal/worker/isolation/train/next", headers=_headers(token))
        if r.status_code == 204:
            print("No isolation train jobs queued")
            return 0
        r.raise_for_status()
        payload = r.json()
        job_id = payload["job_id"]
        model_id = payload["model_id"]
        zip_url = payload["dataset_zip_url"]
        if not zip_url.startswith("http"):
            zip_url = f"{base}{zip_url}"
        work = Path(tempfile.mkdtemp(prefix="iso-train-"))
        zpath = work / "dataset.zip"
        zr = client.get(zip_url)
        zr.raise_for_status()
        zpath.write_bytes(zr.content)
        extract = work / "dataset"
        extract.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(extract)
        out_dir = work / "output"
        metrics = run_training_job(
            dataset_dir=extract,
            output_dir=out_dir,
            base_model=str(payload.get("base_model") or args.base_model),
            epochs=int(payload.get("epochs") or args.epochs),
            val_split=float(payload.get("val_split") or args.val_split),
        )
        onnx = (out_dir / "model.onnx").read_bytes()
        up = client.post(
            f"{base}/internal/worker/isolation/train/{job_id}/complete",
            headers=_headers(token),
            data={"model_id": model_id, "metrics_json": json.dumps(metrics)},
            files={"file": ("model.onnx", onnx, "application/octet-stream")},
        )
        up.raise_for_status()
        print(f"Completed isolation train job {job_id} model {model_id}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolation training worker")
    parser.add_argument("--dataset-dir", type=Path, help="Local extracted dataset directory")
    parser.add_argument("--output-dir", type=Path, help="Write model.onnx here (local mode)")
    parser.add_argument("--api-base", default=os.getenv("POLYGRAPH_API_BASE", ""))
    parser.add_argument("--worker-token", default=os.getenv("POLYGRAPH_WORKER_TOKEN", ""))
    parser.add_argument("--base-model", default="isnet-general-use")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    if args.dataset_dir and args.output_dir:
        return _train_local(args)
    if args.api_base and args.worker_token:
        return _train_via_api(args)
    parser.error("Provide --dataset-dir + --output-dir OR --api-base + --worker-token")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
