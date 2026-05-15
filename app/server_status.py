from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import psutil

from . import jobs_db
from .pipeline_ready import assert_pipeline_ready
from .runtime_settings import RuntimeSettings


def _safe_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_importable(module_name: str) -> tuple[bool, str | None]:
    """Return (importable, error_message). Does a real ``import`` so namespace-packages work."""
    try:
        importlib.import_module(module_name)
        return True, None
    except Exception as exc:  # ImportError, but also misc init errors
        return False, str(exc)


def _path_info(p: str | None, *, must_be_file: bool = False, must_be_dir: bool = False) -> dict:
    info: dict = {"configured": bool(p), "value": p, "exists": False}
    if not p:
        return info
    path = Path(p)
    info["exists"] = path.exists()
    if must_be_file:
        info["is_file"] = path.is_file()
        if path.is_file():
            try:
                info["size_bytes"] = path.stat().st_size
            except OSError:
                pass
    if must_be_dir:
        info["is_dir"] = path.is_dir()
    return info


def _read_gpu_status() -> dict:
    status: dict = {
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "gpus": [],
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
    }

    try:
        import torch  # type: ignore

        status["torch_cuda_available"] = bool(torch.cuda.is_available())
        status["torch_cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            status["gpus"] = [{"name": n} for n in names]
    except Exception:
        pass

    if status["nvidia_smi_available"]:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                gpus = []
                for line in proc.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        gpus.append(
                            {
                                "name": parts[0],
                                "memory_total_mb": int(parts[1]),
                                "memory_free_mb": int(parts[2]),
                                "driver_version": parts[3],
                            }
                        )
                if gpus:
                    status["gpus"] = gpus
        except Exception:
            pass

    return status


def _pipeline_status(settings: RuntimeSettings | None) -> dict:
    sam_importable, sam_err = _module_importable("segment_anything")
    ma_importable, ma_err = _module_importable("mapanything")
    torch_importable, _torch_err = _module_importable("torch")

    sam = {
        "package": "segment-anything",
        "package_version": _safe_version("segment-anything"),
        "importable": sam_importable,
        "import_error": sam_err,
        "model_type": settings.sam_model_type if settings else None,
        "segmentation_mode": settings.sam_segmentation_mode if settings else None,
        "checkpoint": _path_info(
            settings.sam_checkpoint_path if settings else None,
            must_be_file=True,
        ),
    }

    mapanything = {
        "package": "mapanything (facebookresearch/map-anything)",
        "importable": ma_importable,
        "import_error": ma_err,
        "torch": {"installed": torch_importable, "version": _safe_version("torch")},
        "pretrained_model_id": settings.mapanything_pretrained_id if settings else None,
        "memory_efficient_inference": settings.mapanything_memory_efficient_inference if settings else None,
        "minibatch_size": settings.mapanything_minibatch_size if settings else None,
        "max_input_views": settings.mapanything_max_input_views if settings else None,
    }

    gs_repo = _path_info(
        settings.gs_repo_path if settings else None,
        must_be_dir=True,
    )
    gs_train_py = None
    if gs_repo.get("is_dir") and settings and settings.gs_repo_path:
        candidate = Path(settings.gs_repo_path) / "train.py"
        gs_train_py = {"path": str(candidate), "exists": candidate.is_file()}

    gs_cuda = False
    try:
        import torch

        gs_cuda = bool(torch.cuda.is_available())
    except Exception:
        pass

    gaussian_splatting = {
        "repo_path": gs_repo,
        "train_py": gs_train_py,
        "python_executable": settings.gs_python_executable if settings else None,
        "iterations": settings.gs_iterations if settings else None,
        "sh_degree": settings.gs_sh_degree if settings else None,
        "resolution": settings.gs_resolution if settings else None,
        "opacity_reset_interval": settings.gs_opacity_reset_interval if settings else None,
        "torch_cuda_available": gs_cuda,
        "allow_cpu_fallback": settings.gs_allow_cpu_fallback if settings else None,
        "cpu_max_points": settings.gs_cpu_max_points if settings else None,
    }

    ready_flag, ready_reason = (False, "settings unavailable")
    if settings is not None:
        try:
            assert_pipeline_ready(settings)
            ready_flag, ready_reason = True, None
        except Exception as exc:
            ready_flag, ready_reason = False, str(exc)

    return {
        "active_backend": settings.reconstruction_backend if settings else None,
        "device": settings.device if settings else None,
        "allow_placeholder_pipeline": settings.allow_placeholder_pipeline if settings else None,
        "ready": ready_flag,
        "ready_reason": ready_reason,
        "sam": sam,
        "mapanything": mapanything,
        "gaussian_splatting": gaussian_splatting,
    }


def _remote_workers_section() -> dict:
    """Split-deploy flags (GPU workers poll ``/internal/worker/next``). No token value exposed."""
    enabled = os.getenv("APP_REMOTE_WORKERS", "").strip().lower() in ("1", "true", "yes")
    pub = os.getenv("APP_PUBLIC_BASE_URL", "").strip()
    tok = bool(os.getenv("APP_WORKER_TOKEN", "").strip())
    return {
        "remote_workers_enabled": enabled,
        "worker_auth_configured": tok,
        "public_base_url": pub if pub else None,
        "note": (
            "GPU workers do not register or heartbeat on this endpoint; "
            "use jobs.count_by_status (queued / processing) as a backlog hint."
        ),
    }


def collect_server_status(
    root_dir: Path,
    *,
    settings: RuntimeSettings | None = None,
    db_path: Path | None = None,
) -> dict:
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(root_dir))
    boot_ts = psutil.boot_time()

    status: dict = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": shutil.which("python"),
        },
        "hardware": {
            "cpu": {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "usage_percent": psutil.cpu_percent(interval=0.2),
            },
            "memory": {
                "total_gb": round(vm.total / (1024**3), 2),
                "available_gb": round(vm.available / (1024**3), 2),
                "used_percent": vm.percent,
            },
            "disk": {
                "path": str(root_dir),
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "used_percent": disk.percent,
            },
            "gpu": _read_gpu_status(),
        },
        "runtime": {
            "uptime_seconds": int(max(0, time.time() - boot_ts)),
            "load_avg": _load_avg_windows_safe(),
        },
        "software": {
            "fastapi": _safe_version("fastapi"),
            "uvicorn": _safe_version("uvicorn"),
            "pydantic": _safe_version("pydantic"),
            "open3d": _safe_version("open3d"),
            "opencv-python": _safe_version("opencv-python"),
            "trimesh": _safe_version("trimesh"),
            "numpy": _safe_version("numpy"),
            "torch": _safe_version("torch"),
            "torchvision": _safe_version("torchvision"),
            "segment-anything": _safe_version("segment-anything"),
            "plyfile": _safe_version("plyfile"),
        },
        "pipeline": _pipeline_status(settings),
        "remote_workers": _remote_workers_section(),
    }

    if db_path is not None:
        try:
            status["jobs"] = {"count_by_status": jobs_db.count_jobs_by_status(db_path)}
        except Exception as exc:
            status["jobs"] = {"count_by_status": {}, "error": str(exc)}
    else:
        status["jobs"] = {"count_by_status": None, "note": "database path not passed"}

    return status


def _load_avg_windows_safe() -> list[float] | None:
    try:
        la = psutil.getloadavg()
        return [round(la[0], 2), round(la[1], 2), round(la[2], 2)]
    except Exception:
        return None

