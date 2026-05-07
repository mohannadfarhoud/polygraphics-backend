from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import time
from pathlib import Path

import psutil


def _safe_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


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


def collect_server_status(root_dir: Path) -> dict:
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(root_dir))
    boot_ts = psutil.boot_time()

    return {
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
        },
    }


def _load_avg_windows_safe() -> list[float] | None:
    try:
        la = psutil.getloadavg()
        return [round(la[0], 2), round(la[1], 2), round(la[2], 2)]
    except Exception:
        return None

