"""Central GPU memory helpers for PyTorch on CUDA.

Strategy on consumer single-GPU boxes (~8 GB):

1. **Allocator**: Prefer ``expandable_segments`` (PyTorch 2.x) to reduce fragmentation — set via
   ``ensure_cuda_allocator_env()`` early (worker job start + subprocess entries).

2. **Process isolation**: SAM (phase 1) and MapAnything GS scene prep each run in **short-lived
   subprocesses** when ``gpu_isolate_phases`` is true so CUDA contexts tear down between peaks.

3. **Purge**: ``purge_torch_cuda()`` runs aggressive GC + ``empty_cache`` + ``synchronize`` between
   phases that stay in-process.

Environment:

* ``PYTORCH_CUDA_ALLOC_CONF`` — forwarded to PyTorch (default ``expandable_segments:True`` if unset).
* ``POLYGRAPH_GPU_ISOLATE_PHASES`` — optional ``0``/``false`` to disable subprocess isolation on the worker (debug).
"""

from __future__ import annotations

import os

DEFAULT_PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"


def ensure_cuda_allocator_env() -> None:
    """Set recommended PyTorch CUDA allocator env before CUDA-heavy imports or subprocess spawns."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", DEFAULT_PYTORCH_CUDA_ALLOC_CONF)


def purge_torch_cuda() -> None:
    """Aggressive GC + cache flush so in-process peaks drop before the next CUDA workload."""
    try:
        import gc

        for _ in range(3):
            gc.collect()
        import torch

        if not torch.cuda.is_available():
            return
        for _ in range(3):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
    except Exception:
        pass


def bootstrap_worker_cuda() -> None:
    """Start-of-job hook for remote GPU workers (allocator defaults + stale purge)."""
    ensure_cuda_allocator_env()
    purge_torch_cuda()


def effective_gpu_isolate_phases(settings: object) -> bool:
    """Whether to run SAM / GS scene prep in subprocesses (worker tuning + optional env override)."""
    raw = os.getenv("POLYGRAPH_GPU_ISOLATE_PHASES", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return bool(getattr(settings, "gpu_isolate_phases", True))
