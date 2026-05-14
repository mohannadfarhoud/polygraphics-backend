"""Best-effort PyTorch CUDA cleanup so the worker process frees VRAM before heavy subprocesses (e.g. GS train.py)."""

from __future__ import annotations


def purge_torch_cuda() -> None:
    """Run GC and ``empty_cache`` / ``synchronize`` in a tight loop.

    Parent and child Python processes both consume the same physical GPU budget; releasing the
    allocator cache here reduces peaks when ``train.py`` starts right after DUSt3R/SAM.
    """
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
