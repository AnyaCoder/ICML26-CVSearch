from __future__ import annotations

import gc
from typing import Any


def release_cuda_cache(device: Any | None = None) -> None:
    """Drop Python refs and release PyTorch CUDA cache for the requested device."""
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a runtime dependency here.
        return
    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.empty_cache()
        return
    parsed = torch.device(device)
    if parsed.type == "cuda":
        with torch.cuda.device(parsed):
            torch.cuda.empty_cache()
