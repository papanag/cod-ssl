from __future__ import annotations

import platform

import torch


def runtime_info() -> dict[str, object]:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    vram = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": gpu, "vram_bytes": vram}

