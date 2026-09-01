from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch


def benchmark_cuda_aware(function: Callable[[], Any], *, warmup: int = 5, runs: int = 20) -> dict[str, float]:
    if warmup < 0 or runs < 1: raise ValueError("invalid benchmark iteration counts")
    for _ in range(warmup): function()
    if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(runs): function()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return {"mean_ms": 1000 * (time.perf_counter() - started) / runs,
            "warmup_runs": warmup, "timed_runs": runs,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0}
