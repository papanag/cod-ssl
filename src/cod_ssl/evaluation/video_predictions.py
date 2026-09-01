from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def logits_to_float_views(logits: torch.Tensor, output_size: tuple[int, int]) -> dict[str, np.ndarray]:
    if logits.ndim != 4 or logits.shape[:2] != (1, 1):
        raise ValueError("expected one [1,1,H,W] prediction")
    restored = F.interpolate(logits.float(), size=output_size, mode="bilinear", align_corners=False)
    raw = restored.sigmoid()[0, 0]
    minimum, maximum = raw.amin(), raw.amax()
    normalized = (raw - minimum) / (maximum - minimum + 1e-8)
    return {
        "sigmoid_raw": raw.cpu().numpy().astype(np.float32),
        "minmax": normalized.cpu().numpy().astype(np.float32),
        "minmax_uint8": normalized.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy(),
    }
