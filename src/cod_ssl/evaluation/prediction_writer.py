from __future__ import annotations

from pathlib import Path

import numpy as np


class PredictionWriter:
    def __init__(self, root: str | Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def write(self, key: str, *, logits: np.ndarray, sigmoid_raw: np.ndarray, minmax: np.ndarray) -> Path:
        safe = key.replace("/", "__").replace("..", "_")
        target = self.root / f"{safe}.npz"
        np.savez_compressed(target, logits=logits.astype(np.float32),
                            sigmoid_raw=sigmoid_raw.astype(np.float32), minmax=minmax.astype(np.float32))
        return target
