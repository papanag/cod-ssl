from __future__ import annotations

from typing import Any

import numpy as np
from py_sod_metrics import Emeasure, MAE, Smeasure, WeightedFmeasure


class CODMetrics:
    """Established COD/SOD metrics with thesis-compatible headline fields."""

    def __init__(self) -> None:
        self.s_measure = Smeasure()
        self.weighted_f = WeightedFmeasure()
        self.e_measure = Emeasure()
        self.mae = MAE()
        self.count = 0

    def step(self, prediction: np.ndarray, ground_truth: np.ndarray) -> None:
        if prediction.shape != ground_truth.shape:
            raise ValueError(
                f"prediction and GT shapes differ: {prediction.shape} vs {ground_truth.shape}"
            )
        if prediction.ndim != 2:
            raise ValueError(f"metrics require 2D masks, got {prediction.shape}")
        prediction = np.asarray(prediction, dtype=np.uint8)
        ground_truth = np.asarray(ground_truth > 0, dtype=np.uint8) * 255
        self.s_measure.step(prediction, ground_truth)
        self.weighted_f.step(prediction, ground_truth)
        self.e_measure.step(prediction, ground_truth)
        self.mae.step(prediction, ground_truth)
        self.count += 1

    def compute(self) -> dict[str, Any]:
        if self.count == 0:
            raise RuntimeError("cannot compute COD metrics without samples")
        e_results = self.e_measure.get_results()["em"]
        e_curve = np.asarray(e_results["curve"], dtype=np.float64)
        return {
            "s_measure": float(self.s_measure.get_results()["sm"]),
            "e_adaptive": float(e_results["adp"]),
            "weighted_f": float(self.weighted_f.get_results()["wfm"]),
            "mae": float(self.mae.get_results()["mae"]),
            "e_mean": float(e_curve.mean()),
            "e_max": float(e_curve.max()),
            "num_images": self.count,
        }

