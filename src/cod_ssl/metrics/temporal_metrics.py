from __future__ import annotations

import math

import numpy as np


def normalized_centroid_displacement(previous: np.ndarray, current: np.ndarray) -> float | None:
    if previous.shape != current.shape or previous.ndim != 2:
        raise ValueError("motion masks must be aligned 2D arrays")
    previous_points, current_points = np.argwhere(previous > 0), np.argwhere(current > 0)
    if not len(previous_points) or not len(current_points): return None
    distance = np.linalg.norm(current_points.mean(0) - previous_points.mean(0))
    height, width = current.shape
    return float(distance / math.sqrt(height * height + width * width))


def raw_probability_flicker(previous: np.ndarray, current: np.ndarray) -> float:
    if previous.shape != current.shape or previous.ndim != 2:
        raise ValueError("probability maps must be aligned 2D arrays")
    return float(np.abs(current.astype(np.float32) - previous.astype(np.float32)).mean())
