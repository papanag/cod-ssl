from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _aligned(left: pd.DataFrame, right: pd.DataFrame, metric: str) -> pd.Series:
    key = "source_video_id"
    a = left.set_index(key)[metric].sort_index()
    b = right.set_index(key)[metric].sort_index()
    if not a.index.is_unique or not b.index.is_unique or not a.index.equals(b.index):
        raise ValueError("paired comparison requires identical unique source-video keys")
    return b - a


def paired_video_bootstrap(
    left: pd.DataFrame, right: pd.DataFrame, metric: str, *,
    resamples: int = 10_000, seed: int = 2_026_090_1,
) -> dict[str, Any]:
    differences = _aligned(left, right, metric)
    values = differences.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "metric": metric,
        "n_videos": len(values),
        "mean_difference": float(values.mean()),
        "median_difference": float(np.median(values)),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "fraction_right_favored": float((values < 0).mean() if metric == "MAE" else (values > 0).mean()),
        "fraction_left_favored": float((values > 0).mean() if metric == "MAE" else (values < 0).mean()),
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
    }


def paired_regime_interaction(
    small_dt: pd.DataFrame, small_vv: pd.DataFrame,
    large_dt: pd.DataFrame, large_vv: pd.DataFrame,
    metric: str, *, resamples: int = 10_000, seed: int = 2_026_090_1,
) -> dict[str, Any]:
    small = _aligned(small_dt, small_vv, metric)
    large = _aligned(large_dt, large_vv, metric)
    if not small.index.equals(large.index):
        raise ValueError("motion interaction requires identical source videos across regimes")
    values = (large - small).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return {
        "metric": metric, "n_source_videos": len(values),
        "mean_interaction": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "bootstrap_seed": seed, "bootstrap_resamples": resamples,
    }
