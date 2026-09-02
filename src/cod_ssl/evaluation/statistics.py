from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _aligned(left: pd.DataFrame, right: pd.DataFrame, metric: str) -> pd.Series:
    key = "video_id" if "video_id" in left and left["video_id"].is_unique else "source_video_id"
    a = left.set_index(key)[metric].sort_index()
    b = right.set_index(key)[metric].sort_index()
    if not a.index.is_unique or not b.index.is_unique or not a.index.equals(b.index):
        raise ValueError("paired comparison requires identical unique source-video keys")
    return a - b if metric == "MAE" else b - a


def temporal_sampling_summary(
    tables: dict[str, pd.DataFrame], metric: str, *,
    resamples: int = 10_000, seed: int = 2_026_090_1,
) -> dict[str, Any]:
    """Paired D1-versus-S5 contrasts using identical benchmark-sequence draws."""
    required = {"DT_D1", "DT_S5", "VV_D1", "VV_S5"}
    if set(tables) != required:
        raise ValueError(f"temporal sampling summary requires {sorted(required)}")
    indexed = {}
    for name, table in tables.items():
        key = "video_id" if "video_id" in table else "source_video_id"
        values = table.set_index(key)[metric].sort_index()
        if not values.index.is_unique:
            raise ValueError(f"duplicate benchmark-sequence rows for {name}")
        indexed[name] = values
    reference = indexed["DT_D1"].index
    if any(not values.index.equals(reference) for values in indexed.values()):
        raise ValueError("temporal sampling runs require identical benchmark-sequence keys")
    direction = -1.0 if metric == "MAE" else 1.0
    c_d = direction * (indexed["DT_D1"].to_numpy() - indexed["DT_S5"].to_numpy())
    c_v = direction * (indexed["VV_D1"].to_numpy() - indexed["VV_S5"].to_numpy())
    delta = c_v - c_d
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(reference), size=(resamples, len(reference)))

    def estimate(values: np.ndarray) -> dict[str, Any]:
        means = values[draws].mean(axis=1)
        return {"estimate": float(values.mean()),
                "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])]}

    return {
        "metric": metric, "n_sequences": len(reference),
        "C_D": estimate(c_d), "C_V": estimate(c_v), "Delta_C": estimate(delta),
        "bootstrap_seed": seed, "bootstrap_resamples": resamples,
        "interpretation": "D1 versus S5 changes sampling density and source-frame coverage at fixed T",
    }


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
        "fraction_right_favored": float((values > 0).mean()),
        "fraction_left_favored": float((values < 0).mean()),
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
    }


def paired_system_attribute_summary(
    tables: dict[str, pd.DataFrame],
    metric: str,
    *,
    attribute: str | None = None,
    resamples: int = 10_000,
    seed: int = 2_026_090_1,
    minimum_videos: int = 5,
) -> dict[str, Any]:
    """Summarize four systems with one paired sequence draw for every contrast."""
    required = {"DS", "VI", "DT", "VV"}
    if set(tables) != required:
        raise ValueError(f"attribute summary requires systems {sorted(required)}")
    indexed = {}
    for system, table in tables.items():
        key = "video_id" if "video_id" in table else "source_video_id"
        if table[key].duplicated().any():
            raise ValueError(f"duplicate benchmark-sequence rows for {system}")
        indexed[system] = table.set_index(key).sort_index()
    reference = indexed["DS"].index
    if any(not table.index.equals(reference) for table in indexed.values()):
        raise ValueError("attribute summary requires identical paired source-video keys")
    selected = reference
    if attribute is not None:
        column = f"attr_{attribute}"
        if any(column not in table for table in indexed.values()):
            raise ValueError(f"missing explicit CAMotion attribute column: {column}")
        vectors = [table[column].astype(bool) for table in indexed.values()]
        if any(not vector.equals(vectors[0]) for vector in vectors[1:]):
            raise ValueError(f"CAMotion attribute vectors differ across systems: {attribute}")
        selected = reference[vectors[0].to_numpy()]
    if len(selected) == 0:
        raise ValueError(f"CAMotion subset has no videos: {attribute or 'All'}")
    values = {
        system: indexed[system].loc[selected, metric].to_numpy(dtype=np.float64)
        for system in sorted(required)
    }
    direction = -1.0 if metric == "MAE" else 1.0
    contrasts = {
        "VI_minus_DS": direction * (values["VI"] - values["DS"]),
        "VV_minus_DT": direction * (values["VV"] - values["DT"]),
        "dino_gain": direction * (values["DT"] - values["DS"]),
        "vjepa_gain": direction * (values["VV"] - values["VI"]),
    }
    contrasts["gain_advantage"] = contrasts["vjepa_gain"] - contrasts["dino_gain"]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(selected), size=(resamples, len(selected)))
    estimates = {}
    for name, vector in contrasts.items():
        means = vector[draws].mean(axis=1)
        estimates[name] = {
            "estimate": float(vector.mean()),
            "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        }
    return {
        "subset": attribute or "All", "metric": metric,
        "n_videos": len(selected),
        "n_targets": int(indexed["DS"].loc[selected, "n_evaluated_frames"].sum()),
        "inferential_status": "ok" if len(selected) >= minimum_videos else "insufficient_subset",
        "system_means": {system: float(vector.mean()) for system, vector in values.items()},
        "contrasts": estimates,
        "bootstrap_seed": seed, "bootstrap_resamples": resamples,
        "attribute_scope": "sequence" if attribute else None,
    }
