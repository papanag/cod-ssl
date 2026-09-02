from __future__ import annotations

import pandas as pd

from cod_ssl.data.camotion_attributes import ATTRIBUTE_CODES

METRIC_COLUMNS = ("S", "E_adapt", "weightedF", "MAE", "E_mean", "E_max")


def aggregate_frame_and_video(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    missing = {"source_video_id", *METRIC_COLUMNS} - set(frame.columns)
    if missing:
        raise ValueError(f"per-frame results missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("cannot aggregate empty frame results")
    frame_weighted = {metric: float(frame[metric].mean()) for metric in METRIC_COLUMNS}
    benchmark_key = "video_id" if "video_id" in frame else "source_video_id"
    videos = frame.groupby(benchmark_key, sort=True)[list(METRIC_COLUMNS)].mean()
    video_weighted = {metric: float(videos[metric].mean()) for metric in METRIC_COLUMNS}
    return {"frame_weighted": frame_weighted, "video_weighted": video_weighted}


def per_video_table(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations = {metric: (metric, "mean") for metric in METRIC_COLUMNS}
    for optional in ("foreground_fraction", "motion_proxy", "inference_ms"):
        if optional in frame:
            aggregations[f"mean_{optional}"] = (optional, "mean")
    group_keys = ["run_id", "system_id", "dataset", "regime", "seed", "video_id", "source_video_id"]
    for code in ATTRIBUTE_CODES:
        column = f"attr_{code}"
        if column in frame:
            inconsistent = frame.groupby("video_id")[column].nunique(dropna=False).gt(1)
            if inconsistent.any():
                raise ValueError(f"sequence-level CAMotion attribute varies within video: {code}")
            aggregations[column] = (column, "first")
    if "attribute_scope" in frame:
        inconsistent_scope = frame.groupby("video_id")["attribute_scope"].nunique(dropna=False).gt(1)
        if inconsistent_scope.any():
            raise ValueError("attribute_scope varies within a source video")
        aggregations["attribute_scope"] = ("attribute_scope", "first")
    result = frame.groupby(
        group_keys,
        dropna=False, sort=True,
    ).agg(n_evaluated_frames=("frame_id", "size"), **aggregations).reset_index()
    return result


def metric_gain(left: float, right: float, metric: str) -> float:
    """Return right-system improvement; MAE direction is reversed."""
    return float(left - right if metric == "MAE" else right - left)
