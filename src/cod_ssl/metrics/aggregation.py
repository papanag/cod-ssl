from __future__ import annotations

import pandas as pd

METRIC_COLUMNS = ("S", "E_adapt", "weightedF", "MAE", "E_mean", "E_max")


def aggregate_frame_and_video(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    missing = {"source_video_id", *METRIC_COLUMNS} - set(frame.columns)
    if missing:
        raise ValueError(f"per-frame results missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("cannot aggregate empty frame results")
    frame_weighted = {metric: float(frame[metric].mean()) for metric in METRIC_COLUMNS}
    videos = frame.groupby("source_video_id", sort=True)[list(METRIC_COLUMNS)].mean()
    video_weighted = {metric: float(videos[metric].mean()) for metric in METRIC_COLUMNS}
    return {"frame_weighted": frame_weighted, "video_weighted": video_weighted}


def per_video_table(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations = {metric: (metric, "mean") for metric in METRIC_COLUMNS}
    for optional in ("foreground_fraction", "motion_proxy", "inference_ms"):
        if optional in frame:
            aggregations[f"mean_{optional}"] = (optional, "mean")
    result = frame.groupby(
        ["run_id", "system_id", "dataset", "regime", "seed", "video_id", "source_video_id"],
        dropna=False, sort=True,
    ).agg(n_evaluated_frames=("frame_id", "size"), **aggregations).reset_index()
    return result


def metric_gain(left: float, right: float, metric: str) -> float:
    """Return right-system improvement; MAE direction is reversed."""
    return float(left - right if metric == "MAE" else right - left)
