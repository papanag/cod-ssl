import pandas as pd
import pytest

from cod_ssl.evaluation.statistics import (
    paired_system_attribute_summary,
    paired_video_bootstrap,
    temporal_sampling_summary,
)
from cod_ssl.metrics.aggregation import aggregate_frame_and_video, metric_gain, per_video_table


def _videos(values):
    return pd.DataFrame({"source_video_id": ["a", "b"], "S": values, "MAE": values})


def test_video_weighting_does_not_weight_long_video_more():
    rows = pd.DataFrame({
        "source_video_id": ["a", "a", "a", "b"],
        "S": [1, 1, 1, 0], "E_adapt": [1, 1, 1, 0], "weightedF": [1, 1, 1, 0],
        "MAE": [0, 0, 0, 1], "E_mean": [1, 1, 1, 0], "E_max": [1, 1, 1, 0],
    })
    result = aggregate_frame_and_video(rows)
    assert result["frame_weighted"]["S"] == 0.75
    assert result["video_weighted"]["S"] == 0.5
    assert metric_gain(0.3, 0.2, "MAE") == pytest.approx(0.1)


def test_video_weighting_uses_benchmark_subsegments_not_shared_source_identity():
    rows = pd.DataFrame({
        "video_id": ["snow.1", "snow.1", "snow.2"],
        "source_video_id": ["snow", "snow", "snow"],
        "S": [1.0, 1.0, 0.0], "E_adapt": [1.0, 1.0, 0.0],
        "weightedF": [1.0, 1.0, 0.0], "MAE": [0.0, 0.0, 1.0],
        "E_mean": [1.0, 1.0, 0.0], "E_max": [1.0, 1.0, 0.0],
    })
    assert aggregate_frame_and_video(rows)["video_weighted"]["S"] == 0.5


def test_paired_bootstrap_is_reproducible_and_key_strict():
    first = paired_video_bootstrap(_videos([0.2, 0.3]), _videos([0.4, 0.2]), "S", resamples=100, seed=7)
    second = paired_video_bootstrap(_videos([0.2, 0.3]), _videos([0.4, 0.2]), "S", resamples=100, seed=7)
    assert first == second
    with pytest.raises(ValueError, match="identical"):
        paired_video_bootstrap(_videos([0.2, 0.3]), pd.DataFrame({"source_video_id": ["x"], "S": [1]}), "S")


def test_paired_bootstrap_reverses_mae_so_positive_means_right_improves():
    result = paired_video_bootstrap(
        _videos([0.4, 0.3]), _videos([0.2, 0.1]), "MAE", resamples=20
    )
    assert result["mean_difference"] == pytest.approx(0.2)
    assert result["fraction_right_favored"] == 1.0


def _system_tables():
    ids = ["short", "long", "other"]
    attributes = [True, True, False]
    targets = [1, 10, 2]
    values = {
        "DS": [0.2, 0.4, 0.3], "VI": [0.3, 0.5, 0.4],
        "DT": [0.4, 0.5, 0.4], "VV": [0.7, 0.6, 0.5],
    }
    return {
        system: pd.DataFrame({
            "source_video_id": ids, "S": scores,
            "MAE": [1 - value for value in scores],
            "n_evaluated_frames": targets, "attr_OC": attributes,
        })
        for system, scores in values.items()
    }


def test_attribute_summary_filters_paired_videos_and_computes_gain_advantage():
    result = paired_system_attribute_summary(
        _system_tables(), "S", attribute="OC", resamples=50, seed=3, minimum_videos=3
    )
    assert result["n_videos"] == 2 and result["n_targets"] == 11
    assert result["inferential_status"] == "insufficient_subset"
    assert result["contrasts"]["gain_advantage"]["estimate"] == pytest.approx(0.1)
    repeat = paired_system_attribute_summary(
        _system_tables(), "S", attribute="OC", resamples=50, seed=3, minimum_videos=3
    )
    assert result == repeat


def test_attribute_summary_reverses_mae_gain_direction():
    result = paired_system_attribute_summary(
        _system_tables(), "MAE", attribute="OC", resamples=20
    )
    assert result["contrasts"]["VV_minus_DT"]["estimate"] > 0


def test_temporal_sampling_summary_uses_paired_sequences_and_mae_direction():
    tables = {
        "DT_D1": pd.DataFrame({"video_id": ["a", "b"], "S": [0.6, 0.7], "MAE": [0.2, 0.3]}),
        "DT_S5": pd.DataFrame({"video_id": ["a", "b"], "S": [0.5, 0.5], "MAE": [0.3, 0.4]}),
        "VV_D1": pd.DataFrame({"video_id": ["a", "b"], "S": [0.8, 0.9], "MAE": [0.1, 0.2]}),
        "VV_S5": pd.DataFrame({"video_id": ["a", "b"], "S": [0.5, 0.6], "MAE": [0.3, 0.3]}),
    }
    result = temporal_sampling_summary(tables, "S", resamples=50, seed=4)
    assert result["C_D"]["estimate"] == pytest.approx(0.15)
    assert result["C_V"]["estimate"] == pytest.approx(0.3)
    assert result["Delta_C"]["estimate"] == pytest.approx(0.15)
    mae = temporal_sampling_summary(tables, "MAE", resamples=50, seed=4)
    assert mae["C_D"]["estimate"] > 0 and mae["C_V"]["estimate"] > 0
    changed = {name: table.copy() for name, table in tables.items()}
    changed["VV_S5"] = changed["VV_S5"].iloc[:1]
    with pytest.raises(ValueError, match="identical"):
        temporal_sampling_summary(changed, "S")


def test_per_video_table_rejects_frame_varying_sequence_attributes():
    rows = []
    for frame_id, attribute in (("a", True), ("b", False)):
        rows.append({
            "run_id": "r", "system_id": "DS", "dataset": "camotion", "regime": "default",
            "seed": 1, "video_id": "v", "source_video_id": "v", "frame_id": frame_id,
            "attr_OC": attribute, "attribute_scope": "sequence", "foreground_fraction": 0.2,
            **{metric: 0.5 for metric in ("S", "E_adapt", "weightedF", "MAE", "E_mean", "E_max")},
        })
    with pytest.raises(ValueError, match="varies"):
        per_video_table(pd.DataFrame(rows))
