import pandas as pd
import pytest

from cod_ssl.evaluation.statistics import (
    paired_regime_interaction,
    paired_video_bootstrap,
)
from cod_ssl.metrics.aggregation import aggregate_frame_and_video, metric_gain


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


def test_paired_bootstrap_is_reproducible_and_key_strict():
    first = paired_video_bootstrap(_videos([0.2, 0.3]), _videos([0.4, 0.2]), "S", resamples=100, seed=7)
    second = paired_video_bootstrap(_videos([0.2, 0.3]), _videos([0.4, 0.2]), "S", resamples=100, seed=7)
    assert first == second
    with pytest.raises(ValueError, match="identical"):
        paired_video_bootstrap(_videos([0.2, 0.3]), pd.DataFrame({"source_video_id": ["x"], "S": [1]}), "S")


def test_interaction_pairs_same_source_videos():
    result = paired_regime_interaction(_videos([0, 0]), _videos([1, 1]), _videos([0, 0]), _videos([2, 2]), "S", resamples=20)
    assert result["mean_interaction"] == 1.0
