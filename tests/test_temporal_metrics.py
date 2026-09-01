import numpy as np

from cod_ssl.metrics.temporal_metrics import (
    normalized_centroid_displacement,
    raw_probability_flicker,
)


def test_centroid_displacement_is_normalized_and_empty_is_missing():
    left = np.zeros((10, 10)); right = np.zeros((10, 10)); left[2, 2] = 1; right[2, 4] = 1
    assert normalized_centroid_displacement(left, right) == 2 / np.sqrt(200)
    assert normalized_centroid_displacement(np.zeros((10, 10)), right) is None


def test_raw_flicker_does_not_minmax_normalize_frames():
    assert raw_probability_flicker(np.zeros((2, 2)), np.ones((2, 2)) * 0.25) == 0.25
