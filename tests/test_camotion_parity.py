import numpy as np

from cod_ssl.evaluation.camotion_parity import compare_camotion_official_accumulation


def test_official_accumulation_parity_on_fixed_prediction_fixture():
    ground_truth = np.zeros((8, 8), dtype=np.uint8)
    ground_truth[2:6, 2:6] = 255
    prediction = ground_truth.copy()
    report = compare_camotion_official_accumulation(
        [prediction, prediction], [ground_truth, ground_truth]
    )
    for metric, difference in report["absolute_difference"].items():
        assert difference < 1e-12, metric
