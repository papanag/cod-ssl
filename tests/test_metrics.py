import numpy as np

from cod_ssl.metrics import CODMetrics


def evaluate_pair(prediction, ground_truth):
    metrics = CODMetrics()
    metrics.step(prediction, ground_truth)
    return metrics.compute()


def test_perfect_prediction_has_near_perfect_metrics():
    ground_truth = np.zeros((32, 32), dtype=np.uint8)
    ground_truth[8:24, 8:24] = 255
    results = evaluate_pair(ground_truth, ground_truth)
    assert results["s_measure"] > 0.99
    assert results["e_adaptive"] > 0.99
    assert results["weighted_f"] > 0.99
    assert results["mae"] < 1e-6


def test_inverted_prediction_performs_badly():
    ground_truth = np.zeros((32, 32), dtype=np.uint8)
    ground_truth[8:24, 8:24] = 255
    results = evaluate_pair(255 - ground_truth, ground_truth)
    assert results["s_measure"] < 0.1
    assert results["weighted_f"] < 0.1
    assert results["mae"] > 0.99

