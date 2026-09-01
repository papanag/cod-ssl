import json
from pathlib import Path

import numpy as np
import yaml

from cod_ssl.metrics import CODMetrics


def test_static_metric_fixture_is_exact_regression():
    fixture = json.loads(Path("tests/fixtures/static_metric_reference.json").read_text())
    y, x = np.ogrid[:32, :32]
    ground_truth = (((x - 16) ** 2 + (y - 16) ** 2) <= 64).astype(np.uint8) * 255
    prediction = np.clip(255 - np.sqrt((x - 16) ** 2 + (y - 16) ** 2) * 20, 0, 255).astype(np.uint8)
    metrics = CODMetrics()
    metrics.step(prediction, ground_truth)
    actual = metrics.compute()
    for name, expected in fixture["metrics"].items():
        assert abs(actual[name] - expected) <= fixture["tolerance"]


def test_static_result_registry_contains_locked_matrix():
    registry = yaml.safe_load(Path("configs/static_reference_results.yaml").read_text())
    assert set(registry["results"]) == {"CAMO", "COD10K", "CHAMELEON", "NC4K"}
    assert all(set(systems) == {"dinov3_vitb16", "vjepa21_vitb16"}
               for systems in registry["results"].values())
    assert registry["prediction_pipeline"] == "sigmoid_resize_minmax_1e-8_uint8_round"
