import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _module():
    path = Path(__file__).parents[1] / "scripts" / "visualize_smoke_comparison.py"
    spec = importlib.util.spec_from_file_location("smoke_diagnostics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_binary_metrics_distinguish_exact_and_fuzzy_predictions():
    module = _module()
    ground_truth = np.array([[0, 0], [255, 255]], dtype=np.uint8)
    exact = module._binary_metrics(ground_truth, ground_truth)
    fuzzy = module._binary_metrics(np.full((2, 2), 128, dtype=np.uint8), ground_truth)
    assert exact["dice"] == 1.0
    assert exact["iou"] == 1.0
    assert exact["mae"] == 0.0
    assert exact["uncertain_fraction"] == 0.0
    assert fuzzy["uncertain_fraction"] == 1.0
    assert fuzzy["dice"] < exact["dice"]


def test_summaries_report_paired_direction_ci_and_win_rates():
    module = _module()
    frame = pd.DataFrame(
        {
            "dino_dice": [0.9, 0.8, 0.7], "vjepa_dice": [0.8, 0.7, 0.6],
            "dino_iou": [0.8, 0.7, 0.6], "vjepa_iou": [0.7, 0.6, 0.5],
            "dino_mae": [0.1, 0.2, 0.2], "vjepa_mae": [0.2, 0.3, 0.3],
            "dino_uncertain_fraction": [0.1, 0.1, 0.2],
            "vjepa_uncertain_fraction": [0.3, 0.2, 0.4],
        }
    )
    model, paired = module._summaries(frame)
    assert model.loc[model.model == "DINOv3", "paired_dice_win_rate"].iloc[0] == 1.0
    dice = paired[paired.metric == "dice"].iloc[0]
    mae = paired[paired.metric == "mae"].iloc[0]
    assert dice["better"] == "higher" and dice["dino_wins"] == 3
    assert mae["better"] == "lower" and mae["dino_wins"] == 3
    assert dice["bootstrap_95_ci_low"] > 0


def test_balanced_selection_includes_wins_and_hard_cases():
    module = _module()
    frame = pd.DataFrame(
        {
            "index": range(9),
            "dice_difference": [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
            "mean_dice": [0.8, 0.7, 0.6, 0.5, 0.1, 0.2, 0.3, 0.4, 0.9],
        }
    )
    selected = module._select_examples(frame, 6)
    assert len(selected) == 6
    assert {"DINOv3 win", "V-JEPA win", "shared hard case"}.issubset(
        set(selected.selection_reason)
    )
