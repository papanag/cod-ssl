from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cod_ssl.metrics import CODMetrics


def compare_camotion_official_accumulation(
    predictions: Sequence[np.ndarray], ground_truths: Sequence[np.ndarray]
) -> dict[str, Any]:
    """Compare per-frame averaging with CAMotion's global metric accumulation pattern."""
    if len(predictions) != len(ground_truths) or not predictions:
        raise ValueError("parity comparison requires equal non-empty prediction/GT sequences")
    accumulated = CODMetrics()
    individual = []
    for prediction, ground_truth in zip(predictions, ground_truths):
        one = CODMetrics()
        one.step(prediction, ground_truth)
        individual.append(one.compute())
        accumulated.step(prediction, ground_truth)
    official_style = accumulated.compute()
    mappings = {
        "S": "s_measure", "weightedF": "weighted_f", "MAE": "mae",
        "E_adapt": "e_adaptive", "E_mean": "e_mean", "E_max": "e_max",
    }
    per_frame = {
        output: float(np.mean([row[source] for row in individual]))
        for output, source in mappings.items()
    }
    accumulated_view = {output: official_style[source] for output, source in mappings.items()}
    return {
        "per_frame_mean": per_frame,
        "official_style_accumulated": accumulated_view,
        "absolute_difference": {
            key: abs(per_frame[key] - accumulated_view[key]) for key in mappings
        },
        "intentional_difference": (
            "E_max may differ because the study averages each target's maximum E-measure, "
            "whereas official accumulation takes the maximum after averaging threshold curves."
        ),
    }
