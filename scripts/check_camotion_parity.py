#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from cod_ssl.evaluation.camotion_parity import compare_camotion_official_accumulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-root", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    gt_root, prediction_root = Path(args.ground_truth_root), Path(args.prediction_root)
    pairs = []
    for sequence in sorted(path for path in gt_root.iterdir() if path.is_dir()):
        for ground_truth in sorted((sequence / "GT").glob("*.png")):
            prediction = prediction_root / sequence.name / ground_truth.name
            if not prediction.is_file():
                raise FileNotFoundError(f"missing parity prediction: {prediction}")
            pairs.append((prediction, ground_truth))
    predictions, ground_truths = [], []
    for prediction, ground_truth in tqdm(pairs, desc="CAMotion parity", unit="target"):
        with Image.open(prediction) as image:
            predictions.append(np.asarray(image.convert("L"), dtype=np.uint8))
        with Image.open(ground_truth) as image:
            ground_truths.append(np.asarray(image.convert("L"), dtype=np.uint8))
    report = compare_camotion_official_accumulation(predictions, ground_truths)
    comparable = ("S", "weightedF", "MAE", "E_adapt", "E_mean")
    report["tolerance"] = args.tolerance
    report["parity_passed"] = all(
        report["absolute_difference"][metric] <= args.tolerance for metric in comparable
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["parity_passed"]:
        raise SystemExit("CAMotion official-style parity exceeded tolerance")


if __name__ == "__main__":
    main()
