#!/usr/bin/env python3
"""Create paired qualitative overlays from the two intermediate smoke checkpoints."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cod_ssl.backbones import build_backbone
from cod_ssl.data import CODDataset
from cod_ssl.engine.evaluate import logits_to_prediction
from cod_ssl.engine.train import select_amp
from cod_ssl.evaluation import save_qualitative_panel
from cod_ssl.models import FrozenCODModel
from cod_ssl.utils.config import load_config


def predict(
    config_path: str,
    checkpoint: Path,
    dataset: CODDataset,
    indices: list[int],
) -> list[np.ndarray]:
    config = load_config(config_path)
    model = FrozenCODModel(build_backbone(config["model"]["backbone"]["name"]))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.decoder.load_state_dict(state["decoder"], strict=True)
    model.assert_backbone_frozen()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enabled, dtype = select_amp(
        device, bool(config["training"]["amp"]), str(config["training"].get("amp_dtype", "auto"))
    )
    model = model.to(device).eval()
    predictions = []
    for index in indices:
        sample = dataset[index]
        with Image.open(sample["mask_path"]) as mask:
            output_size = (mask.height, mask.width)
        image = sample["image"].unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=dtype, enabled=enabled
        ):
            logits = model(image)
        predictions.append(logits_to_prediction(logits, output_size))
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions


def binary_dice(prediction: np.ndarray, mask_path: str) -> float:
    with Image.open(mask_path) as image:
        ground_truth = np.asarray(image.convert("L")) > 0
    foreground = prediction >= 128
    return float((2 * np.logical_and(foreground, ground_truth).sum() + 1) /
                 (foreground.sum() + ground_truth.sum() + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dino-run", required=True)
    parser.add_argument("--vjepa-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--training-subset", type=int, default=256)
    args = parser.parse_args()
    if args.count < 1 or args.training_subset < args.count:
        raise ValueError("count must be positive and no larger than training-subset")

    dataset = CODDataset(args.manifest, training=False)
    subset_size = min(args.training_subset, len(dataset))
    indices = np.linspace(0, subset_size - 1, args.count, dtype=int).tolist()
    runs = [Path(args.dino_run), Path(args.vjepa_run)]
    checkpoints = [run / "checkpoints" / "last.pt" for run in runs]
    if not all(path.is_file() for path in checkpoints):
        raise FileNotFoundError(f"missing smoke checkpoint: {checkpoints}")
    predictions = [
        predict(config, checkpoint, dataset, indices)
        for config, checkpoint in zip(
            ["configs/frozen_dinov3_vitb16.yaml", "configs/frozen_vjepa21_vitb16.yaml"],
            checkpoints,
        )
    ]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    selection = []
    for position, index in enumerate(indices):
        row = dataset.rows.iloc[index]
        scores = [binary_dice(model_predictions[position], row.mask_path) for model_predictions in predictions]
        panel_row = pd.Series(
            {"dataset": "smoke_train", "id": str(row.id), "image_path": row.image_path,
             "mask_path": row.mask_path, "dino_dice": scores[0], "vjepa_dice": scores[1]}
        )
        target = output / f"{position + 1:02d}__{row.id}.png"
        save_qualitative_panel(
            panel_row, predictions[0][position], predictions[1][position], target,
            ["DINOv3", "V-JEPA 2.1"],
        )
        selection.append({"id": row.id, "image_path": row.image_path, "mask_path": row.mask_path,
                          "dino_dice": scores[0], "vjepa_dice": scores[1], "panel": str(target)})
    pd.DataFrame(selection).to_csv(output / "smoke_visual_scores.csv", index=False)
    print(f"Wrote {len(selection)} paired smoke panels to {output}")


if __name__ == "__main__":
    main()
