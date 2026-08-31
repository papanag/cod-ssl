#!/usr/bin/env python3
"""Evaluate and visualize both smoke checkpoints on their full training subset."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import wilcoxon

from cod_ssl.backbones import build_backbone
from cod_ssl.data import CODDataset
from cod_ssl.engine.evaluate import logits_to_prediction
from cod_ssl.engine.train import select_amp
from cod_ssl.evaluation import save_qualitative_panel
from cod_ssl.metrics import CODMetrics
from cod_ssl.models import FrozenCODModel
from cod_ssl.utils.config import load_config

MODEL_SPECS = [
    ("DINOv3", "dino", "configs/frozen_dinov3_vitb16.yaml"),
    ("V-JEPA 2.1", "vjepa", "configs/frozen_vjepa21_vitb16.yaml"),
]


def _load_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _binary_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    probability = prediction.astype(np.float64) / 255.0
    pred, gt = probability >= 0.5, ground_truth > 0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return {
        "dice": float((2 * intersection + 1) / (pred.sum() + gt.sum() + 1)),
        "iou": float((intersection + 1) / (union + 1)),
        "mae": float(np.abs(probability - gt.astype(np.float64)).mean()),
        "uncertain_fraction": float(np.logical_and(probability > 0.25, probability < 0.75).mean()),
    }


def _predict_subset(
    config_path: str,
    checkpoint: Path,
    dataset: CODDataset,
    subset_size: int,
    prediction_dir: Path,
) -> dict[str, float | int | bool]:
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
    prediction_dir.mkdir(parents=True, exist_ok=True)
    cod_metrics = CODMetrics()
    for index in range(subset_size):
        sample = dataset[index]
        ground_truth = _load_mask(sample["mask_path"])
        image = sample["image"].unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=dtype, enabled=enabled
        ):
            logits = model(image)
        prediction = logits_to_prediction(logits, ground_truth.shape)
        Image.fromarray(prediction).save(prediction_dir / f"{index:04d}.png")
        cod_metrics.step(prediction, ground_truth)
    results = cod_metrics.compute()
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def _bootstrap_mean_ci(
    values: np.ndarray, *, seed: int = 42, iterations: int = 10_000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 500):
        count = min(500, iterations - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _summaries(per_image: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows = []
    dice_difference = per_image.dino_dice - per_image.vjepa_dice
    for label, prefix, _ in MODEL_SPECS:
        dice = per_image[f"{prefix}_dice"]
        wins = dice_difference > 0 if prefix == "dino" else dice_difference < 0
        model_rows.append(
            {
                "model": label,
                "n": len(per_image),
                "dice_mean": dice.mean(),
                "dice_sd": dice.std(ddof=1),
                "dice_median": dice.median(),
                "dice_q1": dice.quantile(0.25),
                "dice_q3": dice.quantile(0.75),
                "dice_min": dice.min(),
                "dice_max": dice.max(),
                "iou_mean": per_image[f"{prefix}_iou"].mean(),
                "mae_mean": per_image[f"{prefix}_mae"].mean(),
                "uncertain_fraction_mean": per_image[f"{prefix}_uncertain_fraction"].mean(),
                "paired_dice_win_rate": wins.mean(),
            }
        )
    model_summary = pd.DataFrame(model_rows)
    differences = {
        "dice": dice_difference,
        "iou": per_image.dino_iou - per_image.vjepa_iou,
        "mae": per_image.dino_mae - per_image.vjepa_mae,
        "uncertain_fraction": per_image.dino_uncertain_fraction - per_image.vjepa_uncertain_fraction,
    }
    paired_rows = []
    for metric, difference in differences.items():
        values = difference.to_numpy(dtype=np.float64)
        ci_low, ci_high = _bootstrap_mean_ci(values)
        nonzero = values[values != 0]
        p_value = float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
        sd = values.std(ddof=1)
        higher_is_better = metric in {"dice", "iou"}
        dino_wins = values > 0 if higher_is_better else values < 0
        vjepa_wins = values < 0 if higher_is_better else values > 0
        paired_rows.append(
            {
                "metric": metric,
                "better": "higher" if higher_is_better else "lower",
                "difference": "DINOv3 - V-JEPA",
                "mean_difference": values.mean(),
                "median_difference": np.median(values),
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
                "paired_effect_size_dz": values.mean() / sd if sd > 0 else 0.0,
                "wilcoxon_p": p_value,
                "dino_wins": int(dino_wins.sum()),
                "vjepa_wins": int(vjepa_wins.sum()),
                "ties": int((values == 0).sum()),
            }
        )
    return model_summary, pd.DataFrame(paired_rows)


def _plot_diagnostics(per_image: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes[0, 0].scatter(per_image.dino_dice, per_image.vjepa_dice, s=18, alpha=0.65)
    axes[0, 0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0, 0].set(xlabel="DINOv3 Dice", ylabel="V-JEPA Dice", title="Paired Dice (equality line)", xlim=(0, 1), ylim=(0, 1))
    difference = per_image.dino_dice - per_image.vjepa_dice
    axes[0, 1].hist(difference, bins=25, color="#5B8FF9", edgecolor="white")
    axes[0, 1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set(xlabel="Dice difference (DINOv3 − V-JEPA)", ylabel="Images", title="Paired differences")
    axes[1, 0].boxplot(
        [per_image.dino_dice, per_image.vjepa_dice],
        tick_labels=["DINOv3", "V-JEPA 2.1"], showmeans=True,
    )
    axes[1, 0].set(ylabel="Dice", title="Dice distributions", ylim=(0, 1))
    ranked = per_image.sort_values("dice_difference").reset_index(drop=True)
    axes[1, 1].plot(ranked.index, ranked.dice_difference, color="#E8684A")
    axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set(xlabel="Images ranked by difference", ylabel="DINOv3 − V-JEPA Dice", title="Ranked paired advantage")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("256-image smoke training-subset diagnostic (not held-out evidence)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _select_examples(per_image: pd.DataFrame, count: int) -> pd.DataFrame:
    if count < 6:
        raise ValueError("count must be at least six for balanced qualitative selection")
    per_group = count // 3
    groups = [
        per_image.nlargest(per_group, "dice_difference").assign(selection_reason="DINOv3 win"),
        per_image.nsmallest(per_group, "dice_difference").assign(selection_reason="V-JEPA win"),
        per_image.nsmallest(count - 2 * per_group, "mean_dice").assign(selection_reason="shared hard case"),
    ]
    selected = pd.concat(groups).drop_duplicates("index")
    if len(selected) < count:
        remaining = per_image[~per_image["index"].isin(selected["index"])].nsmallest(
            count - len(selected), "mean_dice"
        )
        selected = pd.concat([selected, remaining.assign(selection_reason="coverage fill")])
    return selected.head(count).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dino-run", required=True)
    parser.add_argument("--vjepa-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--training-subset", type=int, default=256)
    args = parser.parse_args()
    dataset = CODDataset(args.manifest, training=False)
    subset_size = min(args.training_subset, len(dataset))
    if subset_size < args.count:
        raise ValueError("training-subset must be at least as large as count")

    output = Path(args.output)
    predictions_root = output / "predictions"
    panels = output / "qualitative_panels"
    output.mkdir(parents=True, exist_ok=True)
    panels.mkdir(exist_ok=True)
    for stale_panel in panels.glob("*.png"):
        stale_panel.unlink()
    runs = [Path(args.dino_run), Path(args.vjepa_run)]
    aggregate_rows = []
    for (label, prefix, config), run in zip(MODEL_SPECS, runs):
        checkpoint = run / "checkpoints" / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing smoke checkpoint: {checkpoint}")
        results = _predict_subset(config, checkpoint, dataset, subset_size, predictions_root / prefix)
        aggregate_rows.append({"model": label, **results})

    rows = []
    for index in range(subset_size):
        source = dataset.rows.iloc[index]
        ground_truth = _load_mask(source.mask_path)
        row = {
            "index": index, "id": str(source.id), "source": str(source.source),
            "image_path": source.image_path, "mask_path": source.mask_path,
        }
        for _, prefix, _ in MODEL_SPECS:
            prediction = _load_mask(predictions_root / prefix / f"{index:04d}.png")
            row.update({f"{prefix}_{key}": value for key, value in _binary_metrics(prediction, ground_truth).items()})
        row["dice_difference"] = row["dino_dice"] - row["vjepa_dice"]
        row["mean_dice"] = (row["dino_dice"] + row["vjepa_dice"]) / 2
        rows.append(row)
    per_image = pd.DataFrame(rows)
    per_image.to_csv(output / "per_image_metrics.csv", index=False)
    model_summary, paired_summary = _summaries(per_image)
    aggregate = pd.DataFrame(aggregate_rows)
    for name, frame in (
        ("model_summary", model_summary),
        ("paired_comparison", paired_summary),
        ("aggregate_cod_metrics", aggregate),
    ):
        frame.to_csv(output / f"{name}.csv", index=False)
        (output / f"{name}.md").write_text(_markdown_table(frame))
        (output / f"{name}.tex").write_text(frame.to_latex(index=False, float_format="%.4f"))
    metadata = {
        "scope": "training-subset diagnostic; not held-out publication evidence",
        "num_images": subset_size,
        "prediction_postprocessing": "sigmoid, per-image min-max normalization, uint8 PNG",
        "binary_threshold": 0.5,
        "bootstrap_iterations": 10_000,
        "bootstrap_seed": 42,
    }
    (output / "diagnostic_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _plot_diagnostics(per_image, output / "smoke_diagnostic_plots.png")

    selected = _select_examples(per_image, args.count)
    selected.to_csv(output / "qualitative_selection.csv", index=False)
    for position, row in selected.iterrows():
        predictions = [
            _load_mask(predictions_root / prefix / f"{int(row['index']):04d}.png")
            for _, prefix, _ in MODEL_SPECS
        ]
        panel_row = pd.Series({"dataset": "smoke_train", **row.to_dict()})
        save_qualitative_panel(
            panel_row, predictions[0], predictions[1],
            panels / f"{position + 1:02d}__{row['id']}.png",
            [spec[0] for spec in MODEL_SPECS],
        )
    print(model_summary.to_string(index=False))
    print(f"Wrote the full {subset_size}-image diagnostic to {output}")


if __name__ == "__main__":
    main()
