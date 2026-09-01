from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageFilter
from tqdm.auto import tqdm

DATASET_ORDER = ["camo_test", "cod10k_test", "chameleon", "nc4k"]
METRICS = ["s_measure", "e_adaptive", "weighted_f", "mae"]


def _load_metrics(run: Path) -> dict[str, dict[str, Any]]:
    path = run / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"evaluate the run first; missing {path}")
    metrics = json.loads(path.read_text())
    missing = set(DATASET_ORDER) - set(metrics)
    if missing:
        raise ValueError(f"{run} is missing evaluation datasets: {sorted(missing)}")
    return metrics


def _model_name(run: Path) -> str:
    config = yaml.safe_load((run / "config.yaml").read_text())
    return str(config["model"]["backbone"]["name"])


def _manifest_path(run: Path, dataset: str) -> Path:
    config = yaml.safe_load((run / "config.yaml").read_text())
    path = Path(config["evaluation"]["manifests"][dataset])
    if path.is_file():
        return path
    candidate = Path.cwd() / path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"manifest not found for {dataset}: {path}")


def _mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    pred, gt = prediction >= 128, ground_truth > 0
    return float((2 * np.logical_and(pred, gt).sum() + 1) / (pred.sum() + gt.sum() + 1))


def _overlay(image: Image.Image, prediction: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    probability = prediction.astype(np.float32) / 255.0
    color = np.zeros_like(base)
    color[..., 0] = 255
    alpha = 0.55 * probability[..., None]
    return Image.fromarray(np.uint8(np.clip(base * (1 - alpha) + color * alpha, 0, 255)))


def _ground_truth_overlay(image: Image.Image, ground_truth: np.ndarray) -> Image.Image:
    result = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    edges = np.asarray(Image.fromarray(ground_truth).filter(ImageFilter.FIND_EDGES)) > 0
    result[edges] = (0, 255, 80)
    return Image.fromarray(result)


def _plot_metric_comparison(table: pd.DataFrame, output: Path) -> None:
    names = list(table["backbone"].drop_duplicates())
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    x = np.arange(len(DATASET_ORDER)); width = 0.36
    for axis, metric in zip(axes.flat, METRICS):
        for index, name in enumerate(names):
            values = [
                table[(table.backbone == name) & (table.dataset == dataset)][metric].iloc[0]
                for dataset in DATASET_ORDER
            ]
            axis.bar(x + (index - 0.5) * width, values, width, label=name)
        axis.set_title(metric.replace("_", " ").title() + (" ↓" if metric == "mae" else " ↑"))
        axis.set_xticks(x, [name.replace("_test", "").upper() for name in DATASET_ORDER], rotation=15)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle("Frozen-backbone COD comparison")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _plot_training_curves(runs: list[Path], names: list[str], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for run, name in zip(runs, names):
        log = pd.read_csv(run / "training_log.csv")
        axis.plot(log.epoch, log.loss, marker="o", markersize=3, label=name)
    axis.set(xlabel="Epoch", ylabel="Training loss", title="Phase-1 training curves")
    axis.grid(alpha=0.3); axis.legend()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def save_qualitative_panel(
    row: pd.Series,
    dino_prediction: np.ndarray,
    vjepa_prediction: np.ndarray,
    output: Path,
    names: list[str],
) -> None:
    with Image.open(row.image_path) as raw_image:
        image = raw_image.convert("RGB")
    gt = _mask(Path(row.mask_path))
    if image.size != (gt.shape[1], gt.shape[0]):
        image = image.resize((gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR)
    panels = [
        (image, "Original"),
        (_ground_truth_overlay(image, gt), "Ground truth (green boundary)"),
        (Image.fromarray(dino_prediction), f"{names[0]} probability"),
        (_overlay(image, dino_prediction), f"{names[0]} overlay"),
        (Image.fromarray(vjepa_prediction), f"{names[1]} probability"),
        (_overlay(image, vjepa_prediction), f"{names[1]} overlay"),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(9, 10), constrained_layout=True)
    for axis, (panel, title) in zip(axes.flat, panels):
        axis.imshow(panel, cmap="gray" if panel.mode == "L" else None, vmin=0, vmax=255)
        axis.set_title(title, fontsize=10, pad=3)
        axis.axis("off")
    figure.suptitle(
        f"{row.dataset}/{row.id} — Dice: {names[0]}={row.dino_dice:.3f}, "
        f"{names[1]}={row.vjepa_dice:.3f}",
        fontsize=12,
    )
    figure.savefig(output, dpi=140)
    plt.close(figure)


def compare_runs(
    dino_run: str | Path,
    vjepa_run: str | Path,
    output_dir: str | Path,
    *,
    qualitative_count: int = 24,
    labels: list[str] | None = None,
) -> Path:
    runs = [Path(dino_run), Path(vjepa_run)]
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    names = labels or [_model_name(run) for run in runs]
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("comparison requires two unique labels")
    metrics = [_load_metrics(run) for run in runs]
    rows = []
    for run_metrics, name in zip(metrics, names):
        for dataset in DATASET_ORDER:
            rows.append({"backbone": name, "dataset": dataset, **run_metrics[dataset]})
    table = pd.DataFrame(rows)
    table.to_csv(output / "comparison_metrics.csv", index=False)
    headline = table[["backbone", "dataset", *METRICS]]
    markdown = ["| " + " | ".join(headline.columns) + " |", "|" + "---|" * len(headline.columns)]
    markdown.extend("| " + " | ".join(map(str, row)) + " |" for row in headline.itertuples(index=False, name=None))
    (output / "comparison_metrics.md").write_text("\n".join(markdown) + "\n")
    _plot_metric_comparison(table, output / "metric_comparison.png")
    _plot_training_curves(runs, names, output / "training_curves.png")
    compute_rows = []
    for run, name, run_metrics in zip(runs, names, metrics):
        summary_path = run / "run_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        inference = np.mean([run_metrics[dataset]["inference_ms_per_image"] for dataset in DATASET_ORDER])
        compute_rows.append({"backbone": name, **summary, "mean_inference_ms_per_image": inference})
    pd.DataFrame(compute_rows).to_csv(output / "compute_comparison.csv", index=False)

    candidates = []
    candidate_total = sum(len(pd.read_csv(_manifest_path(runs[0], dataset))) for dataset in DATASET_ORDER)
    candidate_progress = tqdm(
        total=candidate_total,
        desc="score paired predictions",
        unit="image",
        dynamic_ncols=True,
    )
    for dataset in DATASET_ORDER:
        frame = pd.read_csv(_manifest_path(runs[0], dataset))
        for row in frame.itertuples(index=False):
            prediction_paths = [run / "predictions" / dataset / f"{row.id}.png" for run in runs]
            if not all(path.is_file() for path in prediction_paths):
                raise FileNotFoundError(f"missing paired prediction for {dataset}/{row.id}")
            gt = _mask(Path(row.mask_path))
            predictions = [_mask(path) for path in prediction_paths]
            scores = [_dice(prediction, gt) for prediction in predictions]
            candidates.append(
                {"dataset": dataset, "id": str(row.id), "image_path": row.image_path,
                 "mask_path": row.mask_path, "dino_dice": scores[0], "vjepa_dice": scores[1],
                 "difference": scores[0] - scores[1], "mean_dice": sum(scores) / 2}
            )
            candidate_progress.update(1)
            candidate_progress.set_postfix(dataset=dataset, refresh=False)
    candidate_progress.close()
    candidates = pd.DataFrame(candidates)
    selected = []
    base_count, remainder = divmod(qualitative_count, len(DATASET_ORDER))
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        per_dataset = base_count + int(dataset_index < remainder)
        if per_dataset < 3:
            raise ValueError("qualitative_count must provide at least three examples per dataset")
        group = candidates[candidates.dataset == dataset]
        winner_count = max(1, per_dataset // 3)
        thirds = [
            group.nlargest(winner_count, "difference").assign(selection_reason=f"{names[0]}_win"),
            group.nsmallest(winner_count, "difference").assign(selection_reason=f"{names[1]}_win"),
            group.nsmallest(per_dataset - 2 * winner_count, "mean_dice").assign(selection_reason="hard_case"),
        ]
        chosen = pd.concat(thirds).drop_duplicates(["dataset", "id"])
        if len(chosen) < per_dataset:
            remaining = group[~group.id.isin(chosen.id)].nsmallest(per_dataset - len(chosen), "mean_dice")
            chosen = pd.concat([chosen, remaining.assign(selection_reason="coverage_fill")])
        selected.append(chosen.head(per_dataset))
    selected = pd.concat(selected).reset_index(drop=True)
    selected.to_csv(output / "qualitative_selection.csv", index=False)
    panels = output / "qualitative_panels"; panels.mkdir(exist_ok=True)
    for row in tqdm(
        selected.itertuples(index=False),
        total=len(selected),
        desc="render qualitative panels",
        unit="panel",
        dynamic_ncols=True,
    ):
        predictions = [_mask(run / "predictions" / row.dataset / f"{row.id}.png") for run in runs]
        save_qualitative_panel(
            pd.Series(row._asdict()), *predictions,
            panels / f"{row.dataset}__{row.id}.png", names,
        )
    return output
