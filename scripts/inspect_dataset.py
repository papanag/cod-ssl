#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from cod_ssl.data.video_manifest import REQUIRED_COLUMNS, assert_disjoint_video_splits
from cod_ssl.utils.config import load_config
from cod_ssl.utils.run import file_sha256


def _resolve_manifest(config: dict, explicit: str | None) -> Path:
    value = explicit or os.environ.get(config["manifest_env"], "")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"set {config['manifest_env']} to the canonical video manifest or pass --manifest")
    return path.resolve()


def _overlay(image_path: str, mask_path: str, label: str) -> Image.Image:
    with Image.open(image_path) as source, Image.open(mask_path) as raw_mask:
        image = source.convert("RGB")
        mask = raw_mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    red = Image.new("RGB", image.size, (255, 0, 0))
    alpha = mask.point(lambda value: 100 if value else 0)
    image = Image.composite(red, image, alpha)
    image.thumbnail((320, 240))
    canvas = Image.new("RGB", (320, 270), "white")
    canvas.paste(image, ((320 - image.width) // 2, 0))
    ImageDraw.Draw(canvas).text((5, 246), label[:52], fill="black")
    return canvas


def inspect(frame: pd.DataFrame, manifest: Path, output: Path, seed: int) -> dict:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    assert_disjoint_video_splits(frame)
    if frame.duplicated(["dataset", "regime", "split", "video_id", "frame_id"]).any():
        raise ValueError("duplicate dataset keys occur in the manifest")
    ordered = frame.sort_values(["source_video_id", "frame_number"], kind="stable")
    non_monotonic = 0
    gaps = {}
    video_groups = ordered.groupby("source_video_id", sort=True)
    for video_id, group in tqdm(
        video_groups,
        total=ordered["source_video_id"].nunique(),
        desc="checking chronology",
        unit="video",
        dynamic_ncols=True,
    ):
        numbers = list(map(int, group["frame_number"]))
        non_monotonic += int(any(right <= left for left, right in pairwise(numbers)))
        missing_numbers = [right - left for left, right in pairwise(numbers) if right - left > 1]
        if missing_numbers:
            gaps[str(video_id)] = missing_numbers
    missing_images, missing_masks, empty_masks, full_masks = [], [], [], []
    resolutions, foreground = Counter(), []
    for row in tqdm(
        ordered.itertuples(), total=len(ordered), desc="validating image/mask pairs",
        unit="frame", dynamic_ncols=True,
    ):
        image_path, mask_path = Path(row.image_path), Path(row.mask_path)
        if not image_path.is_file():
            missing_images.append(str(image_path)); continue
        if not mask_path.is_file():
            missing_masks.append(str(mask_path)); continue
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            resolutions[f"{image.width}x{image.height}|{mask.width}x{mask.height}"] += 1
            binary = np.asarray(mask.convert("L")) > 0
        fraction = float(binary.mean()); foreground.append(fraction)
        if fraction == 0: empty_masks.append(str(row.frame_id))
        if fraction == 1: full_masks.append(str(row.frame_id))
    if missing_images or missing_masks:
        raise ValueError(f"missing images={len(missing_images)}, masks={len(missing_masks)}")
    counts = ordered.groupby("source_video_id").size()
    warnings = []
    if "fps" not in ordered or ordered["fps"].isna().any(): warnings.append("unknown_fps")
    if empty_masks: warnings.append("empty_foreground_masks")
    if any(value not in {"manual", "official_manual"} for value in ordered["annotation_type"].unique()):
        warnings.append("non_manual_annotations_present")
    report = {
        "schema_version": 1, "manifest": str(manifest), "manifest_sha256": file_sha256(manifest),
        "rows": len(ordered), "videos": int(ordered["video_id"].nunique()),
        "source_videos": int(ordered["source_video_id"].nunique()),
        "splits": ordered.groupby("split")["source_video_id"].nunique().astype(int).to_dict(),
        "regimes": ordered.get("regime", pd.Series(["default"] * len(ordered))).fillna("default").value_counts().to_dict(),
        "annotation_types": ordered["annotation_type"].value_counts().to_dict(),
        "frames_per_video": {"min": int(counts.min()), "median": float(counts.median()), "max": int(counts.max())},
        "resolutions": dict(resolutions), "foreground_fraction": {
            "min": min(foreground), "median": float(np.median(foreground)), "max": max(foreground)},
        "empty_masks": empty_masks, "full_masks": full_masks,
        "non_monotonic_videos": non_monotonic, "frame_number_gaps": gaps,
        "warnings_requiring_acknowledgement": warnings,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    sample = ordered.sample(n=min(20, len(ordered)), random_state=seed)
    panels = [
        _overlay(str(row.image_path), str(row.mask_path), f"{row.video_id}/{row.frame_id}")
        for row in tqdm(
            sample.itertuples(), total=len(sample), desc="rendering overlays",
            unit="overlay", dynamic_ncols=True,
        )
    ]
    if panels:
        columns, width, height = 4, 320, 270
        sheet = Image.new("RGB", (columns * width, ((len(panels) + columns - 1) // columns) * height), "white")
        for index, panel in enumerate(panels): sheet.paste(panel, ((index % columns) * width, (index // columns) * height))
        sheet.save(output / "random_overlays.png")
    markdown = [f"# Dataset inspection: {ordered.iloc[0]['dataset']}", "", f"Manifest SHA-256: `{report['manifest_sha256']}`", "",
                f"Rows: {report['rows']}; source videos: {report['source_videos']}.", "",
                f"Splits: `{report['splits']}`", "", f"Annotations: `{report['annotation_types']}`", "",
                f"Warnings requiring acknowledgement: `{warnings}`", "",
                "Manual reviewer/date/release/sign-off: **PENDING**"]
    (output / "report.md").write_text("\n".join(markdown) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--manifest")
    parser.add_argument("--regime"); parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config); manifest = _resolve_manifest(config, args.manifest)
    frame = pd.read_csv(manifest)
    if args.regime and "regime" in frame: frame = frame[frame.regime.fillna("default") == args.regime]
    report = inspect(frame, manifest, Path(args.output), args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
