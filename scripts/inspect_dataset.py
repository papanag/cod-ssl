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
from cod_ssl.data.camotion_attributes import ATTRIBUTE_CODES
from cod_ssl.data.clip_sampler import ClipSampler, ClipSpec
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


def _stratified_target_sample(targets: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    groups = [
        group.sample(frac=1, random_state=seed).reset_index(drop=True)
        for _, group in targets.groupby(["split", "annotation_type"], sort=True)
    ]
    selected = []
    position = 0
    while len(selected) < min(size, len(targets)):
        added = False
        for group in groups:
            if position < len(group) and len(selected) < size:
                selected.append(group.iloc[position])
                added = True
        if not added:
            break
        position += 1
    return pd.DataFrame(selected)


def _camotion_artifacts(
    frame: pd.DataFrame, output: Path, seed: int, config: dict
) -> dict:
    if not frame["annotation_type"].eq("official_manual").all():
        raise ValueError("CAMotion primary targets must contain official manual masks only")
    if "attribute_scope" not in frame or not frame["attribute_scope"].eq("sequence").all():
        raise ValueError("CAMotion attributes must be declared at sequence scope")
    vectors = {}
    for sequence_id, group in frame.groupby("source_video_id", sort=True):
        parsed = [json.loads(value) for value in group["attributes"]]
        if any(set(value) != set(ATTRIBUTE_CODES) for value in parsed):
            raise ValueError(f"incomplete CAMotion attribute vector: {sequence_id}")
        if any(value != parsed[0] for value in parsed[1:]):
            raise ValueError(f"CAMotion attributes vary within sequence: {sequence_id}")
        vectors[str(sequence_id)] = parsed[0]
    video_attributes = pd.DataFrame.from_dict(vectors, orient="index").astype(bool)
    target_counts = frame.groupby("source_video_id").size()
    attribute_counts = pd.DataFrame({
        "attribute": ATTRIBUTE_CODES,
        "n_videos": [int(video_attributes[code].sum()) for code in ATTRIBUTE_CODES],
        "n_targets": [int(target_counts[video_attributes.index[video_attributes[code]]].sum())
                      for code in ATTRIBUTE_CODES],
    })
    cooccurrence = video_attributes.astype(int).T @ video_attributes.astype(int)
    attribute_counts.to_csv(output / "attribute_counts.csv", index=False)
    cooccurrence.to_csv(output / "attribute_cooccurrence.csv")
    (output / "attribute_manifest.json").write_text(json.dumps({
        "attribute_scope": "sequence", "attribute_codes": list(ATTRIBUTE_CODES),
        "sequences": vectors,
    }, indent=2) + "\n")
    split_ids = {
        split: sorted(group.source_video_id.astype(str).unique())
        for split, group in frame.groupby("split", sort=True)
    }
    (output / "split_manifest.json").write_text(json.dumps(split_ids, indent=2) + "\n")
    columns = ["split", "source_video_id", "frame_id", "frame_number", "image_path", "mask_path"]
    with (output / "target_manifest.jsonl").open("w") as handle:
        for row in frame[columns].to_dict(orient="records"):
            handle.write(json.dumps(row) + "\n")
    (output / "unmatched_sequences.json").write_text(
        json.dumps({"missing_attribute_rows": [], "extra_attribute_rows": []}, indent=2) + "\n"
    )
    clip_config = config.get("clip", {})
    spec = ClipSpec(
        int(clip_config.get("length", 64)), int(clip_config.get("stride", 1)),
        int(clip_config.get("target_index", 32)),
    )
    sampler = ClipSampler(); invalid_slots = total_slots = 0
    for _, group in frame.groupby("source_video_id", sort=True):
        numbers = group.sort_values("frame_number").frame_number.astype(int).tolist()
        for position in range(len(numbers)):
            _, valid = sampler.source_indices(numbers, position, spec)
            invalid_slots += int((~valid).sum()); total_slots += len(valid)
    expected = config.get("expected", {})
    official_counts = frame.groupby("official_partition")["source_video_id"].nunique().astype(int).to_dict()
    official_targets = frame.groupby("official_partition").size().astype(int).to_dict()
    count_differences = {
        "sequences_total": int(frame.source_video_id.nunique()) - int(expected.get("sequences_total", 0)),
        "sequences_train": official_counts.get("train", 0) - int(expected.get("sequences_train", 0)),
        "sequences_test": official_counts.get("test", 0) - int(expected.get("sequences_test", 0)),
        "unique_rgb_total": len(frame) - int(expected.get("unique_rgb_total", 0)),
        "unique_rgb_train": official_targets.get("train", 0) - int(expected.get("unique_rgb_train", 0)),
        "unique_rgb_test": official_targets.get("test", 0) - int(expected.get("unique_rgb_test", 0)),
    }
    per_sequence_counts = frame.groupby("source_video_id").size()
    split_target_counts = frame.groupby("split").size().astype(int).to_dict()
    summary = {
        "schema_version": 2,
        "dataset": "camotion", "attribute_scope": "sequence",
        "sequences": int(frame.source_video_id.nunique()), "targets": len(frame),
        "derived_split_sequences": {key: len(value) for key, value in split_ids.items()},
        "released_rgb_frames_by_split": split_target_counts,
        "manual_targets_by_split": split_target_counts,
        "released_frames_per_sequence": {
            "min": int(per_sequence_counts.min()),
            "median": float(per_sequence_counts.median()),
            "max": int(per_sequence_counts.max()),
        },
        "official_split_sequences": official_counts,
        "official_split_targets": official_targets,
        "count_difference_sign_convention": "discovered_minus_expected",
        "count_differences": count_differences,
        "paper_claimed_collected_frames": config.get("release_profile", {}).get("paper_claimed_collected_frames"),
        "discovered_sequence_rgb_frames": len(frame),
        "source_frame_step": 5, "released_frame_step": 1,
        "context_cadence": "source_stride5", "dense_intermediate_rgb_available": False,
        "annotation_release_note": (
            "The public sequence-organized archive contains the 30,028 annotated RGB/GT pairs; "
            "the paper-described 149,319 collected source frames are not a public dense-RGB release."
        ),
        "sequences_with_no_listed_attributes": int((~video_attributes.any(axis=1)).sum()),
        "targets_with_bbox": int(frame.get("bbox_available", pd.Series(False)).astype(bool).sum()),
        "clip_boundary_padding_rate": invalid_slots / total_slots,
        "attribute_counts": attribute_counts.to_dict(orient="records"),
        "attribute_cooccurrence": cooccurrence.to_dict(),
        "taxonomy_coverage": {
            "available": False,
            "class": None, "subclass": None, "species": None,
            "note": "The pinned official repository metadata does not publish a sequence taxonomy file.",
        },
        "usage": config.get("usage"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "summary.md").write_text(
        "# CAMotion dataset validation\n\n"
        f"Sequences: {summary['sequences']}; manual targets: {summary['targets']}.\n\n"
        f"Derived splits: `{summary['derived_split_sequences']}`.\n\n"
        f"Release semantics: {summary['annotation_release_note']}\n\n"
        "Attributes are overlapping sequence-level labels, not frame-level onset labels or causal isolates.\n"
    )
    overlays = output / "overlays"; overlays.mkdir(exist_ok=True)
    selected = _stratified_target_sample(frame, 20, seed)
    for index, row in enumerate(selected.itertuples()):
        _overlay(row.image_path, row.mask_path, f"{row.split}: {row.video_id}/{row.frame_id}").save(
            overlays / f"random_{index:02d}.png"
        )
    for code in ATTRIBUTE_CODES:
        ids = set(video_attributes.index[video_attributes[code]])
        examples = frame[(frame.split == "test") & frame.source_video_id.astype(str).isin(ids)].groupby(
            "source_video_id", sort=True
        ).head(1).head(3)
        for index, row in enumerate(examples.itertuples()):
            _overlay(
                row.image_path, row.mask_path,
                f"{code} foreground_fraction={row.foreground_fraction:.4f}",
            ).save(overlays / f"attribute_{code}_{index}.png")
    multilabel_ids = set(video_attributes.index[video_attributes.sum(axis=1) >= 2])
    multilabel_examples = frame[
        (frame.split == "test") & frame.source_video_id.astype(str).isin(multilabel_ids)
    ].groupby("source_video_id", sort=True).head(1).head(3)
    for index, row in enumerate(multilabel_examples.itertuples()):
        labels = [code for code, present in json.loads(row.attributes).items() if present]
        _overlay(
            row.image_path, row.mask_path, f"multi-label {','.join(labels)}"
        ).save(overlays / f"multilabel_{index}.png")
    for sequence_id, group in list(frame.groupby("source_video_id", sort=True))[:3]:
        ordered_group = group.sort_values("frame_number").reset_index(drop=True)
        for label, position in (("first", 0), ("middle", len(ordered_group) // 2),
                                ("last", len(ordered_group) - 1)):
            row = ordered_group.iloc[position]
            _overlay(
                row.image_path, row.mask_path,
                f"{label}: {sequence_id}/{row.frame_id}",
            ).save(overlays / f"anchor_{sequence_id}_{label}.png")
    clip_strips = output / "clip_strips"; clip_strips.mkdir(exist_ok=True)
    display_spec = ClipSpec(5, 1, 2)
    for sequence_id, group in list(frame.groupby("source_video_id", sort=True))[:3]:
        ordered_group = group.sort_values("frame_number").reset_index(drop=True)
        numbers = ordered_group.frame_number.astype(int).tolist()
        for label, target_position in (("first", 0), ("last", len(ordered_group) - 1)):
            positions, valid = sampler.source_indices(numbers, target_position, display_spec)
            panels = []
            for position, is_valid in zip(positions, valid.tolist()):
                row = ordered_group.iloc[position]
                with Image.open(row.image_path) as raw:
                    image = raw.convert("RGB"); image.thumbnail((180, 130))
                panel = Image.new("RGB", (190, 155), "white")
                panel.paste(image, ((190 - image.width) // 2, 0))
                ImageDraw.Draw(panel).text(
                    (4, 134), f"source={row.frame_number} valid={is_valid}", fill="black"
                )
                panels.append(panel)
            strip = Image.new("RGB", (190 * len(panels), 155), "white")
            for index, panel in enumerate(panels):
                strip.paste(panel, (190 * index, 0))
            strip.save(clip_strips / f"{sequence_id}_{label}.png")
    return summary


def inspect(frame: pd.DataFrame, manifest: Path, output: Path, seed: int) -> dict:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    assert_disjoint_video_splits(frame)
    if frame.duplicated(["dataset", "regime", "split", "video_id", "frame_id"]).any():
        raise ValueError("duplicate dataset keys occur in the manifest")
    ordered = frame.sort_values(["source_video_id", "frame_number"], kind="stable")
    if "is_target" not in ordered:
        target_mask = pd.Series(True, index=ordered.index)
    elif pd.api.types.is_bool_dtype(ordered["is_target"]):
        target_mask = ordered["is_target"].fillna(False)
    else:
        target_mask = ordered["is_target"].fillna("").astype(str).str.lower().eq("true")
    targets = ordered[target_mask]
    if targets.empty:
        raise ValueError("manifest contains no supervised target rows")
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
    missing_images, missing_masks, empty_masks, full_masks, nonbinary_masks = [], [], [], [], []
    resolutions, foreground = Counter(), []
    for row in tqdm(
        ordered.itertuples(), total=len(ordered), desc="validating frames and target masks",
        unit="frame", dynamic_ncols=True,
    ):
        image_path = Path(row.image_path)
        if not image_path.is_file():
            missing_images.append(str(image_path))
            continue
        raw_target = getattr(row, "is_target", True)
        is_target = raw_target if isinstance(raw_target, bool) else str(raw_target).lower() == "true"
        if not is_target:
            with Image.open(image_path) as image:
                resolutions[f"{image.width}x{image.height}|context"] += 1
            continue
        if pd.isna(row.mask_path) or not str(row.mask_path).strip():
            missing_masks.append(f"{row.video_id}/{row.frame_id}: blank")
            continue
        mask_path = Path(row.mask_path)
        if not mask_path.is_file():
            missing_masks.append(str(mask_path))
            continue
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            resolutions[f"{image.width}x{image.height}|{mask.width}x{mask.height}"] += 1
            mask_array = np.asarray(mask.convert("L"))
            binary = mask_array > 0
            if not set(np.unique(mask_array)).issubset({0, 255}):
                nonbinary_masks.append(str(row.frame_id))
        fraction = float(binary.mean())
        foreground.append(fraction)
        if fraction == 0:
            empty_masks.append(str(row.frame_id))
        if fraction == 1:
            full_masks.append(str(row.frame_id))
    if missing_images or missing_masks:
        raise ValueError(f"missing images={len(missing_images)}, masks={len(missing_masks)}")
    counts = ordered.groupby("source_video_id").size()
    warnings = []
    if "fps" not in ordered or ordered["fps"].isna().any():
        warnings.append("unknown_fps")
    if empty_masks:
        warnings.append("empty_foreground_masks")
    if nonbinary_masks:
        warnings.append("nonbinary_masks")
    if any(value not in {"manual", "official_manual"} for value in targets["annotation_type"].unique()):
        warnings.append("non_manual_annotations_present")
    report = {
        "schema_version": 1, "manifest": str(manifest), "manifest_sha256": file_sha256(manifest),
        "rows": len(ordered), "target_rows": len(targets),
        "videos": int(ordered["video_id"].nunique()),
        "source_videos": int(ordered["source_video_id"].nunique()),
        "splits": ordered.groupby("split")["source_video_id"].nunique().astype(int).to_dict(),
        "regimes": ordered.get("regime", pd.Series(["default"] * len(ordered))).fillna("default").value_counts().to_dict(),
        "annotation_types": ordered["annotation_type"].value_counts().to_dict(),
        "frames_per_video": {"min": int(counts.min()), "median": float(counts.median()), "max": int(counts.max())},
        "resolutions": dict(resolutions), "foreground_fraction": {
            "min": min(foreground), "median": float(np.median(foreground)), "max": max(foreground)},
        "empty_masks": empty_masks, "full_masks": full_masks,
        "nonbinary_masks": nonbinary_masks,
        "non_monotonic_videos": non_monotonic, "frame_number_gaps": gaps,
        "warnings_requiring_acknowledgement": warnings,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    sample = _stratified_target_sample(targets, 20, seed)
    panels = [
        _overlay(
            str(row.image_path),
            str(row.mask_path),
            f"{row.split}/{row.annotation_type}: {row.video_id}/{row.frame_id}",
        )
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
                f"Context rows: {report['rows']}; supervised targets: {report['target_rows']}; source videos: {report['source_videos']}.", "",
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
    loaded = load_config(args.config)
    config = loaded.get("dataset", loaded)
    if "clip" in loaded:
        config = config | {"clip": loaded["clip"]}
    if "dataset_build" in loaded:
        config = config | {"expected": loaded["dataset_build"].get("expected", {})}
    manifest = _resolve_manifest(config, args.manifest)
    if config["name"] == "moca_mask_dense":
        from cod_ssl.data.preprocessing.prepare_moca_mask_dense import verify_moca_mask_dense
        if manifest.name != "runtime_manifest.csv" or manifest.parent.name != "manifest":
            raise ValueError("dense MoCA inspection requires a moca_mask_dense_v1 runtime manifest")
        verify_moca_mask_dense(manifest.parent.parent)
    frame = pd.read_csv(manifest)
    if args.regime and "regime" in frame: frame = frame[frame.regime.fillna("default") == args.regime]
    report = inspect(frame, manifest, Path(args.output), args.seed)
    if config["name"] == "camotion":
        _camotion_artifacts(frame, Path(args.output), args.seed, config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
