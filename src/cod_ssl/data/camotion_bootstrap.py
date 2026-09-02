from __future__ import annotations

import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from cod_ssl.data.bootstrap import IMAGE_SUFFIXES
from cod_ssl.data.camotion_attributes import ATTRIBUTE_CODES, align_camotion_attributes
from cod_ssl.data.video_manifest import assert_disjoint_video_splits

CAMOTION_EXPECTED = {
    "official_train_sequences": 359,
    "official_test_sequences": 115,
    "annotated_train_frames": 23_253,
    "annotated_test_frames": 6_775,
    "reported_total_rgb_frames": 149_319,
    "archive_size_bytes": 16_912_530_524,
    "released_original_frame_interval": 5,
}


def verify_camotion_flattened_segmentation_duplicates(
    archive: str | Path,
) -> dict[str, object]:
    """Verify flattened RGB/GT exports duplicate the canonical segmentation assets."""
    archive = Path(archive)
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"CAMotion release is not a ZIP archive: {archive}")
    report: dict[str, object] = {
        "flattened_rgb_gt_are_duplicates": True,
        "validated_assets": ["Imgs", "GT"],
        "ignored_assets": ["Edge", "Bbox", "BBox"],
        "partitions": {},
    }
    with zipfile.ZipFile(archive) as handle:
        files = [member for member in handle.infolist() if not member.is_dir()]
        for partition, canonical_token, flattened_token in (
            ("train", "/TrainDataset_per_sq/", "/CAMotion-TR/"),
            ("test", "/TestDataset_per_sq/", "/CAMotion-TE/"),
        ):
            partition_report = {}
            for asset, aliases in (("Imgs", {"Imgs"}), ("GT", {"GT"})):
                canonical = Counter(
                    (member.CRC, member.file_size)
                    for member in files
                    if canonical_token in f"/{member.filename}"
                    and aliases.intersection(Path(member.filename).parts)
                )
                flattened = Counter(
                    (member.CRC, member.file_size)
                    for member in files
                    if flattened_token in f"/{member.filename}"
                    and aliases.intersection(Path(member.filename).parts)
                )
                if not canonical or canonical != flattened:
                    raise ValueError(
                        f"CAMotion flattened {partition}/{asset} is absent or differs from canonical assets"
                    )
                partition_report[asset] = {"canonical": sum(canonical.values()), "flattened": sum(flattened.values()),
                                           "mismatches": 0}
            report["partitions"][partition] = partition_report
    return report


def _unique_named_directory(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_dir() and path.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name!r} directory under {root}, found {len(matches)}")
    return matches[0]


def discover_camotion_roots(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    return {
        "official_train": _unique_named_directory(root, "TrainDataset_per_sq"),
        "official_test": _unique_named_directory(root, "TestDataset_per_sq"),
    }


def _numeric_files(directory: Path, *, suffixes: set[str]) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing CAMotion directory: {directory}")
    result: dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            key = int(path.stem)
        except ValueError as error:
            raise ValueError(f"non-numeric CAMotion frame stem: {path}") from error
        if key in result:
            raise ValueError(f"ambiguous CAMotion numeric frame key {key} in {directory}")
        result[key] = path.resolve()
    if not result:
        raise ValueError(f"no CAMotion assets found in {directory}")
    return dict(sorted(result.items()))


def _index_partition(root: Path) -> dict[str, dict[str, dict[int, Path]]]:
    result = {}
    for sequence in tqdm(
        sorted(path for path in root.iterdir() if path.is_dir()),
        desc=f"index {root.name}", unit="sequence", dynamic_ncols=True,
    ):
        images = _numeric_files(sequence / "Imgs", suffixes=IMAGE_SUFFIXES)
        masks = _numeric_files(sequence / "GT", suffixes=IMAGE_SUFFIXES)
        if images.keys() != masks.keys():
            raise ValueError(
                f"ambiguous CAMotion RGB/GT pairing for {sequence.name}: "
                f"RGB-only={sorted(images.keys() - masks.keys())[:5]}, "
                f"GT-only={sorted(masks.keys() - images.keys())[:5]}"
            )
        frame_ids = list(images)
        if frame_ids[0] != 0 or any(number % 5 for number in frame_ids):
            raise ValueError(f"CAMotion source IDs violate the public stride-5 profile: {sequence.name}")
        if any(right - left != 5 for left, right in zip(frame_ids, frame_ids[1:])):
            raise ValueError(f"CAMotion sequence has a non-stride-5 filename gap: {sequence.name}")
        result[sequence.name] = {"images": images, "masks": masks}
    return result


def build_camotion_manifest(
    release_root: str | Path,
    attributes: dict[str, dict[str, bool]],
    output: str | Path,
    *,
    validation_fraction: float = 0.1,
    seed: int = 42,
    verify_official_counts: bool = True,
) -> tuple[pd.DataFrame, dict]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one")
    roots = discover_camotion_roots(release_root)
    official_train = _index_partition(roots["official_train"])
    official_test = _index_partition(roots["official_test"])
    if set(official_train) & set(official_test):
        raise ValueError("official CAMotion train/test sequence identities overlap")
    counts = {
        "official_train_sequences": len(official_train),
        "official_test_sequences": len(official_test),
        "annotated_train_frames": sum(len(value["images"]) for value in official_train.values()),
        "annotated_test_frames": sum(len(value["images"]) for value in official_test.values()),
    }
    expected_selected = {key: CAMOTION_EXPECTED[key] for key in counts}
    if verify_official_counts and counts != expected_selected:
        raise ValueError(f"official CAMotion selected-asset counts differ: {counts}")
    all_ids = set(official_train) | set(official_test)
    aligned_attributes = align_camotion_attributes(all_ids, attributes)

    shuffled = sorted(official_train)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation_ids = set(shuffled[:validation_count])
    rows = []
    partitions = (
        ("train", {key: value for key, value in official_train.items() if key not in validation_ids}),
        ("val", {key: value for key, value in official_train.items() if key in validation_ids}),
        ("test", official_test),
    )
    total_targets = counts["annotated_train_frames"] + counts["annotated_test_frames"]
    progress = tqdm(total=total_targets, desc="build CAMotion manifest", unit="target", dynamic_ncols=True)
    for split, sequences in partitions:
        for sequence_id in sorted(sequences):
            assets = sequences[sequence_id]
            attribute_vector = aligned_attributes[sequence_id]
            for sequence_position, frame_number in enumerate(sorted(assets["images"])):
                image_path = assets["images"][frame_number]
                mask_path = assets["masks"][frame_number]
                with Image.open(mask_path) as mask:
                    foreground_fraction = float((np.asarray(mask.convert("L")) > 0).mean())
                rows.append({
                    "dataset": "camotion", "regime": "S5", "split": split,
                    "video_id": sequence_id, "source_video_id": sequence_id,
                    "frame_id": image_path.stem, "frame_number": frame_number,
                    "source_frame_number": frame_number, "sequence_position": sequence_position,
                    "image_path": str(image_path), "mask_path": str(mask_path),
                    "annotation_type": "official_manual", "is_target": True, "fps": None,
                    "attributes": json.dumps(attribute_vector, sort_keys=True),
                    "attribute_scope": "sequence",
                    "foreground_fraction": foreground_fraction,
                    "class": None, "subclass": None, "species": None,
                    "official_partition": "test" if split == "test" else "train",
                    "release_profile": "camotion_public_stride5_v1",
                    "context_cadence": "source_stride5", "released_frame_step": 1,
                    "source_frame_step": 5, "dense_intermediate_rgb_available": False,
                    "boundary_policy": "public_sequence_extent_v1",
                })
                progress.update(1)
    progress.close()
    frame = pd.DataFrame(rows).sort_values(
        ["split", "source_video_id", "frame_number"], kind="stable"
    ).reset_index(drop=True)
    assert_disjoint_video_splits(frame)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    report = {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "validation_source_video_ids": sorted(validation_ids),
        "source_videos": frame.groupby("split")["source_video_id"].nunique().astype(int).to_dict(),
        "target_frames": frame.groupby("split").size().astype(int).to_dict(),
        "official_selected_asset_counts": counts,
        "reported_total_rgb_frames": CAMOTION_EXPECTED["reported_total_rgb_frames"],
        "paper_claimed_collected_frames": CAMOTION_EXPECTED["reported_total_rgb_frames"],
        "discovered_sequence_rgb_frames": len(frame),
        "released_original_frame_interval": 5,
        "dense_intermediate_rgb_available": False,
        "flattened_rgb_gt_are_duplicates": True,
        "release_profile": "camotion_public_stride5_v1",
        "attribute_scope": "sequence",
        "attribute_codes": list(ATTRIBUTE_CODES),
    }
    output.with_suffix(".splits.json").write_text(json.dumps(report, indent=2) + "\n")
    return frame, report
