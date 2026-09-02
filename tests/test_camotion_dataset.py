import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

from cod_ssl.data.camotion_attributes import ATTRIBUTE_CODES
from cod_ssl.data.camotion_bootstrap import (
    build_camotion_manifest,
    verify_camotion_flattened_segmentation_duplicates,
)
from cod_ssl.data.clip_sampler import ClipSpec
from cod_ssl.data.video_manifest import ManifestVideoCODDataset


def _sequence(root: Path, split_root: str, sequence_id: str, stems=("0", "5", "10")) -> None:
    sequence = root / "release" / split_root / sequence_id
    for directory in ("Imgs", "GT", "Bbox"):
        (sequence / directory).mkdir(parents=True, exist_ok=True)
    for stem in stems:
        Image.new("RGB", (9, 7), (int(stem), 0, 0)).save(sequence / "Imgs" / f"{stem}.jpg")
        Image.new("L", (9, 7), 255).save(sequence / "GT" / f"{stem}.png")
        (sequence / "Bbox" / f"{stem}.txt").write_text("0 0 8 6\n")


def _attributes(ids):
    return {
        sequence_id: {code: code in ({"OC", "MB"} if index % 2 else set()) for code in ATTRIBUTE_CODES}
        for index, sequence_id in enumerate(ids)
    }


def test_camotion_manifest_preserves_official_test_and_numeric_order(tmp_path):
    train_ids = ["train_a", "train_b", "train_c"]
    test_ids = ["test_a"]
    for sequence_id in train_ids:
        _sequence(tmp_path, "TrainDataset_per_sq", sequence_id)
    _sequence(tmp_path, "TestDataset_per_sq", test_ids[0])
    output = tmp_path / "camotion.csv"
    frame, report = build_camotion_manifest(
        tmp_path / "release", _attributes(train_ids + test_ids), output,
        validation_fraction=1 / 3, seed=9, verify_official_counts=False,
    )
    assert set(frame[frame.official_partition == "test"].source_video_id) == set(test_ids)
    assert set(frame[frame.split == "val"].source_video_id) <= set(train_ids)
    assert not set(frame[frame.split == "train"].source_video_id) & set(
        frame[frame.split == "val"].source_video_id
    )
    assert all(
        numbers == [0, 5, 10]
        for numbers in frame.groupby("source_video_id").frame_number.apply(list)
    )
    assert frame.is_target.all() and set(frame.annotation_type) == {"official_manual"}
    assert frame.attribute_scope.eq("sequence").all()
    assert "bbox_available" not in frame and "bbox_path" not in frame
    assert report["release_profile"] == "camotion_public_stride5_v1"
    assert report["dense_intermediate_rgb_available"] is False
    assert output.with_suffix(".splits.json").is_file()
    assert pd.read_csv(output).shape == frame.shape


def test_camotion_sparse_release_clip_uses_available_rgb_without_crossing_sequence(tmp_path):
    ids = ["train_a", "train_b", "test_a"]
    for sequence_id in ids[:2]:
        _sequence(tmp_path, "TrainDataset_per_sq", sequence_id)
    _sequence(tmp_path, "TestDataset_per_sq", ids[2])
    output = tmp_path / "camotion.csv"
    frame, _ = build_camotion_manifest(
        tmp_path / "release", _attributes(ids), output,
        validation_fraction=0.5, seed=1, verify_official_counts=False,
    )
    train_id = frame[frame.split == "train"].source_video_id.iloc[0]
    dataset = ManifestVideoCODDataset(
        output, split="train", regime="S5", clip_spec=ClipSpec(3, 1, 1), size=16
    )
    sample = dataset[1]
    assert sample["source_video_id"] == train_id
    assert sample["source_frame_indices"] == [0, 5, 10]
    assert sample["source_sequence_positions"] == [0, 1, 2]
    assert sample["released_frame_step"] == 1
    assert sample["source_frame_step"] == 5
    assert sample["context_cadence"] == "source_stride5"
    assert sample["attributes"] == json.loads(
        frame[(frame.split == "train") & (frame.frame_number == 5)].iloc[0].attributes
    )
    assert sample["metadata"]["attribute_scope"] == "sequence"


def test_camotion_inspection_emits_attribute_and_split_artifacts(tmp_path):
    ids = ["train_a", "train_b", "test_a"]
    for sequence_id in ids[:2]:
        _sequence(tmp_path, "TrainDataset_per_sq", sequence_id)
    _sequence(tmp_path, "TestDataset_per_sq", ids[2])
    manifest = tmp_path / "camotion.csv"
    build_camotion_manifest(
        tmp_path / "release", _attributes(ids), manifest,
        validation_fraction=0.5, seed=1, verify_official_counts=False,
    )
    output = tmp_path / "inspection"
    root = Path(__file__).parents[1]
    subprocess.run(
        [sys.executable, "scripts/inspect_dataset.py",
         "--config", "configs/datasets/camotion.yaml",
         "--manifest", str(manifest), "--output", str(output)],
        cwd=root, check=True,
    )
    for name in (
        "summary.json", "summary.md", "split_manifest.json", "target_manifest.jsonl",
        "attribute_manifest.json", "attribute_counts.csv", "attribute_cooccurrence.csv",
        "unmatched_sequences.json", "random_overlays.png",
    ):
        assert (output / name).is_file()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["attribute_scope"] == "sequence"
    assert summary["targets"] == 9
    assert any((output / "clip_strips").iterdir())
    cooccurrence = pd.read_csv(output / "attribute_cooccurrence.csv", index_col=0)
    assert cooccurrence.equals(cooccurrence.T)


def test_camotion_only_requires_flattened_rgb_gt_equivalence(tmp_path):
    archive = tmp_path / "CAMotion.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for partition, canonical, flattened in (
            ("train", "TrainDataset_per_sq", "CAMotion-TR"),
            ("test", "TestDataset_per_sq", "CAMotion-TE"),
        ):
            for asset in ("Imgs", "GT"):
                content = f"{partition}-{asset}".encode()
                handle.writestr(f"CAMotion/CAMotion/{canonical}/seq/{asset}/00000.bin", content)
                handle.writestr(f"CAMotion/CAMotion/{flattened}/{asset}/seq_00000.bin", content)
            handle.writestr(
                f"CAMotion/CAMotion/{canonical}/seq/Bbox/00000.txt", b"canonical-only"
            )
            handle.writestr(
                f"CAMotion/CAMotion/{flattened}/Edge/seq_00000.png", b"flattened-only"
            )
    report = verify_camotion_flattened_segmentation_duplicates(archive)
    assert report["flattened_rgb_gt_are_duplicates"] is True
    assert report["validated_assets"] == ["Imgs", "GT"]
    assert report["ignored_assets"] == ["Edge", "Bbox", "BBox"]
    assert report["partitions"]["train"]["Imgs"]["mismatches"] == 0
