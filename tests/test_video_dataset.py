import json

import pandas as pd
import pytest
from PIL import Image

from cod_ssl.data.clip_sampler import ClipSpec
from cod_ssl.data.video_manifest import (
    ManifestVideoCODDataset,
    assert_disjoint_video_splits,
)


def _manifest(tmp_path):
    rows = []
    for video, split in (("v1", "train"), ("v2", "test")):
        for position, number in enumerate((2, 7, 20)):
            image = tmp_path / f"{video}_{number}.png"; mask = tmp_path / f"{video}_{number}_mask.png"
            Image.new("RGB", (10, 8), (position * 30, 0, 0)).save(image)
            Image.new("L", (10, 8), 255 if position == 1 else 0).save(mask)
            rows.append({"dataset": "synthetic", "regime": "default", "split": split,
                         "video_id": video, "source_video_id": video, "frame_id": str(number),
                         "frame_number": number, "image_path": str(image), "mask_path": str(mask),
                         "annotation_type": "manual", "attributes": json.dumps({"kind": "toy"})})
    path = tmp_path / "manifest.csv"; pd.DataFrame(rows).to_csv(path, index=False); return path


def test_manifest_dataset_emits_complete_contract_without_crossing_video(tmp_path):
    dataset = ManifestVideoCODDataset(_manifest(tmp_path), split="train", regime="default",
                                      clip_spec=ClipSpec(3, 1, 1), size=16)
    sample = dataset[0]
    assert sample["frames"].shape == (3, 3, 16, 16)
    assert sample["target_mask"].shape == (1, 16, 16)
    assert sample["source_frame_indices"] == [2, 2, 7]
    assert sample["valid_temporal_mask"].tolist() == [False, True, True]
    assert sample["source_video_id"] == "v1" and sample["attributes"] == {"kind": "toy"}


def test_source_video_leakage_is_fatal(tmp_path):
    frame = pd.read_csv(_manifest(tmp_path)); frame.loc[3, "source_video_id"] = "v1"
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_video_splits(frame)


def test_context_only_rows_are_sampled_but_are_not_supervised_targets(tmp_path):
    path = _manifest(tmp_path)
    frame = pd.read_csv(path)
    frame["is_target"] = False
    frame.loc[(frame.video_id == "v1") & (frame.frame_number == 7), "is_target"] = True
    frame.loc[(frame.video_id == "v2") & (frame.frame_number == 7), "is_target"] = True
    frame.loc[~frame.is_target, "mask_path"] = ""
    frame.to_csv(path, index=False)
    dataset = ManifestVideoCODDataset(
        path, split="train", regime="default", clip_spec=ClipSpec(3, 1, 1), size=16
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["frame_number"] == 7
    assert sample["source_frame_indices"] == [2, 7, 20]


def test_shuffled_diagnostic_preserves_frame_multiset_and_target_slot(tmp_path):
    path = _manifest(tmp_path)
    ordered = ManifestVideoCODDataset(
        path, split="train", regime="default", clip_spec=ClipSpec(3, 1, 1), size=16
    )[1]
    shuffled = ManifestVideoCODDataset(
        path, split="train", regime="default", clip_spec=ClipSpec(3, 1, 1), size=16,
        temporal_order="shuffled", diagnostic_seed=9,
    )[1]
    assert sorted(shuffled["source_frame_indices"]) == sorted(ordered["source_frame_indices"])
    assert shuffled["source_frame_indices"][1] == ordered["source_frame_indices"][1] == 7
    assert shuffled["context_cadence"] == "shuffled_dense"


def test_declared_source_step_and_causal_context_are_enforced(tmp_path):
    path = _manifest(tmp_path)
    wrong = ManifestVideoCODDataset(
        path, split="train", regime="default", clip_spec=ClipSpec(3, 1, 1), size=16,
        source_frame_step=5,
    )
    with pytest.raises(ValueError, match="declared cadence"):
        wrong[1]
    causal = ManifestVideoCODDataset(
        path, split="train", regime="default", clip_spec=ClipSpec(3, 1, 1), size=16,
        context_direction="causal",
    )[1]
    assert causal["source_frame_indices"] == [2, 7, 7]
    assert causal["valid_temporal_mask"].tolist() == [True, True, False]
