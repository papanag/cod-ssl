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
