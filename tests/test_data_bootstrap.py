from pathlib import Path

from PIL import Image

from cod_ssl.data.bootstrap import (
    build_standard_train_manifest,
    build_test_manifest,
    discover_dataset_pair,
    discover_standard_training_pair,
)


def test_discovery_and_manifest_source_counts(tmp_path):
    image_dir = tmp_path / "TrainDataset" / "Imgs"
    mask_dir = tmp_path / "TrainDataset" / "GT"
    image_dir.mkdir(parents=True); mask_dir.mkdir(parents=True)
    for index in range(3040):
        stem = f"COD10K-TR-{index:04d}"
        Image.new("RGB", (1, 1)).save(image_dir / f"{stem}.jpg")
        Image.new("L", (1, 1)).save(mask_dir / f"{stem}.png")
    for index in range(1000):
        stem = f"camo_{index:04d}"
        Image.new("RGB", (1, 1)).save(image_dir / f"{stem}.jpg")
        Image.new("L", (1, 1)).save(mask_dir / f"{stem}.png")
    assert discover_standard_training_pair(tmp_path) == (image_dir, mask_dir)
    frame = build_standard_train_manifest(image_dir, mask_dir, tmp_path / "train.csv")
    assert frame.groupby("source").size().to_dict() == {"camo": 1000, "cod10k": 3040}


def test_standard_test_pair_discovery_and_manifest(tmp_path, monkeypatch):
    from cod_ssl.data import bootstrap

    monkeypatch.setitem(bootstrap.STANDARD_TEST_COUNTS, "tiny_test", 3)
    image_dir = tmp_path / "tiny" / "Images"
    mask_dir = tmp_path / "tiny" / "GT"
    image_dir.mkdir(parents=True); mask_dir.mkdir(parents=True)
    for index in range(3):
        Image.new("RGB", (2, 2)).save(image_dir / f"sample_{index}.jpg")
        Image.new("L", (2, 2)).save(mask_dir / f"sample_{index}.png")
    assert discover_dataset_pair(tmp_path, 3) == (image_dir, mask_dir)
    frame = build_test_manifest("tiny_test", image_dir, mask_dir, tmp_path / "tiny.csv")
    assert len(frame) == 3
    assert set(frame.source) == {"tiny_test"}
