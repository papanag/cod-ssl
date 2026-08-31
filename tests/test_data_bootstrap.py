from pathlib import Path

from PIL import Image

from cod_ssl.data.bootstrap import build_standard_train_manifest, discover_standard_training_pair


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
