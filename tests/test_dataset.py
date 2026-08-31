import pandas as pd
import torch
from PIL import Image
from cod_ssl.data.dataset import CODDataset


def test_dataset_binary_mask_and_metadata(tmp_path):
    image = Image.new("RGB", (20, 10), "white"); mask = Image.new("L", (20, 10), 0)
    mask.putpixel((2, 3), 7); image.save(tmp_path / "x.jpg"); mask.save(tmp_path / "x.png")
    pd.DataFrame([{"id":"x", "source":"test", "image_path":"x.jpg", "mask_path":"x.png"}]).to_csv(tmp_path / "m.csv", index=False)
    sample = CODDataset(tmp_path / "m.csv")[0]
    assert sample["image"].shape == (3, 384, 384)
    assert sample["mask"].shape == (1, 384, 384)
    assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}
    assert sample["original_size"] == (10, 20)

