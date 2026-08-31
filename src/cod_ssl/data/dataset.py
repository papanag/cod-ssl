from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from cod_ssl.data.transforms import PairedTransform


class CODDataset(Dataset):
    def __init__(self, manifest: str | Path, training: bool = False):
        self.manifest = Path(manifest).resolve()
        self.rows = pd.read_csv(self.manifest)
        required = {"id", "source", "image_path", "mask_path"}
        if not required.issubset(self.rows.columns):
            raise ValueError(f"manifest missing columns: {sorted(required - set(self.rows.columns))}")
        self.transform = PairedTransform(training=training)

    def __len__(self): return len(self.rows)

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.manifest.parent / path

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        image_path, mask_path = self._path(row.image_path), self._path(row.mask_path)
        with Image.open(image_path) as raw_image, Image.open(mask_path) as raw_mask:
            image, mask = raw_image.convert("RGB"), raw_mask.convert("L")
            original_size = (raw_mask.height, raw_mask.width)
            image_tensor, mask_tensor = self.transform(image, mask)
        return {"image": image_tensor, "mask": mask_tensor, "id": str(row.id),
                "source": str(row.source), "original_size": original_size,
                "image_path": str(image_path), "mask_path": str(mask_path)}

