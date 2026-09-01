#!/usr/bin/env python3
"""Build manifests from explicit image/mask directories without dataset assumptions."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from cod_ssl.data.manifests import create_dev_split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split-train", help="existing 4,038-row decontaminated train manifest to split")
    p.add_argument("--train-dev", default="manifests/train_dev.csv")
    p.add_argument("--val-dev", default="manifests/val_dev.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source"); p.add_argument("--images")
    p.add_argument("--masks"); p.add_argument("--output")
    p.add_argument("--image-glob", default="*"); p.add_argument("--mask-suffix", default=".png")
    a = p.parse_args()
    if a.split_train:
        frame = pd.read_csv(a.split_train)
        if len(frame) != 4038: raise ValueError(f"expected 4038 decontaminated training rows, got {len(frame)}")
        create_dev_split(a.split_train, a.train_dev, a.val_dev, seed=a.seed)
        return
    if not all((a.source, a.images, a.masks, a.output)):
        p.error("pair generation requires --source, --images, --masks, and --output")
    images, masks = Path(a.images), Path(a.masks)
    rows = []
    for image in sorted(path for path in images.glob(a.image_glob) if path.is_file()):
        mask = masks / f"{image.stem}{a.mask_suffix}"
        if not mask.is_file(): raise FileNotFoundError(f"missing mask for {image}: {mask}")
        rows.append({"id": image.stem, "source": a.source,
                     "image_path": str(image.resolve()), "mask_path": str(mask.resolve())})
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["id", "source", "image_path", "mask_path"]).to_csv(a.output, index=False)


if __name__ == "__main__": main()
