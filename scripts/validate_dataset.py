#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

EXPECTED = {"train_all": 4040, "camo_test": 250, "cod10k_test": 2026,
            "chameleon": 76, "nc4k": 4121}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--manifest-dir", default="manifests")
    p.add_argument("--skip-hashes", action="store_true"); a = p.parse_args()
    seen_paths, train_hashes, test_hashes = set(), set(), set()
    for name, expected in EXPECTED.items():
        frame = pd.read_csv(Path(a.manifest_dir) / f"{name}.csv")
        if len(frame) != expected: raise ValueError(f"{name}: expected {expected}, got {len(frame)}")
        for row in frame.itertuples():
            image, mask = Path(row.image_path).resolve(), Path(row.mask_path).resolve()
            if image in seen_paths: raise ValueError(f"duplicate image path: {image}")
            seen_paths.add(image)
            with Image.open(image) as im: im.verify()
            with Image.open(mask) as gt:
                if gt.width <= 0 or gt.height <= 0: raise ValueError(f"empty mask: {mask}")
                # Force the same binary conversion used by the dataset.
                gt.convert("L").point(lambda value: 255 if value > 0 else 0)
            if not a.skip_hashes:
                (train_hashes if name == "train_all" else test_hashes).add(digest(image))
    overlap = train_hashes & test_hashes
    if overlap: raise ValueError(f"{len(overlap)} image hashes overlap between train and tests")
    print("Dataset validation passed.")


if __name__ == "__main__": main()

