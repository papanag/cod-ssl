#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from cod_ssl.data.exclusions import load_exclusion_policy

VALIDATOR_VERSION = "3.0.0"
EXPECTED = {"train_all": 4033, "camo_test": 250, "cod10k_test": 2019,
            "chameleon": 76, "nc4k": 4121}


def digest(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validation_identity(manifest_dir: str | Path, exclusions: str | Path, *, skip_hashes: bool):
    manifest_dir = Path(manifest_dir)
    manifest_hashes, counts = {}, {}
    for name in EXPECTED:
        path = manifest_dir / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"manifest not found: {path}")
        manifest_hashes[name] = digest(path)
        counts[name] = len(pd.read_csv(path))
    settings = {
        "skip_image_hashes": bool(skip_hashes),
        "require_zero_train_test_overlap": not skip_hashes,
        "exclusions_file": str(Path(exclusions).resolve()),
        "exclusions_sha256": digest(exclusions),
        "excluded_ids_by_split": {
            name: sorted(ids) for name, ids in load_exclusion_policy(exclusions).items()
        },
        "expected_effective_counts": EXPECTED,
    }
    return manifest_hashes, counts, settings


def receipt_matches(receipt: dict[str, Any], manifest_hashes, counts, settings) -> bool:
    return bool(
        receipt.get("validation_passed") is True
        and receipt.get("validator_version") == VALIDATOR_VERSION
        and receipt.get("manifest_hashes") == manifest_hashes
        and receipt.get("dataset_counts") == counts
        and receipt.get("validation_settings") == settings
    )


def validate(manifest_dir: str | Path, exclusions: str | Path, *, skip_hashes: bool = False):
    manifest_dir = Path(manifest_dir)
    exclusion_policy = load_exclusion_policy(exclusions)
    manifest_hashes, counts, settings = validation_identity(
        manifest_dir, exclusions, skip_hashes=skip_hashes
    )
    if counts != EXPECTED:
        raise ValueError(f"effective dataset counts differ: expected {EXPECTED}, got {counts}")
    seen_paths: set[Path] = set()
    train_hashes: dict[str, str] = {}
    test_hashes: dict[str, str] = {}
    for name, expected in EXPECTED.items():
        frame = pd.read_csv(manifest_dir / f"{name}.csv")
        excluded = exclusion_policy.get(name, set())
        present = sorted(set(frame.id.astype(str).map(lambda value: Path(value).stem)) & excluded)
        if present:
            raise ValueError(f"{name} still contains excluded overlap IDs: {present}")
        print(f"Validating {name}: {expected} pairs", flush=True)
        for index, row in enumerate(frame.itertuples(), start=1):
            image, mask = Path(row.image_path).resolve(), Path(row.mask_path).resolve()
            if image in seen_paths:
                raise ValueError(f"duplicate image path: {image}")
            seen_paths.add(image)
            with Image.open(image) as source:
                source.verify()
            with Image.open(mask) as ground_truth:
                if ground_truth.width <= 0 or ground_truth.height <= 0:
                    raise ValueError(f"empty mask: {mask}")
                ground_truth.convert("L").point(lambda value: 255 if value > 0 else 0)
            if not skip_hashes:
                target = train_hashes if name == "train_all" else test_hashes
                target[digest(image)] = f"{name}/{row.id}"
            if index % 500 == 0 or index == expected:
                print(f"  {index}/{expected}", flush=True)
    if not skip_hashes:
        overlap = sorted(set(train_hashes) & set(test_hashes))
        if overlap:
            details = [f"{train_hashes[value]} == {test_hashes[value]}" for value in overlap]
            raise ValueError("train/test image overlap remains:\n" + "\n".join(details))
    return manifest_hashes, counts, settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", default="manifests")
    parser.add_argument("--exclusions", default="configs/dataset_exclusions.csv")
    parser.add_argument("--receipt")
    parser.add_argument("--skip-hashes", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest_hashes, counts, settings = validation_identity(
        args.manifest_dir, args.exclusions, skip_hashes=args.skip_hashes
    )
    receipt_path = Path(args.receipt) if args.receipt else None
    if receipt_path and receipt_path.is_file() and not args.force:
        try:
            receipt = json.loads(receipt_path.read_text())
        except json.JSONDecodeError:
            receipt = {}
        if receipt_matches(receipt, manifest_hashes, counts, settings):
            print(f"Using cached dataset-validation receipt: {receipt_path}")
            print(f"Validated at: {receipt['completion_timestamp_utc']}")
            return
        print("Validation receipt is stale; running full validation.")
    manifest_hashes, counts, settings = validate(
        args.manifest_dir, args.exclusions, skip_hashes=args.skip_hashes
    )
    receipt = {
        "validation_passed": True,
        "validator_version": VALIDATOR_VERSION,
        "manifest_hashes": manifest_hashes,
        "dataset_counts": counts,
        "validation_settings": settings,
        "completion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if receipt_path:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Validation receipt written: {receipt_path}")
    print("Dataset validation passed.")


if __name__ == "__main__":
    main()
