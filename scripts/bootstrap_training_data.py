#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gdown

from cod_ssl.data.bootstrap import (
    build_standard_train_manifest,
    discover_standard_training_pair,
    extract_archive,
)
from cod_ssl.data.exclusions import exclude_manifest_rows, load_exclusion_policy

OFFICIAL_TRAIN_FILE_ID = "1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--accept-noncommercial-license", action="store_true")
    parser.add_argument("--exclusions", default="configs/dataset_exclusions.csv")
    args = parser.parse_args()
    if not args.accept_noncommercial_license:
        raise PermissionError(
            "COD10K is non-commercial; pass --accept-noncommercial-license only after reviewing its license"
        )
    root = Path(args.data_root)
    archive = root / "archives" / "cod_standard_train.zip"
    extracted = root / "standard_train"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        print("Downloading the official SINet/COD10K combined training archive...")
        result = gdown.download(id=OFFICIAL_TRAIN_FILE_ID, output=str(archive), quiet=False)
        if result is None or not archive.is_file():
            raise RuntimeError("Google Drive dataset download failed or quota was exceeded")
    else:
        print(f"Using cached archive: {archive}")
    try:
        image_dir, mask_dir = discover_standard_training_pair(extracted)
    except RuntimeError:
        print(f"Extracting {archive} to {extracted}...")
        extract_archive(archive, extracted)
        image_dir, mask_dir = discover_standard_training_pair(extracted)
    frame = build_standard_train_manifest(image_dir, mask_dir, args.manifest)
    exclusions = load_exclusion_policy(args.exclusions)["train_all"]
    frame, removed = exclude_manifest_rows(
        frame, exclusions, dataset_name="train_all", require_all=True
    )
    frame.to_csv(args.manifest, index=False)
    exclusion_report = Path(args.manifest).with_suffix(".exclusions.json")
    exclusion_report.write_text(
        json.dumps({"dataset": "train_all", "removed_ids": removed}, indent=2) + "\n"
    )
    print(f"Images: {image_dir}")
    print(f"Masks: {mask_dir}")
    print(frame.groupby("source").size())
    print(f"Excluded known train/test overlaps: {removed}")
    print(f"Manifest: {args.manifest} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
