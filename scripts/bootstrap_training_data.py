#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

import gdown

from cod_ssl.data.bootstrap import build_standard_train_manifest, discover_standard_training_pair

OFFICIAL_TRAIN_FILE_ID = "1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt"


def _safe_destination(root: Path, member_name: str) -> Path:
    destination = (root / member_name).resolve()
    if root.resolve() not in destination.parents and destination != root.resolve():
        raise ValueError(f"archive contains unsafe path: {member_name}")
    return destination


def extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                _safe_destination(destination, member.filename)
            handle.extractall(destination)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                _safe_destination(destination, member.name)
            handle.extractall(destination, filter="data")
        return
    raise ValueError(f"unsupported training archive: {archive}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--accept-noncommercial-license", action="store_true")
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
    print(f"Images: {image_dir}")
    print(f"Masks: {mask_dir}")
    print(frame.groupby("source").size())
    print(f"Manifest: {args.manifest} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
