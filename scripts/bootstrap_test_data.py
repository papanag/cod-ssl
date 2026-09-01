#!/usr/bin/env python3
"""Download/cache the locked COD test sets and create their four manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gdown

from cod_ssl.data.bootstrap import (
    STANDARD_TEST_COUNTS,
    build_test_manifest,
    discover_dataset_pair,
    extract_archive,
)
from cod_ssl.data.exclusions import exclude_manifest_rows, load_exclusion_stems

# Official SINet archive: CAMO-Test, COD10K-Test and CHAMELEON.
SINET_TEST_FILE_ID = "1QEGnP9O7HbN_2tH999O3HRIsErIVYalx"
# Public NC4K release mirrored by the COD dataset index referenced by SINet-V2.
NC4K_FILE_ID = "1kzpX_U3gbgO9MuwZIWTuRVpiB7V6yrAQ"


def download(file_id: str, destination: Path) -> None:
    if destination.is_file():
        print(f"Using cached archive: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=file_id, output=str(destination), quiet=False)
    if result is None or not destination.is_file():
        raise RuntimeError(f"Google Drive download failed or quota was exceeded: {file_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-dir", default="manifests")
    parser.add_argument("--accept-noncommercial-license", action="store_true")
    parser.add_argument("--exclusions", default="configs/dataset_exclusions.txt")
    args = parser.parse_args()
    if not args.accept_noncommercial_license:
        raise PermissionError(
            "COD10K is non-commercial; pass --accept-noncommercial-license only after reviewing its license"
        )

    root = Path(args.data_root)
    extracted = root / "standard_tests"
    archives = {
        "sinet_tests": (
            SINET_TEST_FILE_ID, root / "archives" / "sinet_tests.zip", (250, 2026, 76)
        ),
        "nc4k": (NC4K_FILE_ID, root / "archives" / "nc4k.zip", (4121,)),
    }
    for name, (file_id, archive, expected_counts) in archives.items():
        download(file_id, archive)
        destination = extracted / name
        try:
            for count in expected_counts:
                discover_dataset_pair(destination, count)
        except RuntimeError:
            print(f"Extracting {archive} to {destination}...")
            extract_archive(archive, destination)

    manifest_dir = Path(args.manifest_dir)
    exclusions = load_exclusion_stems(args.exclusions)
    exclusion_report = {}
    for dataset_name, expected in STANDARD_TEST_COUNTS.items():
        image_dir, mask_dir = discover_dataset_pair(extracted, expected)
        frame = build_test_manifest(
            dataset_name, image_dir, mask_dir, manifest_dir / f"{dataset_name}.csv"
        )
        frame, removed = exclude_manifest_rows(
            frame,
            exclusions,
            dataset_name=dataset_name,
            require_all=dataset_name == "cod10k_test",
        )
        frame.to_csv(manifest_dir / f"{dataset_name}.csv", index=False)
        exclusion_report[dataset_name] = removed
        print(f"{dataset_name}: {len(frame)} pairs -> {manifest_dir / f'{dataset_name}.csv'}")
    (manifest_dir / "test_exclusions.json").write_text(
        json.dumps(exclusion_report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
