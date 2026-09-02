#!/usr/bin/env python3
"""Download/cache public VCOD releases and build canonical research manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import gdown
from tqdm.auto import tqdm

from cod_ssl.data.bootstrap import extract_archive, extract_zip_prefixes, merge_tree_parallel
from cod_ssl.data.camotion_attributes import parse_camotion_attributes
from cod_ssl.data.camotion_bootstrap import (
    build_camotion_manifest,
    discover_camotion_roots,
    verify_camotion_flattened_segmentation_duplicates,
)
from cod_ssl.data.preprocessing.prepare_moca_mask_dense import build_moca_mask_dense
from cod_ssl.data.preprocessing.moca_manifest_schema import write_checksums, write_json
from cod_ssl.utils.run import file_sha256

MOCA_RELEASES = {
    "manual": {
        "file_id": "1FB24BGVrPOeUpmYbKZJYL5ermqUvBo_6",
        "archive": "MoCA-Mask.zip",
    },
    "original": {
        "url": "https://thor.robots.ox.ac.uk/datasets/MoCA/MoCA.zip",
        "archive": "MoCA.zip",
    },
}
OFFICIAL_SOURCE = "https://github.com/XuelianCheng/SLT-Net"
ORIGINAL_MOCA_SOURCE = "https://www.robots.ox.ac.uk/~vgg/data/MoCA/"
CAMOTION_FILE_ID = "1YzNdlDhsfgXTZ-Ya1w9wn3SjTXwU2xFs"
CAMOTION_ARCHIVE_BYTES = 16_912_530_524
CAMOTION_REPOSITORY = "https://github.com/Garyson1204/CAMotion"
CAMOTION_METADATA_COMMIT = "bf92692f9f9f2820185f9aa9a06fd2891dadf9a7"
CAMOTION_ATTRIBUTES_SHA256 = "6ad95102a836ef5a199e6e0a642ee7ddfbf2f6d8065c40742014cfab934abcd9"
CAMOTION_ATTRIBUTES_URL = (
    f"https://raw.githubusercontent.com/Garyson1204/CAMotion/"
    f"{CAMOTION_METADATA_COMMIT}/attributes_per_sequence.txt"
)


def sha256_with_progress(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle, tqdm(
        total=path.stat().st_size, desc=f"hash {path.name}", unit="B", unit_scale=True,
        unit_divisor=1024, dynamic_ncols=True,
    ) as progress:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
            progress.update(len(block))
    return digest.hexdigest()


def _download(
    file_id: str, destination: Path, *, expected_bytes: int | None = None,
) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    if destination.is_file():
        actual_bytes = destination.stat().st_size
        if expected_bytes is None or actual_bytes == expected_bytes:
            print(f"Using cached archive: {destination}")
            return
        if actual_bytes > expected_bytes:
            raise ValueError(
                f"cached archive is larger than expected and cannot be resumed: {destination} "
                f"({actual_bytes} bytes; expected {expected_bytes})"
            )
        # An interrupted older bootstrap may have left its partial download at the
        # final path. Move the most complete copy back to gdown's resumable path.
        if not temporary.is_file() or actual_bytes > temporary.stat().st_size:
            destination.replace(temporary)
        print(
            f"Resuming incomplete archive: {destination} "
            f"({temporary.stat().st_size}/{expected_bytes} bytes)"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(
        id=file_id,
        output=str(temporary),
        quiet=False,
        resume=temporary.is_file(),
    )
    if result is None or not temporary.is_file():
        raise RuntimeError(f"Google Drive download failed or quota was exceeded: {file_id}")
    if expected_bytes is not None and temporary.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"Google Drive download is incomplete; rerun to resume: {temporary} "
            f"({temporary.stat().st_size}/{expected_bytes} bytes)"
        )
    temporary.replace(destination)


def _download_http(url: str, destination: Path) -> None:
    if destination.is_file():
        print(f"Using cached archive: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    offset = temporary.stat().st_size if temporary.is_file() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request) as response:  # noqa: S310 - official pinned HTTPS dataset URL
        resumed = offset > 0 and response.status == 206
        if offset and not resumed:
            offset = 0
        total = response.headers.get("Content-Length")
        total = None if total is None else int(total) + offset
        with temporary.open("ab" if resumed else "wb") as handle, tqdm(
            total=total, initial=offset, desc=f"download {destination.name}", unit="B",
            unit_scale=True, unit_divisor=1024, dynamic_ncols=True,
        ) as progress:
            while block := response.read(8 * 1024 * 1024):
                handle.write(block)
                progress.update(len(block))
    temporary.replace(destination)


def _extract_staged(
    archive: Path,
    destination: Path,
    staging_root: Path | None,
    *,
    prefixes: tuple[str, ...] | None = None,
    required_path_parts: set[str] | None = None,
) -> None:
    if staging_root is None:
        if prefixes is None:
            extract_archive(archive, destination)
        else:
            extract_zip_prefixes(
                archive, destination, prefixes,
                required_path_parts=required_path_parts,
            )
        return
    staged = staging_root / destination.name
    print(f"Extracting {archive.name} on local storage: {staged}")
    if prefixes is None:
        extract_archive(archive, staged)
    else:
        extract_zip_prefixes(
            archive, staged, prefixes,
            required_path_parts=required_path_parts,
        )
    report = merge_tree_parallel(staged, destination)
    print(f"Persistent copy complete: {report}")
    shutil.rmtree(staged)


def _ensure_extracted(
    archive: Path, destination: Path, release_kind: str, staging_root: Path | None,
) -> None:
    expected = ("TrainDataset_per_sq", "TestDataset_per_sq")
    try:
        for name in expected:
            matches = [path for path in destination.rglob(name) if path.is_dir() and path.name == name]
            if len(matches) != 1:
                raise RuntimeError(f"expected one extracted {name} directory")
        return
    except RuntimeError:
        pass
    print(f"Extracting {archive} to {destination}...")
    _extract_staged(archive, destination, staging_root)


def _ensure_original_moca_extracted(
    archive: Path, destination: Path, staging_root: Path | None,
) -> None:
    marker = destination / ".selected_extraction_complete"
    if marker.is_file() and any(path.is_dir() for path in destination.rglob("JPEGImages")):
        return
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"official Original MoCA download is not a ZIP archive: {archive}")
    _extract_staged(
        archive, destination, staging_root,
        prefixes=("MoCA/JPEGImages/", "MoCA/Annotations/"),
    )
    if not any(path.is_dir() for path in destination.rglob("JPEGImages")):
        raise ValueError("Original MoCA archive did not expose the expected JPEGImages tree")
    marker.write_text("complete\n")


def _download_metadata(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.is_file() and file_sha256(destination) == expected_sha256:
        print(f"Using pinned CAMotion metadata: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:  # noqa: S310 - pinned HTTPS URL + checksum
        total = int(response.headers.get("Content-Length", 0)) or None
        digest = hashlib.sha256()
        temporary = destination.with_suffix(destination.suffix + ".part")
        with temporary.open("wb") as handle, tqdm(
            total=total, desc="download CAMotion attributes", unit="B", unit_scale=True,
            dynamic_ncols=True,
        ) as progress:
            while block := response.read(64 * 1024):
                handle.write(block)
                digest.update(block)
                progress.update(len(block))
    if digest.hexdigest() != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError("CAMotion attribute metadata checksum differs from pinned official commit")
    temporary.replace(destination)


def _ensure_camotion_extracted(
    archive: Path, destination: Path, staging_root: Path | None,
) -> None:
    marker = destination / ".selected_extraction_complete"
    if marker.is_file():
        discover_camotion_roots(destination)
        return
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"official CAMotion download is not a ZIP archive: {archive}")
    prefixes = (
        "CAMotion/CAMotion/TrainDataset_per_sq/",
        "CAMotion/CAMotion/TestDataset_per_sq/",
    )
    _extract_staged(
        archive, destination, staging_root, prefixes=prefixes,
        required_path_parts={"Imgs", "GT"},
    )
    discover_camotion_roots(destination)
    marker.write_text("complete\n")


def _cached_archive_sha256(archive: Path, cache: Path, key: str) -> str:
    cached = json.loads(cache.read_text()) if cache.is_file() else {}
    fingerprint = {"bytes": archive.stat().st_size, "mtime_ns": archive.stat().st_mtime_ns}
    entry = cached.get(key, {})
    if entry.get("fingerprint") == fingerprint and entry.get("sha256"):
        print(f"Using cached SHA-256 for {archive.name}")
        return entry["sha256"]
    digest = sha256_with_progress(archive)
    cached[key] = {"fingerprint": fingerprint, "sha256": digest}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(cached, indent=2) + "\n")
    return digest


def _bootstrap_camotion(args, root: Path, manifest_dir: Path) -> dict:
    if not args.accept_camotion_academic_license:
        raise PermissionError(
            "CAMotion is academic-research-only; pass --accept-camotion-academic-license "
            "after reviewing the official usage notice"
        )
    archive = root / "archives" / "CAMotion.zip"
    extracted = root / "camotion"
    metadata = root / "metadata" / f"attributes_per_sequence_{CAMOTION_METADATA_COMMIT}.txt"
    progress = tqdm(total=5, desc="bootstrap CAMotion", unit="stage", dynamic_ncols=True)
    _download(CAMOTION_FILE_ID, archive, expected_bytes=CAMOTION_ARCHIVE_BYTES)
    progress.update(1)
    _download_metadata(CAMOTION_ATTRIBUTES_URL, metadata, CAMOTION_ATTRIBUTES_SHA256)
    progress.update(1)
    _ensure_camotion_extracted(archive, extracted, args.staging_root)
    progress.update(1)
    duplicate_report = verify_camotion_flattened_segmentation_duplicates(archive)
    attributes = parse_camotion_attributes(metadata.read_text().splitlines())
    manifest = manifest_dir / "camotion.csv"
    frame, split_report = build_camotion_manifest(
        extracted, attributes, manifest,
        validation_fraction=args.validation_fraction, seed=args.seed,
    )
    progress.update(1)
    archive_sha256 = (
        None if args.skip_archive_sha256 else
        _cached_archive_sha256(archive, root / "archives" / "vcod_archive_hashes.json", "camotion")
    )
    progress.update(1)
    progress.close()
    attribute_manifest = {
        "schema_version": 1, "attribute_scope": "sequence",
        "attribute_codes": list(next(iter(attributes.values())).keys()),
        "sequences": attributes,
        "source_url": CAMOTION_ATTRIBUTES_URL,
        "repository_commit": CAMOTION_METADATA_COMMIT,
        "sha256": CAMOTION_ATTRIBUTES_SHA256,
    }
    attribute_path = manifest_dir / "camotion.attributes.json"
    attribute_path.write_text(json.dumps(attribute_manifest, indent=2) + "\n")
    receipt = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "official_source": CAMOTION_REPOSITORY,
        "usage": "academic_research_only",
        "archive": {"google_drive_file_id": CAMOTION_FILE_ID, "path": str(archive.resolve()),
                    "bytes": archive.stat().st_size, "sha256": archive_sha256},
        "metadata": {"path": str(metadata.resolve()), "source_url": CAMOTION_ATTRIBUTES_URL,
                     "repository_commit": CAMOTION_METADATA_COMMIT,
                     "sha256": CAMOTION_ATTRIBUTES_SHA256},
        "manifest": str(manifest.resolve()), "manifest_sha256": file_sha256(manifest),
        "target_manifest_sha256": file_sha256(manifest),
        "split_manifest_sha256": file_sha256(manifest.with_suffix(".splits.json")),
        "context_policy": "released_sequence_rgb_frames_only_no_dense_rgb_archive_available",
        "release_profile": "camotion_public_stride5_v1",
        "flattened_export_verification": duplicate_report,
        "attribute_manifest": str(attribute_path.resolve()),
        "attribute_manifest_sha256": file_sha256(attribute_path),
        "rows": len(frame), "splits": split_report,
    }
    manifest.with_suffix(".bootstrap.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument(
        "--staging-root", type=Path,
        help="Fast local directory used before a resumable parallel copy to persistent storage",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-archive-sha256", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=("moca_mask_dense", "camotion"),
                        default=("moca_mask_dense", "camotion"))
    parser.add_argument("--accept-camotion-academic-license", action="store_true")
    args = parser.parse_args()
    if "camotion" in args.datasets and not args.accept_camotion_academic_license:
        raise PermissionError(
            "CAMotion is academic-research-only; pass --accept-camotion-academic-license "
            "after reviewing the official usage notice"
        )

    root = Path(args.data_root)
    archives = root / "archives"
    manifest_dir = Path(args.manifest_dir)
    if args.staging_root is not None:
        args.staging_root.mkdir(parents=True, exist_ok=True)
    if "moca_mask_dense" not in args.datasets:
        receipt = _bootstrap_camotion(args, root, manifest_dir)
        print(json.dumps(receipt, indent=2))
        return
    releases = root / "moca"
    processed = root / "processed" / "moca_mask_dense_v1"
    progress = tqdm(total=7, desc="bootstrap MoCA dense", unit="stage", dynamic_ncols=True)
    manual_archive = archives / MOCA_RELEASES["manual"]["archive"]
    original_archive = archives / MOCA_RELEASES["original"]["archive"]
    _download(MOCA_RELEASES["manual"]["file_id"], manual_archive)
    progress.update(1)
    _download_http(MOCA_RELEASES["original"]["url"], original_archive)
    progress.update(1)
    _ensure_extracted(
        manual_archive, releases / "moca_mask_public", "manual", args.staging_root,
    )
    progress.update(1)
    _ensure_original_moca_extracted(
        original_archive, releases / "original_moca", args.staging_root,
    )
    progress.update(1)
    build_summary = build_moca_mask_dense(
        "configs/datasets/moca_mask_dense.yaml",
        original_moca_root=releases / "original_moca",
        moca_mask_root=releases / "moca_mask_public",
        output_root=processed,
        materialization="manifest_only",
    )
    progress.update(1)
    hashes = {}
    if not args.skip_archive_sha256:
        hash_cache = archives / "moca_archive_hashes.json"
        cached = json.loads(hash_cache.read_text()) if hash_cache.is_file() else {}
        for kind, archive in {"manual": manual_archive, "original": original_archive}.items():
            fingerprint = {"bytes": archive.stat().st_size, "mtime_ns": archive.stat().st_mtime_ns}
            entry = cached.get(kind, {})
            if entry.get("fingerprint") == fingerprint and entry.get("sha256"):
                hashes[kind] = entry["sha256"]
                print(f"Using cached SHA-256 for {archive.name}")
            else:
                hashes[kind] = sha256_with_progress(archive)
            cached[kind] = {"fingerprint": fingerprint, "sha256": hashes[kind]}
        hash_cache.write_text(json.dumps(cached, indent=2) + "\n")
    release_path = processed / "manifest" / "release_manifest.json"
    release_manifest = json.loads(release_path.read_text())
    release_manifest["original_moca"].update({
        "archive_sha256": hashes.get("original"),
        "archive_size_bytes": original_archive.stat().st_size,
        "source_url": MOCA_RELEASES["original"]["url"],
    })
    release_manifest["moca_mask"].update({
        "archive_sha256": hashes.get("manual"),
        "archive_size_bytes": manual_archive.stat().st_size,
        "google_drive_file_id": MOCA_RELEASES["manual"]["file_id"],
    })
    write_json(release_path, release_manifest)
    build_summary["manifest_checksums"] = write_checksums(processed / "manifest")
    write_json(processed / "audit" / "summary.json", build_summary)
    progress.update(1)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pointer = manifest_dir / "moca_mask_dense.json"
    pointer.write_text(json.dumps({
        "dataset_build_id": "moca_mask_dense_v1",
        "processed_root": str(processed.resolve()),
        "runtime_manifest": str((processed / "manifest" / "runtime_manifest.csv").resolve()),
        "manifest_checksums": str((processed / "manifest" / "manifest_checksums.sha256").resolve()),
    }, indent=2) + "\n")
    progress.update(1)
    progress.close()
    receipt = {
        "schema_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "official_sources": {"moca_mask": OFFICIAL_SOURCE, "original_moca": ORIGINAL_MOCA_SOURCE},
        "releases": {
            "manual": {"google_drive_file_id": MOCA_RELEASES["manual"]["file_id"],
                       "archive": str(manual_archive.resolve()), "bytes": manual_archive.stat().st_size,
                       "sha256": hashes.get("manual")},
            "original": {"url": MOCA_RELEASES["original"]["url"],
                         "archive": str(original_archive.resolve()), "bytes": original_archive.stat().st_size,
                         "sha256": hashes.get("original")},
        },
        "processed_root": str(processed.resolve()),
        "runtime_manifest": str((processed / "manifest" / "runtime_manifest.csv").resolve()),
        "build": build_summary,
    }
    receipt_path = manifest_dir / "moca_mask_dense.bootstrap.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if "camotion" in args.datasets:
        print(json.dumps(_bootstrap_camotion(args, root, manifest_dir), indent=2))


if __name__ == "__main__":
    main()
