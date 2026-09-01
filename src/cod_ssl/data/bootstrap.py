from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

STANDARD_TEST_COUNTS = {
    "camo_test": 250,
    "cod10k_test": 2026,
    "chameleon": 76,
    "nc4k": 4121,
}


def _safe_destination(root: Path, member_name: str) -> Path:
    destination = (root / member_name).resolve()
    if root.resolve() not in destination.parents and destination != root.resolve():
        raise ValueError(f"archive contains unsafe path: {member_name}")
    return destination


def extract_archive(archive: str | Path, destination: str | Path) -> None:
    archive, destination = Path(archive), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            members = handle.infolist()
            for member in members:
                _safe_destination(destination, member.filename)
            for member in tqdm(
                members,
                desc=f"extract {archive.name}",
                unit="file",
                dynamic_ncols=True,
            ):
                handle.extract(member, destination)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            members = handle.getmembers()
            for member in members:
                _safe_destination(destination, member.name)
            for member in tqdm(
                members,
                desc=f"extract {archive.name}",
                unit="file",
                dynamic_ncols=True,
            ):
                handle.extract(member, destination, filter="data")
        return
    raise ValueError(f"unsupported archive: {archive}")


def _files_by_stem(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def discover_standard_training_pair(root: str | Path) -> tuple[Path, Path]:
    """Find an unambiguous 4,040-pair image/object-mask directory pair."""
    root = Path(root)
    candidates = []
    for directory in [root, *(path for path in root.rglob("*") if path.is_dir())]:
        files = _files_by_stem(directory)
        if len(files) == 4040:
            candidates.append((directory, files))
    matches: list[tuple[int, Path, Path]] = []
    for image_dir, image_files in candidates:
        for mask_dir, mask_files in candidates:
            if image_dir == mask_dir or image_files.keys() != mask_files.keys():
                continue
            image_name, mask_name = str(image_dir).lower(), str(mask_dir).lower()
            image_score = int("img" in image_name or "image" in image_name)
            mask_score = 3 * int("gt_object" in mask_name) + 2 * int("mask" in mask_name)
            mask_score += int("gt" in mask_name) - 4 * int("edge" in mask_name)
            score = image_score + mask_score
            if score > 0:
                matches.append((score, image_dir, mask_dir))
    if not matches:
        raise RuntimeError(f"could not find paired 4,040-image training directories under {root}")
    matches.sort(key=lambda item: (-item[0], str(item[1]), str(item[2])))
    best_score = matches[0][0]
    best = {(image_dir, mask_dir) for score, image_dir, mask_dir in matches if score == best_score}
    if len(best) != 1:
        raise RuntimeError(f"ambiguous 4,040-pair training layout: {sorted(map(str, best))}")
    return next(iter(best))


def build_standard_train_manifest(
    image_dir: str | Path, mask_dir: str | Path, output: str | Path
) -> pd.DataFrame:
    images, masks = _files_by_stem(Path(image_dir)), _files_by_stem(Path(mask_dir))
    if images.keys() != masks.keys() or len(images) != 4040:
        raise ValueError("standard training directories must contain the same 4,040 stems")
    rows = []
    for stem in sorted(images):
        source = "cod10k" if stem.upper().startswith("COD10K") else "camo"
        rows.append(
            {"id": stem, "source": source, "image_path": str(images[stem]), "mask_path": str(masks[stem])}
        )
    frame = pd.DataFrame(rows)
    counts = frame.groupby("source").size().to_dict()
    if counts != {"camo": 1000, "cod10k": 3040}:
        raise ValueError(f"unexpected source counts inferred from official filenames: {counts}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def discover_dataset_pair(root: str | Path, expected_count: int) -> tuple[Path, Path]:
    """Find one unambiguous image/GT directory pair with matching stems."""
    root = Path(root)
    candidates: list[tuple[Path, dict[str, Path]]] = []
    for directory in (path for path in root.rglob("*") if path.is_dir()):
        files = _files_by_stem(directory)
        if len(files) == expected_count:
            candidates.append((directory, files))
    matches: list[tuple[int, Path, Path]] = []
    for image_dir, images in candidates:
        for mask_dir, masks in candidates:
            if image_dir == mask_dir or images.keys() != masks.keys():
                continue
            image_name, mask_name = str(image_dir).lower(), str(mask_dir).lower()
            score = 2 * int("image" in image_name or "img" in image_name)
            score += 3 * int("gt" in mask_name) + 2 * int("mask" in mask_name)
            score -= 5 * int("edge" in mask_name)
            if score > 0:
                matches.append((score, image_dir, mask_dir))
    if not matches:
        raise RuntimeError(f"could not find a paired {expected_count}-image dataset under {root}")
    matches.sort(key=lambda item: (-item[0], str(item[1]), str(item[2])))
    best_score = matches[0][0]
    best = {(images, masks) for score, images, masks in matches if score == best_score}
    if len(best) != 1:
        raise RuntimeError(f"ambiguous {expected_count}-pair dataset layout: {sorted(map(str, best))}")
    return next(iter(best))


def build_test_manifest(
    dataset_name: str,
    image_dir: str | Path,
    mask_dir: str | Path,
    output: str | Path,
) -> pd.DataFrame:
    if dataset_name not in STANDARD_TEST_COUNTS:
        raise ValueError(f"unknown standard test dataset: {dataset_name}")
    images, masks = _files_by_stem(Path(image_dir)), _files_by_stem(Path(mask_dir))
    expected = STANDARD_TEST_COUNTS[dataset_name]
    if images.keys() != masks.keys() or len(images) != expected:
        raise ValueError(f"{dataset_name} directories must contain the same {expected} stems")
    frame = pd.DataFrame(
        {
            "id": stem,
            "source": dataset_name,
            "image_path": str(images[stem]),
            "mask_path": str(masks[stem]),
        }
        for stem in sorted(images)
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
