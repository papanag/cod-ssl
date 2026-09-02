from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from cod_ssl.data.bootstrap import IMAGE_SUFFIXES


@dataclass(frozen=True)
class OriginalSequence:
    sequence_id: str
    directory: Path
    frames: dict[int, Path]


@dataclass(frozen=True)
class MaskSequence:
    sequence_id: str
    official_split: str
    directory: Path
    images: dict[int, Path]
    masks: dict[int, Path]
    cadence_exceptions: tuple[tuple[int, int], ...]


def _inside(path: Path, root: Path) -> Path:
    resolved, resolved_root = path.resolve(), root.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"path escapes configured root: {path}")
    return resolved


def unique_named_directory(root: Path, name: str) -> Path:
    root = root.resolve()
    matches = [_inside(path, root) for path in root.rglob(name) if path.is_dir() and path.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {name!r} directory under {root}, found {len(matches)}")
    return matches[0]


def numeric_files(directory: Path, *, suffixes: set[str] = IMAGE_SUFFIXES) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing directory: {directory}")
    result: dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            number = int(path.stem)
        except ValueError as error:
            raise ValueError(f"non-numeric frame stem: {path}") from error
        if number in result:
            raise ValueError(f"duplicate numeric frame stem {number}: {directory}")
        resolved = path.resolve()
        if not resolved.is_relative_to(directory.resolve()):
            raise ValueError(f"raw asset symlink escapes its sequence directory: {path}")
        result[number] = resolved
    if not result:
        raise ValueError(f"no supported images in {directory}")
    return dict(sorted(result.items()))


def inventory_original_moca(
    root: str | Path,
    *,
    expected_sequences: int = 141,
    expected_frames: int = 37_250,
    verify_counts: bool = True,
) -> tuple[Path, dict[str, OriginalSequence]]:
    root = Path(root).resolve()
    images_root = unique_named_directory(root, "JPEGImages")
    sequences: dict[str, OriginalSequence] = {}
    directories = sorted(path for path in images_root.iterdir() if path.is_dir())
    for directory in tqdm(directories, desc="inventory Original MoCA", unit="sequence", dynamic_ncols=True):
        frames = numeric_files(directory, suffixes={".jpg", ".jpeg"})
        ids = list(frames)
        if ids != list(range(len(ids))):
            raise ValueError(f"Original MoCA sequence is not zero-based consecutive: {directory.name}")
        if directory.name.casefold() in {key.casefold() for key in sequences}:
            raise ValueError(f"case-insensitive Original MoCA sequence collision: {directory.name}")
        sequences[directory.name] = OriginalSequence(directory.name, directory.resolve(), frames)
    actual = (len(sequences), sum(len(sequence.frames) for sequence in sequences.values()))
    if verify_counts and actual != (expected_sequences, expected_frames):
        raise ValueError(f"Original MoCA counts differ: sequences={actual[0]}, frames={actual[1]}")
    return images_root, sequences


def _mask_quality(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    values = sorted(map(int, np.unique(array)))
    binary = set(values).issubset({0, 1, 255})
    return {
        "width": int(array.shape[1]), "height": int(array.shape[0]),
        "values": values, "binary": binary,
        "empty": bool(np.all(array == 0)), "full": bool(np.all(array > 0)),
    }


def inventory_moca_mask(
    root: str | Path,
    *,
    expected_train_sequences: int = 71,
    expected_test_sequences: int = 16,
    expected_targets: int = 4_691,
    verify_counts: bool = True,
    require_binary_masks: bool = True,
    mask_quality_workers: int = 16,
) -> tuple[dict[str, MaskSequence], list[dict[str, object]]]:
    root = Path(root).resolve()
    partitions = {
        "train": unique_named_directory(root, "TrainDataset_per_sq"),
        "test": unique_named_directory(root, "TestDataset_per_sq"),
    }
    sequences: dict[str, MaskSequence] = {}
    mask_quality: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=mask_quality_workers) as executor:
        for split, partition in partitions.items():
            directories = sorted(path for path in partition.iterdir() if path.is_dir())
            for directory in tqdm(
                directories, desc=f"inventory MoCA-Mask {split}", unit="sequence",
                dynamic_ncols=True,
            ):
                if directory.name in sequences:
                    raise ValueError(f"MoCA-Mask sequence occurs in both official splits: {directory.name}")
                images, masks = numeric_files(directory / "Imgs"), numeric_files(directory / "GT")
                if images.keys() != masks.keys():
                    raise ValueError(f"MoCA-Mask RGB/GT keys differ: {directory.name}")
                ids = list(images)
                exceptions = tuple((left, right) for left, right in zip(ids, ids[1:]) if right - left != 5)
                quality_rows = executor.map(_mask_quality, masks.values())
                for (number, mask_path), quality in zip(masks.items(), quality_rows):
                    quality |= {"sequence_id": directory.name, "frame_number": number}
                    if require_binary_masks and not quality["binary"]:
                        raise ValueError(f"non-binary MoCA-Mask target: {mask_path}")
                    mask_quality.append(quality)
                sequences[directory.name] = MaskSequence(
                    directory.name, split, directory.resolve(), images, masks, exceptions
                )
    actual = {
        "train_sequences": sum(sequence.official_split == "train" for sequence in sequences.values()),
        "test_sequences": sum(sequence.official_split == "test" for sequence in sequences.values()),
        "targets": sum(len(sequence.images) for sequence in sequences.values()),
    }
    expected = {
        "train_sequences": expected_train_sequences,
        "test_sequences": expected_test_sequences,
        "targets": expected_targets,
    }
    if verify_counts and actual != expected:
        raise ValueError(f"MoCA-Mask counts differ: {actual}")
    return sequences, mask_quality
