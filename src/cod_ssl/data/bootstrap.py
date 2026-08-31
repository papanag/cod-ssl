from __future__ import annotations

from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


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
    for directory in (path for path in root.rglob("*") if path.is_dir()):
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

