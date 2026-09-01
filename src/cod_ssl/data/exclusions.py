from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_exclusion_stems(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset exclusion policy not found: {path}")
    stems = {
        Path(line.strip()).stem
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not stems:
        raise ValueError(f"dataset exclusion policy is empty: {path}")
    return stems


def exclude_manifest_rows(
    frame: pd.DataFrame,
    exclusion_stems: set[str],
    *,
    dataset_name: str,
    require_all: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    if "id" not in frame.columns:
        raise ValueError("manifest has no id column")
    normalized_ids = frame.id.astype(str).map(lambda value: Path(value).stem)
    removed = sorted(set(normalized_ids[normalized_ids.isin(exclusion_stems)]))
    if require_all and set(removed) != exclusion_stems:
        missing = sorted(exclusion_stems - set(removed))
        raise ValueError(f"{dataset_name} is missing configured overlap exclusions: {missing}")
    clean = frame.loc[~normalized_ids.isin(exclusion_stems)].reset_index(drop=True)
    if clean.id.astype(str).map(lambda value: Path(value).stem).isin(exclusion_stems).any():
        raise RuntimeError(f"{dataset_name} still contains an excluded ID")
    return clean, removed
