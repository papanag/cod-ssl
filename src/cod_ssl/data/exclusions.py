from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def _stem(value: str) -> str:
    return Path(value.strip()).stem


def load_exclusion_policy(path: str | Path) -> dict[str, set[str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset exclusion policy not found: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"train_id", "test_id", "detection"}
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"dataset exclusion policy must have columns {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"dataset exclusion policy is empty: {path}")
    if any(not row["train_id"].strip() or not row["test_id"].strip() for row in rows):
        raise ValueError("every exclusion pair requires both a train_id and test_id")
    train = {_stem(row["train_id"]) for row in rows}
    test = {_stem(row["test_id"]) for row in rows}
    if len(train) != len(rows) or len(test) != len(rows):
        raise ValueError("exclusion policy contains duplicate train or test IDs")
    return {"train_all": train, "cod10k_test": test}


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
