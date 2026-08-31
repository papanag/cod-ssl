from __future__ import annotations

from pathlib import Path

import pandas as pd


def create_dev_split(
    train_manifest: str | Path,
    train_output: str | Path,
    val_output: str | Path,
    *,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic, approximately source-stratified development split."""
    frame = pd.read_csv(train_manifest)
    train_parts, val_parts = [], []
    for _, group in frame.groupby("source", sort=True):
        shuffled = group.sample(frac=1, random_state=seed)
        val_count = round(len(group) * val_fraction)
        val_parts.append(shuffled.iloc[:val_count])
        train_parts.append(shuffled.iloc[val_count:])
    train = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    val = pd.concat(val_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    Path(train_output).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_output, index=False)
    val.to_csv(val_output, index=False)
    return train, val
