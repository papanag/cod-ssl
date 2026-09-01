from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

_COD10K_CATEGORY = re.compile(
    r"^COD10K-CAM-\d+-(?P<supercategory>[^-]+)-\d+-(?P<category>.+)-\d+$",
    re.IGNORECASE,
)


def cod10k_category(sample_id: str) -> str:
    """Extract the published COD10K supercategory/category encoded in its ID."""
    match = _COD10K_CATEGORY.match(str(sample_id))
    if match is None:
        raise ValueError(f"cannot derive COD10K category from sample id: {sample_id}")
    return f"{match.group('supercategory')}/{match.group('category')}"


def _proportional_allocation(
    counts: pd.Series, total: int, *, ensure_each: bool = True
) -> pd.Series:
    """Largest-remainder allocation with one sample per stratum when possible."""
    counts = counts.sort_index().astype(int)
    if total < 1 or total > int(counts.sum()):
        raise ValueError(f"sample size {total} is invalid for {int(counts.sum())} rows")
    minimum = pd.Series(0, index=counts.index, dtype=int)
    if ensure_each and total >= len(counts):
        minimum[:] = 1
    remaining = total - int(minimum.sum())
    capacity = counts - minimum
    if remaining == 0:
        return minimum
    quotas = capacity / capacity.sum() * remaining
    allocation = minimum + quotas.astype(int)
    leftover = total - int(allocation.sum())
    order = pd.DataFrame(
        {"remainder": quotas - quotas.astype(int), "capacity": counts - allocation},
        index=counts.index,
    )
    order = order[order.capacity > 0].sort_values(
        ["remainder", "capacity"], ascending=[False, False], kind="stable"
    )
    for stratum in order.index[:leftover]:
        allocation.loc[stratum] += 1
    if int(allocation.sum()) != total or (allocation > counts).any():
        raise RuntimeError("failed to allocate the requested stratified sample")
    return allocation


def create_stratified_smoke_manifest(
    train_manifest: str | Path,
    output: str | Path,
    *,
    size: int = 256,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a reproducible source- and COD10K-category-stratified smoke set."""
    frame = pd.read_csv(train_manifest)
    required = {"id", "source", "image_path", "mask_path"}
    if not required.issubset(frame.columns):
        raise ValueError(f"manifest missing columns: {sorted(required - set(frame.columns))}")
    source_counts = frame.groupby("source").size()
    if set(source_counts.index) != {"camo", "cod10k"}:
        raise ValueError(f"expected CAMO and COD10K sources, got {source_counts.to_dict()}")
    source_allocation = _proportional_allocation(source_counts, size, ensure_each=False)
    rng_seed = int(seed)
    selected_parts = []

    camo = frame[frame.source == "camo"].copy()
    camo["smoke_stratum"] = "camo/unlabelled"
    selected_parts.append(camo.sample(n=int(source_allocation.camo), random_state=rng_seed))

    cod10k = frame[frame.source == "cod10k"].copy()
    cod10k["smoke_stratum"] = cod10k.id.map(cod10k_category)
    category_counts = cod10k.groupby("smoke_stratum").size()
    category_allocation = _proportional_allocation(
        category_counts, int(source_allocation.cod10k)
    )
    for offset, (category, count) in enumerate(category_allocation.items()):
        group = cod10k[cod10k.smoke_stratum == category]
        selected_parts.append(group.sample(n=int(count), random_state=rng_seed + offset + 1))

    selected = pd.concat(selected_parts).sample(frac=1, random_state=rng_seed).reset_index(drop=True)
    if len(selected) != size or selected.id.duplicated().any():
        raise RuntimeError("stratified smoke selection is not the requested unique size")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected[["id", "source", "image_path", "mask_path"]].to_csv(output, index=False)
    report = (
        selected.groupby(["source", "smoke_stratum"], sort=True)
        .size().rename("selected").reset_index()
    )
    return selected, report


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
