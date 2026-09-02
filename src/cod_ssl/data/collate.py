from __future__ import annotations

from collections.abc import Mapping

from torch.utils.data._utils.collate import default_collate


def video_collate(batch):
    """Default PyTorch collation extended for optional video metadata."""
    first = batch[0]
    if first is None:
        if not all(value is None for value in batch):
            raise ValueError("optional video metadata must be uniformly present within a batch")
        return None
    if isinstance(first, Mapping):
        if any(value.keys() != first.keys() for value in batch):
            raise ValueError("video metadata mappings have inconsistent keys")
        return {key: video_collate([value[key] for value in batch]) for key in first}
    return default_collate(batch)
