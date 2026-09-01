from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class VideoSampleMeta:
    dataset: str
    regime: str | None
    split: str
    video_id: str
    source_video_id: str
    frame_id: str
    frame_number: int
    source_frame_indices: tuple[int, ...]
    target_index: int
    fps: float | None
    timestamps_sec: tuple[float, ...] | None
    annotation_type: str
    attributes: dict[str, Any]


class VideoCODDataset(torch.utils.data.Dataset, ABC):
    @property
    @abstractmethod
    def video_ids(self) -> tuple[str, ...]: ...

    @abstractmethod
    def frames_for_video(self, video_id: str) -> tuple[int, ...]: ...


REQUIRED_SAMPLE_KEYS = frozenset({
    "frames", "target_mask", "target_index", "video_id", "source_video_id",
    "frame_id", "frame_number", "source_frame_indices", "timestamps_sec", "fps",
    "dataset", "regime", "split", "annotation_type", "valid_temporal_mask",
    "attributes", "metadata",
})


def validate_video_sample(sample: dict[str, Any]) -> None:
    missing = REQUIRED_SAMPLE_KEYS - sample.keys()
    if missing:
        raise ValueError(f"video sample missing fields: {sorted(missing)}")
    frames, mask, valid = sample["frames"], sample["target_mask"], sample["valid_temporal_mask"]
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"frames must be [T,3,H,W], got {tuple(frames.shape)}")
    if tuple(mask.shape) != (1, frames.shape[-2], frames.shape[-1]):
        raise ValueError("target mask must be [1,H,W] and aligned with frames")
    if valid.dtype != torch.bool or tuple(valid.shape) != (frames.shape[0],):
        raise ValueError("valid_temporal_mask must be bool [T]")
    if len(sample["source_frame_indices"]) != frames.shape[0]:
        raise ValueError("source_frame_indices must contain one index per clip frame")
