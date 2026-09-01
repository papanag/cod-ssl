from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import torch


@dataclass(frozen=True)
class ClipSpec:
    length: int
    stride: int
    target_index: int
    boundary_mode: Literal["replicate"] = "replicate"

    def __post_init__(self) -> None:
        if self.length < 1 or self.stride < 1:
            raise ValueError("clip length and stride must be positive")
        if not 0 <= self.target_index < self.length:
            raise ValueError("target_index must be inside the clip")
        if self.boundary_mode != "replicate":
            raise ValueError("only deterministic replicate padding is supported")


class ClipSampler:
    """Sample sequence positions; filenames and numeric frame gaps are irrelevant."""

    def source_indices(
        self,
        ordered_frame_numbers: Sequence[int],
        target_position: int,
        spec: ClipSpec,
    ) -> tuple[list[int], torch.BoolTensor]:
        if not ordered_frame_numbers:
            raise ValueError("cannot sample an empty video")
        if not 0 <= target_position < len(ordered_frame_numbers):
            raise IndexError("target_position is outside this video")
        if any(b <= a for a, b in pairwise(ordered_frame_numbers)):
            raise ValueError("frame numbers must be strictly chronological")
        raw = [target_position + (slot - spec.target_index) * spec.stride for slot in range(spec.length)]
        valid = torch.tensor([0 <= position < len(ordered_frame_numbers) for position in raw], dtype=torch.bool)
        positions = [min(max(position, 0), len(ordered_frame_numbers) - 1) for position in raw]
        return positions, valid
