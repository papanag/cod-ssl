from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class TemporalAdapter(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, features: torch.Tensor, temporal_valid: torch.BoolTensor,
        target_index: int, *, state: Any | None = None,
    ) -> tuple[torch.Tensor, Any | None]: ...

    @abstractmethod
    def reset_state(self, batch_size: int, spatial_size: tuple[int, int]): ...


def validate_temporal_inputs(features: torch.Tensor, temporal_valid: torch.BoolTensor) -> None:
    if features.ndim != 5:
        raise ValueError(f"features must be [B,T,C,H,W], got {tuple(features.shape)}")
    if temporal_valid.dtype != torch.bool or tuple(temporal_valid.shape) != tuple(features.shape[:2]):
        raise ValueError("temporal_valid must be bool [B,T]")
