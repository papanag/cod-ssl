from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class DenseFeatureBatch:
    features: torch.Tensor  # B,T,C,Hf,Wf
    temporal_valid: torch.BoolTensor  # B,T
    spatial_size: tuple[int, int]
    source_frame_intervals: tuple[tuple[int, int], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.features.ndim != 5:
            raise ValueError(f"dense features must be [B,T,C,Hf,Wf], got {tuple(self.features.shape)}")
        if self.temporal_valid.dtype != torch.bool or tuple(self.temporal_valid.shape) != tuple(self.features.shape[:2]):
            raise ValueError("temporal_valid must be bool [B,T]")
        if self.spatial_size != tuple(self.features.shape[-2:]):
            raise ValueError("spatial_size does not match dense features")
        if len(self.source_frame_intervals) != self.features.shape[1]:
            raise ValueError("one source-frame interval is required per temporal feature")


class FrozenBackbone(nn.Module, ABC):
    input_size = 384
    grid_size = 24
    layer_indices = (2, 5, 8, 11)

    def configure_layers(self, layers: Sequence[int] | None) -> None:
        selected = tuple(self.layer_indices if layers is None else map(int, layers))
        if len(selected) not in {4, 12} or len(set(selected)) != len(selected):
            raise ValueError("backbone extraction requires four or twelve unique layers")
        if tuple(sorted(selected)) != selected or selected[0] < 0 or selected[-1] > 11:
            raise ValueError("backbone layers must be increasing indices in [0, 11]")
        self.layer_indices = selected

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> FrozenBackbone:
        # A frozen feature extractor must remain in evaluation mode even when a
        # containing model is switched to train mode.
        return super().train(False)

    @property
    @abstractmethod
    def feature_dims(self) -> list[int]: ...

    @abstractmethod
    def forward_features(self, images: torch.Tensor) -> list[torch.Tensor]: ...

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        return self.forward_features(images)

    def validate_features(
        self, images: torch.Tensor, features: Sequence[torch.Tensor]
    ) -> list[torch.Tensor]:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 384, 384):
            raise ValueError(f"expected input [B,3,384,384], got {tuple(images.shape)}")
        if len(features) != len(self.feature_dims):
            raise RuntimeError(
                f"expected {len(self.feature_dims)} features, got {len(features)}"
            )
        checked = list(features)
        for index, (feature, channels) in enumerate(zip(checked, self.feature_dims)):
            expected = (images.shape[0], channels, 24, 24)
            if tuple(feature.shape) != expected:
                raise RuntimeError(
                    f"feature {index} has shape {tuple(feature.shape)}; expected {expected}"
                )
            if feature.requires_grad:
                raise RuntimeError("frozen feature unexpectedly requires gradients")
        return checked


def build_backbone(name: str, **kwargs: object) -> FrozenBackbone:
    if name == "dinov3_vitb16":
        from cod_ssl.backbones.dinov3 import DINOv3ViTB16
        return DINOv3ViTB16(**kwargs)
    if name == "vjepa21_vitb16":
        from cod_ssl.backbones.vjepa21 import VJEPA21ViTB16
        return VJEPA21ViTB16(**kwargs)
    raise ValueError(f"unknown backbone: {name}")
