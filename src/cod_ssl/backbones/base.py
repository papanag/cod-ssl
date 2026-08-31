from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import nn


class FrozenBackbone(nn.Module, ABC):
    input_size = 384
    grid_size = 24
    layer_indices = (2, 5, 8, 11)

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenBackbone":
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
        if len(features) != 4:
            raise RuntimeError(f"expected exactly 4 features, got {len(features)}")
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

