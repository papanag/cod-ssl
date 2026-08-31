from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    """Choose a stable GroupNorm divisor while keeping groups reasonably sized."""
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormGELU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class UpsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = ConvNormGELU(in_channels, out_channels, kernel_size=3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(inputs, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.block(inputs)


class CommonCODDecoder(nn.Module):
    """The backbone-agnostic Phase-1 decoder fixed by the scientific protocol."""

    def __init__(self, feature_dims: Sequence[int], projection_dim: int = 128):
        super().__init__()
        if len(feature_dims) != 4:
            raise ValueError(f"expected four feature dimensions, got {len(feature_dims)}")
        if projection_dim != 128:
            raise ValueError("Phase-1 protocol fixes projection_dim at 128")

        self.feature_dims = tuple(int(value) for value in feature_dims)
        self.projections = nn.ModuleList(
            ConvNormGELU(channels, projection_dim, kernel_size=1)
            for channels in self.feature_dims
        )
        self.fusion = ConvNormGELU(4 * projection_dim, 256, kernel_size=3)
        self.upsampling = nn.Sequential(
            UpsampleStage(256, 192),
            UpsampleStage(192, 128),
            UpsampleStage(128, 64),
            UpsampleStage(64, 32),
        )
        self.classifier = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(f"expected exactly four features, got {len(features)}")
        batch_size = features[0].shape[0]
        for index, (feature, channels) in enumerate(zip(features, self.feature_dims)):
            expected = (batch_size, channels, 24, 24)
            if feature.ndim != 4 or tuple(feature.shape) != expected:
                raise ValueError(
                    f"feature {index} has shape {tuple(feature.shape)}; expected {expected}"
                )
        projected = [projection(feature) for projection, feature in zip(self.projections, features)]
        fused = self.fusion(torch.cat(projected, dim=1))
        return self.classifier(self.upsampling(fused))

