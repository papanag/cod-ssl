from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from cod_ssl.models.decoder import ConvNormGELU


class SmallConvSegmentationHead(nn.Module):
    """Shared two-block decoder topology for every VCOD system."""

    def __init__(self, input_channels: int = 256, hidden_channels: int = 128):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvNormGELU(input_channels, hidden_channels, 3),
            ConvNormGELU(hidden_channels, hidden_channels, 3),
        )
        self.classifier = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, features: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        logits = self.classifier(self.blocks(features))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
