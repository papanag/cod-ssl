from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class LearnedLayerMixer(nn.Module):
    """Create four softmax mixtures from all frozen transformer layers."""

    def __init__(self, feature_dims: Sequence[int], mixtures: int = 4):
        super().__init__()
        if len(feature_dims) != 12 or len(set(map(int, feature_dims))) != 1:
            raise ValueError("learned mixing requires twelve equal-width feature maps")
        if mixtures != 4:
            raise ValueError("the common decoder requires exactly four learned mixtures")
        self.num_layers = len(feature_dims)
        self.num_mixtures = mixtures
        self.logits = nn.Parameter(torch.zeros(mixtures, self.num_layers))

    @property
    def weights(self) -> torch.Tensor:
        return self.logits.softmax(dim=1)

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} layer features, got {len(features)}")
        stacked = torch.stack(list(features), dim=1)
        return list(torch.einsum("ml,blchw->bmchw", self.weights, stacked).unbind(dim=1))
