from __future__ import annotations

import torch
from torch import nn

from cod_ssl.backbones.base import FrozenBackbone
from cod_ssl.models.decoder import CommonCODDecoder


class FrozenCODModel(nn.Module):
    """Compose a frozen representation with the common trainable decoder."""

    def __init__(self, backbone: FrozenBackbone, decoder: CommonCODDecoder | None = None):
        super().__init__()
        self.backbone = backbone
        self.backbone.freeze()
        self.decoder = decoder or CommonCODDecoder(backbone.feature_dims)

    def train(self, mode: bool = True) -> "FrozenCODModel":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            features = self.backbone.forward_features(images)
        # Cloning outside inference_mode converts inference tensors into ordinary
        # detached tensors that trainable decoder layers may save for backward.
        decoder_features = [feature.clone().detach() for feature in features]
        return self.decoder(decoder_features)

    def assert_backbone_frozen(self) -> None:
        if any(parameter.requires_grad for parameter in self.backbone.parameters()):
            raise RuntimeError("backbone contains trainable parameters")
        if any(parameter.grad is not None for parameter in self.backbone.parameters()):
            raise RuntimeError("backbone parameter unexpectedly has a gradient")
