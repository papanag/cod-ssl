from __future__ import annotations

import torch
from torch import nn

from cod_ssl.backbones import build_backbone
from cod_ssl.backbones.base import FrozenBackbone
from cod_ssl.models.decoder import CommonCODDecoder
from cod_ssl.models.layer_mixer import LearnedLayerMixer


class FrozenCODModel(nn.Module):
    """Compose a frozen representation with the common trainable decoder."""

    def __init__(
        self,
        backbone: FrozenBackbone,
        decoder: CommonCODDecoder | None = None,
        *,
        learned_layer_mixtures: int | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.freeze()
        self.layer_mixer = (
            LearnedLayerMixer(backbone.feature_dims, learned_layer_mixtures)
            if learned_layer_mixtures is not None else None
        )
        decoder_dims = (
            [backbone.feature_dims[0]] * learned_layer_mixtures
            if self.layer_mixer is not None else backbone.feature_dims
        )
        self.decoder = decoder or CommonCODDecoder(decoder_dims)

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
        if self.layer_mixer is not None:
            decoder_features = self.layer_mixer(decoder_features)
        return self.decoder(decoder_features)

    def readout_parameters(self):
        yield from self.decoder.parameters()
        if self.layer_mixer is not None:
            yield from self.layer_mixer.parameters()

    def assert_backbone_frozen(self) -> None:
        if any(parameter.requires_grad for parameter in self.backbone.parameters()):
            raise RuntimeError("backbone contains trainable parameters")
        if any(parameter.grad is not None for parameter in self.backbone.parameters()):
            raise RuntimeError("backbone parameter unexpectedly has a gradient")


def build_frozen_cod_model(config: dict) -> FrozenCODModel:
    backbone_config = config["model"]["backbone"]
    backbone = build_backbone(backbone_config["name"], layers=backbone_config["layers"])
    mixer_config = config["model"].get("layer_mixer")
    if mixer_config and mixer_config.get("name") != "learned_softmax":
        raise ValueError(f"unsupported layer mixer: {mixer_config.get('name')}")
    mixtures = int(mixer_config["mixtures"]) if mixer_config else None
    return FrozenCODModel(backbone, learned_layer_mixtures=mixtures)
