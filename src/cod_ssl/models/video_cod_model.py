from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from cod_ssl.backbones.base import DenseFeatureBatch
from cod_ssl.heads import SmallConvSegmentationHead
from cod_ssl.temporal import VJEPATargetSelector
from cod_ssl.temporal.base import TemporalAdapter


class VideoCODModel(nn.Module):
    """Common assembly; system differences are fixed at construction time."""

    def __init__(
        self, backbone: nn.Module, temporal_adapter: TemporalAdapter, *,
        pathway: Literal["image", "framewise_video", "native_video"],
        feature_dim: int, projected_channels: int = 256, hidden_channels: int = 128,
        repeat_target: bool = False,
    ):
        super().__init__()
        self.backbone, self.temporal_adapter, self.pathway = backbone, temporal_adapter, pathway
        self.repeat_target = repeat_target
        if hasattr(backbone, "freeze"):
            backbone.freeze()
        self.projection = nn.Conv2d(feature_dim, projected_channels, 1)
        self.decoder = SmallConvSegmentationHead(projected_channels, hidden_channels)

    @staticmethod
    def _uniform_int(value: Any, name: str) -> int:
        if isinstance(value, torch.Tensor):
            flat = value.flatten()
            if not torch.all(flat == flat[0]):
                raise ValueError(f"{name} must be uniform within a batch")
            return int(flat[0])
        return int(value)

    def _encode(self, batch: dict[str, Any]) -> tuple[DenseFeatureBatch, int]:
        frames = batch["frames"]
        valid = batch["valid_temporal_mask"].bool()
        source_target = self._uniform_int(batch["target_index"], "target_index")
        if self.pathway == "image":
            return self.backbone.encode_image(frames[:, source_target]), 0
        if self.repeat_target:
            frames = frames[:, source_target:source_target + 1].expand_as(frames)
        dense = self.backbone.encode_video(frames, valid)
        if self.pathway == "native_video":
            if not isinstance(self.temporal_adapter, VJEPATargetSelector):
                raise TypeError("native V-JEPA video requires VJEPATargetSelector")
            source_target = self.temporal_adapter.token_index(source_target, frames.shape[1])
        return dense, source_target

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | None]:
        with torch.inference_mode():
            dense, target_index = self._encode(batch)
        selected, next_state = self.temporal_adapter(
            dense.features.clone().detach(), dense.temporal_valid, target_index, state=None
        )
        logits = self.decoder(self.projection(selected), tuple(batch["target_mask"].shape[-2:]))
        return {"logits": logits, "state": next_state}

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def assert_gradient_contract(self, *, after_backward: bool = False) -> None:
        if any(parameter.requires_grad for parameter in self.backbone.parameters()):
            raise RuntimeError("backbone is not frozen")
        groups = [self.projection, self.decoder]
        if sum(parameter.numel() for parameter in self.temporal_adapter.parameters()):
            groups.append(self.temporal_adapter)
        if any(not parameter.requires_grad for group in groups for parameter in group.parameters()):
            raise RuntimeError("a readout/adapter parameter is unexpectedly frozen")
        if after_backward:
            if any(parameter.grad is not None for parameter in self.backbone.parameters()):
                raise RuntimeError("frozen backbone received gradients")
            gradients = [parameter.grad for group in groups for parameter in group.parameters()]
            if not any(gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
                       for gradient in gradients):
                raise RuntimeError("trainable path has no finite non-zero gradient")
