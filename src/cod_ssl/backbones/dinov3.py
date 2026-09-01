from __future__ import annotations

import os
from pathlib import Path

import torch

from cod_ssl.backbones.base import DenseFeatureBatch, FrozenBackbone


class DINOv3ViTB16(FrozenBackbone):
    def __init__(
        self,
        repo_dir: str | Path | None = None,
        weights: str | Path | None = None,
        layers: list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.configure_layers(layers)
        repo = Path(repo_dir or os.environ.get("DINOV3_REPO_DIR", ""))
        checkpoint = Path(weights or os.environ.get("DINOV3_WEIGHTS", ""))
        if not repo.is_dir():
            raise FileNotFoundError("set DINOV3_REPO_DIR to the official dinov3 clone")
        if not checkpoint.is_file():
            raise FileNotFoundError("set DINOV3_WEIGHTS to the approved local checkpoint")
        self.encoder = torch.hub.load(
            str(repo), "dinov3_vitb16", source="local", weights=str(checkpoint)
        )
        self.freeze()

    @property
    def feature_dims(self) -> list[int]:
        return [768] * len(self.layer_indices)

    def forward_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        with torch.inference_mode():
            outputs = self.encoder.get_intermediate_layers(
                images, n=list(self.layer_indices), reshape=True, norm=True
            )
            features = [output.detach() for output in outputs]
        return self.validate_features(images, features)

    @property
    def feature_dim(self) -> int:
        return 768

    @property
    def patch_size(self) -> tuple[int, int]:
        return (16, 16)

    def encode_image(self, image: torch.Tensor) -> DenseFeatureBatch:
        feature = self.forward_features(image)[-1]
        return DenseFeatureBatch(
            features=feature.unsqueeze(1),
            temporal_valid=torch.ones((image.shape[0], 1), dtype=torch.bool, device=image.device),
            spatial_size=(24, 24), source_frame_intervals=((0, 0),),
            metadata={"pathway": "native_image", "layer": self.layer_indices[-1]},
        )

    def encode_video(self, frames: torch.Tensor, temporal_valid: torch.BoolTensor) -> DenseFeatureBatch:
        if frames.ndim != 5 or tuple(frames.shape[2:]) != (3, 384, 384):
            raise ValueError(f"expected [B,T,3,384,384], got {tuple(frames.shape)}")
        batch, time = frames.shape[:2]
        if tuple(temporal_valid.shape) != (batch, time):
            raise ValueError("temporal_valid shape differs from video")
        feature = self.forward_features(frames.reshape(batch * time, 3, 384, 384))[-1]
        feature = feature.reshape(batch, time, 768, 24, 24)
        return DenseFeatureBatch(
            features=feature, temporal_valid=temporal_valid.bool(), spatial_size=(24, 24),
            source_frame_intervals=tuple((index, index) for index in range(time)),
            metadata={"pathway": "framewise_image", "layer": self.layer_indices[-1]},
        )
