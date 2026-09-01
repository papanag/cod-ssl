from __future__ import annotations

import os
from pathlib import Path

import torch

from cod_ssl.backbones.base import FrozenBackbone


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
