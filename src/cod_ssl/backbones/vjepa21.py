from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch

from cod_ssl.backbones.base import DenseFeatureBatch, FrozenBackbone


def _clean_encoder_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Mirrors facebookresearch/vjepa2 src/hub/backbones.py.
    return {
        key.removeprefix("module.").removeprefix("backbone."): value
        for key, value in state.items()
    }


class VJEPA21ViTB16(FrozenBackbone):
    tubelet_size = 2
    def __init__(
        self,
        repo_dir: str | Path | None = None,
        weights: str | Path | None = None,
        layers: list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.configure_layers(layers)
        supported_layers = {2, 5, 8, 11}
        if not set(self.layer_indices).issubset(supported_layers):
            raise ValueError(
                "V-JEPA 2.1 exposes only its official hierarchical layers [2, 5, 8, 11]"
            )
        repo = Path(repo_dir or os.environ.get("VJEPA2_REPO_DIR", ""))
        checkpoint = Path(weights or os.environ.get("VJEPA21_WEIGHTS", ""))
        if not (repo / "app" / "vjepa_2_1").is_dir():
            raise FileNotFoundError("set VJEPA2_REPO_DIR to the official vjepa2 clone")
        if not checkpoint.is_file():
            raise FileNotFoundError("set VJEPA21_WEIGHTS to the local V-JEPA 2.1 checkpoint")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        vit = importlib.import_module("app.vjepa_2_1.models.vision_transformer")
        self.encoder = vit.vit_base(
            img_size=(384, 384), patch_size=16, num_frames=64, tubelet_size=2,
            use_sdpa=True, use_SiLU=False, wide_SiLU=True, uniform_power=False,
            use_rope=True, img_temporal_dim_size=1, interpolate_rope=True,
            out_layers=list(self.layer_indices), n_output_distillation=1,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "ema_encoder" not in payload:
            raise KeyError("V-JEPA 2.1 checkpoint has no 'ema_encoder' state")
        state = _clean_encoder_state(payload["ema_encoder"])
        self.encoder.load_state_dict(state, strict=True)
        if int(self.encoder.embed_dim) != 768:
            raise RuntimeError(f"expected ViT-B embed_dim 768, got {self.encoder.embed_dim}")
        self.freeze()

    @property
    def feature_dims(self) -> list[int]:
        return [int(self.encoder.embed_dim)] * len(self.layer_indices)

    def forward_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 384, 384):
            raise ValueError(f"expected input [B,3,384,384], got {tuple(images.shape)}")
        # Native image pathway: a single temporal frame activates patch_embed_img.
        with torch.inference_mode():
            tokens = self.encoder(images.unsqueeze(2))
            if not isinstance(tokens, (list, tuple)) or len(tokens) != len(self.layer_indices):
                raise RuntimeError("V-JEPA returned an unexpected number of layer tensors")
            features = []
            for layer_tokens in tokens:
                if layer_tokens.ndim != 3 or layer_tokens.shape[1] != 24 * 24:
                    raise RuntimeError(f"ambiguous V-JEPA token shape {tuple(layer_tokens.shape)}")
                feature = layer_tokens.transpose(1, 2).reshape(images.shape[0], -1, 24, 24)
                features.append(feature.detach())
        return self.validate_features(images, features)

    @property
    def feature_dim(self) -> int:
        return int(self.encoder.embed_dim)

    @property
    def patch_size(self) -> tuple[int, int]:
        return (16, 16)

    def encode_image(self, image: torch.Tensor) -> DenseFeatureBatch:
        feature = self.forward_features(image)[-1]
        return DenseFeatureBatch(
            features=feature.unsqueeze(1),
            temporal_valid=torch.ones((image.shape[0], 1), dtype=torch.bool, device=image.device),
            spatial_size=(24, 24), source_frame_intervals=((0, 0),),
            metadata={"pathway": "official_image", "img_temporal_dim_size": 1,
                      "layer": self.layer_indices[-1]},
        )

    def encode_video(self, frames: torch.Tensor, temporal_valid: torch.BoolTensor) -> DenseFeatureBatch:
        if frames.ndim != 5 or tuple(frames.shape[2:]) != (3, 384, 384):
            raise ValueError(f"expected [B,T,3,384,384], got {tuple(frames.shape)}")
        batch, time = frames.shape[:2]
        if time < self.tubelet_size or time % self.tubelet_size:
            raise ValueError(f"V-JEPA clip length must be divisible by tubelet size {self.tubelet_size}")
        if tuple(temporal_valid.shape) != (batch, time):
            raise ValueError("temporal_valid shape differs from video")
        with torch.inference_mode():
            outputs = self.encoder(frames.permute(0, 2, 1, 3, 4))
        if not isinstance(outputs, (list, tuple)) or not outputs:
            raise RuntimeError("V-JEPA video pathway returned no hierarchical features")
        tokens = outputs[-1]
        temporal_grid = time // self.tubelet_size
        expected_tokens = temporal_grid * 24 * 24
        if tokens.ndim != 3 or tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"ambiguous V-JEPA video tokens {tuple(tokens.shape)}; expected [B,{expected_tokens},C]"
            )
        # Official ViT patch embedding flattens temporal, height, width in that order.
        features = tokens.reshape(batch, temporal_grid, 24, 24, -1).permute(0, 1, 4, 2, 3).detach()
        # A boundary tubelet remains usable when it contains at least one real
        # observation; its exact partial coverage is retained by the source mask.
        valid = temporal_valid.reshape(batch, temporal_grid, self.tubelet_size).any(dim=-1)
        intervals = tuple(
            (index * self.tubelet_size, (index + 1) * self.tubelet_size - 1)
            for index in range(temporal_grid)
        )
        return DenseFeatureBatch(
            features=features, temporal_valid=valid, spatial_size=(24, 24),
            source_frame_intervals=intervals,
            metadata={"pathway": "native_video", "token_order": "temporal_height_width",
                      "tubelet_size": self.tubelet_size, "layer": self.layer_indices[-1]},
        )
