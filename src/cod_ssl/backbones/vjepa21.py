from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch

from cod_ssl.backbones.base import FrozenBackbone


def _clean_encoder_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Mirrors facebookresearch/vjepa2 src/hub/backbones.py.
    return {
        key.removeprefix("module.").removeprefix("backbone."): value
        for key, value in state.items()
    }


class VJEPA21ViTB16(FrozenBackbone):
    def __init__(self, repo_dir: str | Path | None = None, weights: str | Path | None = None):
        super().__init__()
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
        return [int(self.encoder.embed_dim)] * 4

    def forward_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 384, 384):
            raise ValueError(f"expected input [B,3,384,384], got {tuple(images.shape)}")
        # Native image pathway: a single temporal frame activates patch_embed_img.
        with torch.inference_mode():
            tokens = self.encoder(images.unsqueeze(2))
            if not isinstance(tokens, (list, tuple)) or len(tokens) != 4:
                raise RuntimeError("V-JEPA did not return four unambiguous layer tensors")
            features = []
            for layer_tokens in tokens:
                if layer_tokens.ndim != 3 or layer_tokens.shape[1] != 24 * 24:
                    raise RuntimeError(f"ambiguous V-JEPA token shape {tuple(layer_tokens.shape)}")
                feature = layer_tokens.transpose(1, 2).reshape(images.shape[0], -1, 24, 24)
                features.append(feature.detach())
        return self.validate_features(images, features)

