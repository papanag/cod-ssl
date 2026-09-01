from __future__ import annotations

from typing import Any

from cod_ssl.backbones import build_backbone
from cod_ssl.models.video_cod_model import VideoCODModel
from cod_ssl.temporal import (
    GatedMambaMixAdapter,
    MeanTemporalAdapter,
    TargetFrameAdapter,
    VJEPATargetSelector,
)
from cod_ssl.utils.vcod_config import validate_vcod_config


def build_video_cod_model(config: dict[str, Any]) -> VideoCODModel:
    validate_vcod_config(config)
    system = config["experiment"]["system_id"]
    backbone = build_backbone(config["backbone"]["name"], layers=[2, 5, 8, 11])
    adapter_name = config["temporal_adapter"]["name"]
    if adapter_name == "target":
        adapter = TargetFrameAdapter()
    elif adapter_name == "mean":
        adapter = MeanTemporalAdapter()
    elif adapter_name == "vjepa_native":
        adapter = VJEPATargetSelector(tubelet_size=2)
    elif adapter_name == "gated_mamba_mix":
        adapter = GatedMambaMixAdapter(input_dim=backbone.feature_dim, **config["temporal_adapter"].get("config", {}))
    else:
        raise ValueError(f"unsupported temporal adapter: {adapter_name}")
    decoder = config["decoder"]
    return VideoCODModel(
        backbone, adapter, pathway=config["pathway"], feature_dim=backbone.feature_dim,
        projected_channels=int(decoder["projected_channels"]),
        hidden_channels=int(decoder["hidden_channels"]), repeat_target=system == "VR",
    )
