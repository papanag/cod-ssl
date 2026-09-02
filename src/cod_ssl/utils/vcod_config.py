from __future__ import annotations

from copy import deepcopy
from typing import Any

SYSTEMS = {
    "DS": ("dinov3_vitb16", "target_only", "target", "image"),
    "VI": ("vjepa21_vitb16", "target_only", "target", "image"),
    "DT": ("dinov3_vitb16", "ordered_real_clip", "gated_mamba_mix", "framewise_video"),
    "VV": ("vjepa21_vitb16", "ordered_real_clip", "vjepa_native", "native_video"),
    "DM": ("dinov3_vitb16", "ordered_real_clip", "mean", "framewise_video"),
    "VR": ("vjepa21_vitb16", "repeated_target", "vjepa_native", "native_video"),
}

SCIENTIFIC_PATHS = (
    ("clip", "length"), ("clip", "stride"), ("clip", "target_index"),
    ("backbone", "feature_layer"), ("backbone", "input_size"),
    ("training", "learning_rate"), ("training", "max_steps"),
    ("training", "early_stopping_patience"),
)


def configure_system(config: dict[str, Any], system_id: str) -> dict[str, Any]:
    if system_id not in SYSTEMS:
        raise ValueError(f"unknown VCOD system: {system_id}")
    result = deepcopy(config)
    backbone, input_kind, adapter, pathway = SYSTEMS[system_id]
    result["experiment"]["system_id"] = system_id
    result["backbone"]["name"] = backbone
    result["temporal_adapter"]["name"] = adapter
    result["input_treatment"] = input_kind
    result["pathway"] = pathway
    return result


def validate_vcod_config(config: dict[str, Any]) -> None:
    for path in SCIENTIFIC_PATHS:
        value: Any = config
        for key in path:
            if not isinstance(value, dict) or key not in value:
                raise ValueError(f"missing scientific config field: {'.'.join(path)}")
            value = value[key]
        if value is None:
            raise ValueError(f"unresolved scientific config field: {'.'.join(path)}")
    system = config["experiment"]["system_id"]
    if system not in SYSTEMS:
        raise ValueError(f"unknown VCOD system: {system}")
    expected = SYSTEMS[system]
    actual = (config["backbone"]["name"], config.get("input_treatment"),
              config["temporal_adapter"]["name"], config.get("pathway"))
    if actual != expected:
        raise ValueError(f"system {system} requires {expected}, got {actual}")
    if not config["backbone"].get("frozen", False):
        raise ValueError("all primary backbones must remain frozen")
    clip = config["clip"]
    if clip["length"] < 1 or not 0 <= clip["target_index"] < clip["length"]:
        raise ValueError("invalid clip length/target index")
    if system in {"VI", "VV", "VR"} and system != "VI" and clip["length"] % 2:
        raise ValueError("V-JEPA video clip length must be divisible by tubelet size 2")
    if system in {"DM", "VR"} and config["experiment"].get("primary", False):
        raise ValueError("DM and VR are diagnostics, never primary systems")
    if clip.get("context_direction") not in {"causal", "bidirectional"}:
        raise ValueError("clip.context_direction must explicitly be causal or bidirectional")
    dataset = config["dataset"]
    if dataset.get("name") == "camotion":
        if dataset.get("release_profile") != "camotion_public_stride5_v1":
            raise ValueError("CAMotion requires release profile camotion_public_stride5_v1")
        if dataset.get("dense_intermediate_rgb_available", False):
            raise ValueError("public CAMotion cannot declare dense intermediate RGB")
        if clip.get("released_stride") != 1 or clip.get("source_frame_stride") != 5:
            raise ValueError("CAMotion primary clips require released stride 1 and source-frame stride 5")
    if dataset.get("name") == "moca_mask_dense":
        if dataset.get("release_profile") != "moca_mask_dense_v1":
            raise ValueError("dense MoCA requires the moca_mask_dense_v1 preprocessing product")
        if dataset.get("boundary_policy") != "manual_target_hull_v1":
            raise ValueError("dense MoCA boundary policy must be manual_target_hull_v1")
        if clip.get("source_frame_stride") not in {1, 5}:
            raise ValueError("dense MoCA supports declared source-frame strides 1 or 5")
    evaluation = config["evaluation"]
    if evaluation["cod_prediction_view"] != "minmax" or evaluation["diagnostic_prediction_view"] != "sigmoid_raw":
        raise ValueError("COD and raw diagnostic prediction views must remain distinct")
