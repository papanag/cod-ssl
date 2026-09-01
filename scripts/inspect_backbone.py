#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from cod_ssl.backbones import build_backbone
from cod_ssl.data.transforms import MEAN, STD
from cod_ssl.utils.runtime import runtime_info


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["dinov3_vitb16", "vjepa21_vitb16"])
    parser.add_argument("--model", choices=["dinov3_vitb16", "vjepa21_vitb16"])
    parser.add_argument("--pathway", choices=["static", "image", "video"], default="static")
    parser.add_argument("--clip-length", type=int, default=64)
    parser.add_argument("--target-index", type=int, default=32)
    parser.add_argument("--input-size", nargs=2, type=int, default=(384, 384))
    parser.add_argument("--image", help="optional COD image; otherwise uses a synthetic image")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    args.backbone = args.backbone or args.model
    if args.backbone is None:
        parser.error("one of --backbone or --model is required")
    if tuple(args.input_size) != (384, 384):
        raise ValueError("the locked ViT-B/16 comparison uses 384x384 inputs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_backbone(args.backbone).to(device)
    if args.image:
        with Image.open(args.image) as source: image = source.convert("RGB").resize((384, 384))
        sample = TF.normalize(TF.to_tensor(image), MEAN, STD).unsqueeze(0).to(device)
    else:
        sample = torch.zeros(1, 3, 384, 384, device=device)
    def encode():
        if args.pathway == "video":
            if args.backbone != "vjepa21_vitb16":
                raise ValueError("video token inspection is specific to native V-JEPA")
            frames = sample[:, None].expand(-1, args.clip_length, -1, -1, -1)
            return model.encode_video(frames, torch.ones(1, args.clip_length, dtype=torch.bool, device=device))
        if args.pathway == "image":
            return model.encode_image(sample)
        return model.forward_features(sample)
    for _ in tqdm(
        range(args.warmup), desc="backbone warmup", unit="run", dynamic_ncols=True
    ):
        features = encode()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in tqdm(
        range(args.runs), desc="measure backbone", unit="run", dynamic_ncols=True
    ):
        features = encode()
    if device.type == "cuda": torch.cuda.synchronize()
    repo_var = "DINOV3_REPO_DIR" if args.backbone.startswith("dino") else "VJEPA2_REPO_DIR"
    weight_var = "DINOV3_WEIGHTS" if args.backbone.startswith("dino") else "VJEPA21_WEIGHTS"
    repo, weight = Path(os.environ[repo_var]), Path(os.environ[weight_var])
    report = {"model": args.backbone, "repo_commit": subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(weight), "checkpoint_sha256": sha256(weight),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "input_shape": list(sample.shape),
        "feature_shapes": ([list(x.shape) for x in features] if isinstance(features, list) else [list(features.features.shape)]),
        "dtype": str((features[0] if isinstance(features, list) else features.features).dtype),
        "pathway": args.pathway,
        "dense_mapping": (None if isinstance(features, list) else {
            "temporal_grid": features.features.shape[1], "spatial_grid": list(features.spatial_size),
            "source_frame_intervals": features.source_frame_intervals,
            "target_source_index": args.target_index,
            "target_token_index": (args.target_index // 2 if args.pathway == "video" else 0),
            "output_shape": list(features.features.shape),
        }),
        "mean_forward_ms": 1000*(time.perf_counter()-started)/args.runs,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "runtime": runtime_info()}
    assert report["trainable_parameters"] == 0
    assert all(p.grad is None for p in model.parameters())
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
