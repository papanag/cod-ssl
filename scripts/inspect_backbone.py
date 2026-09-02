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
    parser.add_argument(
        "--pathway", choices=["static", "image", "video"], nargs="+", default=["static"],
        help="one or more pathways to inspect while keeping a single model loaded",
    )
    parser.add_argument("--clip-length", type=int, default=64)
    parser.add_argument("--target-index", type=int, default=32)
    parser.add_argument("--input-size", nargs=2, type=int, default=(384, 384))
    parser.add_argument("--image", help="optional COD image; otherwise uses a synthetic image")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, help="cache the completed inspection report")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.backbone = args.backbone or args.model
    if args.backbone is None:
        parser.error("one of --backbone or --model is required")
    if tuple(args.input_size) != (384, 384):
        raise ValueError("the locked ViT-B/16 comparison uses 384x384 inputs")
    if "video" in args.pathway and args.backbone != "vjepa21_vitb16":
        raise ValueError("video token inspection is specific to native V-JEPA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_var = "DINOV3_REPO_DIR" if args.backbone.startswith("dino") else "VJEPA2_REPO_DIR"
    weight_var = "DINOV3_WEIGHTS" if args.backbone.startswith("dino") else "VJEPA21_WEIGHTS"
    repo, weight = Path(os.environ[repo_var]), Path(os.environ[weight_var])
    repo_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
    ).strip()
    identity = {
        "inspector_version": 2,
        "inspector_sha256": sha256(Path(__file__)),
        "model": args.backbone,
        "pathways": args.pathway,
        "clip_length": args.clip_length,
        "target_index": args.target_index,
        "input_size": list(args.input_size),
        "image": args.image,
        "warmup": args.warmup,
        "runs": args.runs,
        "repo_commit": repo_commit,
        "checkpoint_fingerprint": {
            "path": str(weight.resolve()),
            "bytes": weight.stat().st_size,
            "mtime_ns": weight.stat().st_mtime_ns,
        },
        "torch": torch.__version__,
        "device": str(device),
    }
    if args.output and args.output.is_file() and not args.force:
        try:
            cached = json.loads(args.output.read_text())
        except (json.JSONDecodeError, OSError):
            cached = {}
        if cached.get("complete") is True and cached.get("identity") == identity:
            print(f"Using cached backbone inspection: {args.output}")
            print(json.dumps(cached, indent=2))
            return
    model = build_backbone(args.backbone).to(device)
    if args.image:
        with Image.open(args.image) as source: image = source.convert("RGB").resize((384, 384))
        sample = TF.normalize(TF.to_tensor(image), MEAN, STD).unsqueeze(0).to(device)
    else:
        sample = torch.zeros(1, 3, 384, 384, device=device)
    def encode(pathway: str):
        if pathway == "video":
            frames = sample[:, None].expand(-1, args.clip_length, -1, -1, -1)
            return model.encode_video(frames, torch.ones(1, args.clip_length, dtype=torch.bool, device=device))
        if pathway == "image":
            return model.encode_image(sample)
        return model.forward_features(sample)
    reports = []
    for pathway in args.pathway:
        for _ in tqdm(
            range(args.warmup), desc=f"{pathway} warmup", unit="run", dynamic_ncols=True,
        ):
            features = encode(pathway)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in tqdm(
            range(args.runs), desc=f"measure {pathway}", unit="run", dynamic_ncols=True,
        ):
            features = encode(pathway)
        if device.type == "cuda":
            torch.cuda.synchronize()
        reports.append({
            "pathway": pathway,
            "input_shape": list(sample.shape),
            "feature_shapes": (
                [list(x.shape) for x in features]
                if isinstance(features, list) else [list(features.features.shape)]
            ),
            "dtype": str((features[0] if isinstance(features, list) else features.features).dtype),
            "dense_mapping": (None if isinstance(features, list) else {
                "temporal_grid": features.features.shape[1],
                "spatial_grid": list(features.spatial_size),
                "source_frame_intervals": features.source_frame_intervals,
                "target_source_index": args.target_index,
                "target_token_index": args.target_index // 2 if pathway == "video" else 0,
                "output_shape": list(features.features.shape),
            }),
            "mean_forward_ms": 1000 * (time.perf_counter() - started) / args.runs,
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
            ),
        })
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    payload = {
        "complete": True,
        "identity": identity,
        "model": args.backbone,
        "repo_commit": repo_commit,
        "checkpoint": str(weight),
        "checkpoint_sha256": sha256(weight),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": trainable_parameters,
        "reports": reports,
        "runtime": runtime_info(),
    }
    assert trainable_parameters == 0
    assert all(p.grad is None for p in model.parameters())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
