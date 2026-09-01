from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchvision
import yaml

from cod_ssl.utils.runtime import runtime_info


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def create_run_dir(root: str | Path, backbone_name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(root) / f"{stamp}_{backbone_name}_seed42"
    for child in ("checkpoints", "tensorboard", "predictions", "samples"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_metadata(
    run_dir: str | Path,
    config: dict[str, Any],
    *,
    amp_dtype: str,
    trainable_parameters: int,
) -> None:
    run_dir = Path(run_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    train_manifest = Path(config["data"]["train_manifest"])
    if not train_manifest.is_file():
        raise FileNotFoundError(f"training manifest not found while recording run: {train_manifest}")
    shutil.copy2(train_manifest, run_dir / "train_manifest.csv")
    (run_dir / "train_manifest.sha256").write_text(file_sha256(train_manifest) + "\n")
    (run_dir / "git_commit.txt").write_text(git_commit(Path.cwd()) + "\n")
    environment = runtime_info() | {
        "platform": platform.platform(),
        "torchvision": torchvision.__version__,
        "mixed_precision_dtype": amp_dtype,
        "seed": config["experiment"]["seed"],
        "trainable_parameters": trainable_parameters,
    }
    (run_dir / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n"
    )
    upstream = {
        "dinov3_repo_commit": git_commit(os.environ.get("DINOV3_REPO_DIR", "")),
        "vjepa2_repo_commit": git_commit(os.environ.get("VJEPA2_REPO_DIR", "")),
        "dinov3_checkpoint_sha256": None,
        "vjepa21_checkpoint_sha256": None,
    }
    for env_name, key in (
        ("DINOV3_WEIGHTS", "dinov3_checkpoint_sha256"),
        ("VJEPA21_WEIGHTS", "vjepa21_checkpoint_sha256"),
    ):
        path = os.environ.get(env_name)
        if path and Path(path).is_file():
            upstream[key] = file_sha256(path)
    (run_dir / "upstream_versions.json").write_text(json.dumps(upstream, indent=2) + "\n")
