#!/usr/bin/env python3
"""Idempotent post-clone bootstrap shared by every Colab notebook."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import torch

DINOV3_REPO_URL = "https://github.com/facebookresearch/dinov3.git"
VJEPA2_REPO_URL = "https://github.com/facebookresearch/vjepa2.git"
VJEPA21_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
SAMPLE_IMAGE_URL = (
    "https://raw.githubusercontent.com/DengPingFan/SINet/"
    "master/Images/CamouflagedTask.png"
)


def clone_or_update(url: str, destination: Path) -> None:
    if (destination / ".git").is_dir():
        subprocess.run(["git", "-C", str(destination), "pull", "--ff-only"], check=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", url, str(destination)], check=True)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "Wget/1.21.4"})
    try:
        with urlopen(request) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            received = 0
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                received += len(chunk)
                if total:
                    print(f"\r{destination.name}: {100 * received / total:.1f}%", end="")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"\nSaved {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default="/content/cod-ssl")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/cod-ssl")
    parser.add_argument("--state-file", default="/content/cod_ssl_bootstrap_state.json")
    parser.add_argument("--ensure-training-data", action="store_true")
    parser.add_argument("--accept-noncommercial-license", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU-backed Colab runtime before running this notebook")
    project_dir = Path(args.project_dir)
    if not (project_dir / "pyproject.toml").is_file():
        raise FileNotFoundError(f"project checkout is incomplete: {project_dir}")
    third_party = Path("/content/third_party")
    dinov3_repo = third_party / "dinov3"
    vjepa2_repo = third_party / "vjepa2"
    clone_or_update(DINOV3_REPO_URL, dinov3_repo)
    clone_or_update(VJEPA2_REPO_URL, vjepa2_repo)

    drive_root = Path(args.drive_root)
    weights_dir = drive_root / "checkpoints"
    dinov3_weights = weights_dir / "dinov3_vitb16.pth"
    vjepa_weights = weights_dir / "vjepa2_1_vitb_dist_vitG_384.pt"
    if vjepa_weights.is_file():
        print(f"Using cached V-JEPA checkpoint: {vjepa_weights}")
    else:
        print("Downloading public V-JEPA 2.1 ViT-B/16 checkpoint...")
        download(VJEPA21_URL, vjepa_weights)
    if dinov3_weights.is_file():
        print(f"Using cached DINOv3 checkpoint: {dinov3_weights}")
    else:
        private_url = os.environ.pop("COD_SSL_DINOV3_DOWNLOAD_URL", "").strip()
        if not private_url:
            raise PermissionError(
                "DINOv3 is not cached. The notebook launcher must provide the approved Meta URL."
            )
        download(private_url, dinov3_weights)

    sample_image = Path("/content/cod_ssl_sample_image.png")
    if not sample_image.is_file():
        download(SAMPLE_IMAGE_URL, sample_image)
    else:
        print(f"Using cached smoke-test image: {sample_image}")

    train_manifest = project_dir / "manifests" / "train_all.csv"
    if args.ensure_training_data:
        if not args.accept_noncommercial_license:
            raise PermissionError(
                "Review the COD10K non-commercial license and pass "
                "--accept-noncommercial-license explicitly"
            )
        subprocess.run(
            [
                sys.executable,
                str(project_dir / "scripts" / "bootstrap_training_data.py"),
                "--data-root", str(drive_root / "data"),
                "--manifest", str(train_manifest),
                "--accept-noncommercial-license",
            ],
            cwd=project_dir,
            check=True,
        )

    environment = {
        "DINOV3_REPO_DIR": str(dinov3_repo),
        "DINOV3_WEIGHTS": str(dinov3_weights),
        "VJEPA2_REPO_DIR": str(vjepa2_repo),
        "VJEPA21_WEIGHTS": str(vjepa_weights),
    }
    required = [project_dir, dinov3_repo, vjepa2_repo, dinov3_weights, vjepa_weights, sample_image]
    if args.ensure_training_data:
        required.append(train_manifest)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Bootstrap missing required paths:\n" + "\n".join(missing))
    state = {
        "project_dir": str(project_dir),
        "drive_root": str(drive_root),
        "data_root": str(drive_root / "data"),
        "runs_root": str(drive_root / "runs"),
        "comparisons_root": str(drive_root / "comparisons"),
        "sample_image": str(sample_image),
        "train_manifest": str(train_manifest),
        "environment": environment,
        "gpu": torch.cuda.get_device_name(0),
    }
    state_file = Path(args.state_file)
    state_file.write_text(json.dumps(state, indent=2) + "\n")
    if hasattr(os, "sync"):
        os.sync()
    print(f"Bootstrap complete on {state['gpu']}; state: {state_file}")


if __name__ == "__main__":
    main()
