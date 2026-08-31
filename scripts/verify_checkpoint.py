#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as TF

from cod_ssl.backbones import build_backbone
from cod_ssl.data.transforms import MEAN, STD
from cod_ssl.models import FrozenCODModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    run_dir = Path(args.run)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "last.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrozenCODModel(build_backbone(config["model"]["backbone"]["name"])).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.decoder.load_state_dict(payload["decoder"], strict=True)
    model.eval()
    with Image.open(args.image) as raw:
        image = raw.convert("RGB").resize((384, 384))
    tensor = TF.normalize(TF.to_tensor(image), MEAN, STD).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
    model.assert_backbone_frozen()
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "logit_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "backbone_trainable_parameters": sum(
            parameter.numel() for parameter in model.backbone.parameters() if parameter.requires_grad
        ),
        "backbone_parameters_with_gradients": sum(
            parameter.numel() for parameter in model.backbone.parameters() if parameter.grad is not None
        ),
    }
    if report["logit_shape"] != [1, 1, 384, 384] or not report["logits_finite"]:
        raise RuntimeError(f"invalid reloaded checkpoint output: {report}")
    if report["backbone_trainable_parameters"] or report["backbone_parameters_with_gradients"]:
        raise RuntimeError(f"backbone freeze invariant failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
