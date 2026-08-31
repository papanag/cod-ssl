#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from cod_ssl.backbones import build_backbone
from cod_ssl.data import CODDataset
from cod_ssl.engine import Trainer, TrainingOptions
from cod_ssl.engine.evaluate import logits_to_prediction
from cod_ssl.losses import BCESoftIoULoss
from cod_ssl.models import FrozenCODModel
from cod_ssl.utils.config import load_config
from cod_ssl.utils.reproducibility import seed_everything
from cod_ssl.utils.run import create_run_dir, write_run_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--resume")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    seed_everything(int(config["experiment"]["seed"]))
    backbone_name = config["model"]["backbone"]["name"]
    if config["training"]["optimizer"] != "adamw":
        raise ValueError("Phase-1 training requires optimizer: adamw")
    if config["training"]["scheduler"] != "cosine":
        raise ValueError("Phase-1 training requires scheduler: cosine")
    run_dir = Path(args.run_dir) if args.run_dir else create_run_dir(args.runs_root, backbone_name)
    dataset = CODDataset(config["data"]["train_manifest"], training=True)
    if args.limit_train is not None:
        if args.limit_train < 1:
            raise ValueError("--limit-train must be positive")
        dataset = Subset(dataset, range(min(args.limit_train, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config["data"]["num_workers"]) > 0,
    )
    backbone = build_backbone(backbone_name)
    model = FrozenCODModel(backbone)
    training = config["training"]
    options = TrainingOptions(
        epochs=int(training["epochs"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        accumulation_steps=int(training["gradient_accumulation_steps"]),
        amp=bool(training["amp"]),
        amp_dtype=str(training.get("amp_dtype", "auto")),
        grad_clip_norm=float(training["grad_clip_norm"]),
        checkpoint_every=int(training.get("checkpoint_every", 1)),
    )
    trainer = Trainer(
        model,
        loader,
        run_dir,
        options,
        loss_fn=BCESoftIoULoss(float(config["loss"]["smooth"])),
    )
    write_run_metadata(
        run_dir,
        config,
        amp_dtype=str(trainer.amp_dtype),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    if args.resume:
        trainer.resume(args.resume)
    print(f"Run directory: {run_dir}")
    print(f"AMP: enabled={trainer.amp_enabled}, dtype={trainer.amp_dtype}")
    trainer.fit()
    sample = dataset[0]
    model.eval()
    sample_image = sample["image"].unsqueeze(0).to(trainer.device)
    with torch.inference_mode(), torch.autocast(
        device_type=trainer.device.type,
        dtype=trainer.amp_dtype,
        enabled=trainer.amp_enabled,
    ):
        sample_logits = model(sample_image)
    sample_prediction = logits_to_prediction(sample_logits, (384, 384))
    Image.fromarray(sample_prediction).save(run_dir / "samples" / "training_sample.png")
    model.assert_backbone_frozen()
    if hasattr(os, "sync"):
        os.sync()
    print("Training complete; backbone freeze invariant passed.")


if __name__ == "__main__":
    main()
