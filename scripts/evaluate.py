#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from cod_ssl.data import CODDataset
from cod_ssl.engine import Evaluator
from cod_ssl.engine.train import select_amp
from cod_ssl.models import build_frozen_cod_model
from cod_ssl.utils.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset", choices=["camo_test", "cod10k_test", "chameleon", "nc4k"])
    args = parser.parse_args()

    run_dir = Path(args.run)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    seed_everything(int(config["experiment"]["seed"]))
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    model = build_frozen_cod_model(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.decoder.load_state_dict(state["decoder"], strict=True)
    if model.layer_mixer is not None:
        if state.get("layer_mixer") is None:
            raise KeyError("checkpoint has no learned layer-mixer state")
        model.layer_mixer.load_state_dict(state["layer_mixer"], strict=True)
    model.assert_backbone_frozen()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype = select_amp(
        device,
        bool(config["training"]["amp"]),
        str(config["training"].get("amp_dtype", "auto")),
    )
    evaluation = config["evaluation"]
    if not evaluation["minmax_normalize"]:
        raise ValueError("Phase-1 headline evaluation requires per-image min-max normalization")
    if not evaluation["save_predictions"]:
        raise ValueError("Phase-1 evaluation requires saving the exact evaluator PNGs")
    dataset_names = [args.dataset] if args.dataset else list(evaluation["manifests"])
    all_results: dict[str, dict] = {}
    for dataset_name in dataset_names:
        dataset = CODDataset(evaluation["manifests"][dataset_name], training=False)
        loader = DataLoader(
            dataset,
            batch_size=int(evaluation["batch_size"]),
            shuffle=False,
            num_workers=int(evaluation["num_workers"]),
            pin_memory=device.type == "cuda",
            persistent_workers=int(evaluation["num_workers"]) > 0,
        )
        evaluator = Evaluator(
            model,
            loader,
            run_dir / "predictions" / dataset_name,
            device=device,
            minmax_normalize=bool(evaluation["minmax_normalize"]),
            save_predictions=bool(evaluation["save_predictions"]),
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        all_results[dataset_name] = evaluator.evaluate()
        print(dataset_name, json.dumps(all_results[dataset_name], indent=2))

    metrics_path = run_dir / "metrics.json"
    existing_results = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    existing_results.update(all_results)
    metrics_path.write_text(json.dumps(existing_results, indent=2) + "\n")
    rows = [{"dataset": name, **values} for name, values in existing_results.items()]
    with (run_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if hasattr(os, "sync"):
        os.sync()


if __name__ == "__main__":
    main()
