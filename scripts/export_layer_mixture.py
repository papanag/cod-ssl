#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    run_dir = Path(args.run)
    output = Path(args.output) if args.output else run_dir / "layer_mixture"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        run_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False
    )
    state = checkpoint.get("layer_mixer")
    if not state or "logits" not in state:
        raise ValueError(f"run has no learned layer-mixture weights: {run_dir}")
    weights = state["logits"].float().softmax(dim=1).numpy()
    with (output / "layer_mixture_weights.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mixture", *[f"layer_{index}" for index in range(weights.shape[1])]])
        for index, row in enumerate(weights, start=1):
            writer.writerow([f"mixture_{index}", *map(float, row)])
    figure, axis = plt.subplots(figsize=(11, 3.8))
    image = axis.imshow(weights, aspect="auto", cmap="viridis", vmin=0)
    axis.set_xticks(range(weights.shape[1]), labels=range(weights.shape[1]))
    axis.set_yticks(range(weights.shape[0]), labels=range(1, weights.shape[0] + 1))
    axis.set_xlabel("Transformer layer")
    axis.set_ylabel("Learned mixture")
    axis.set_title(run_dir.name)
    figure.colorbar(image, ax=axis, label="Softmax weight")
    figure.tight_layout()
    figure.savefig(output / "layer_mixture_heatmap.png", dpi=180)
    plt.close(figure)
    print(output / "layer_mixture_weights.csv")


if __name__ == "__main__":
    main()
