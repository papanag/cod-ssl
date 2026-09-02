#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from tqdm.auto import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated/shuffled diagnostics after primary runs")
    parser.add_argument("--dataset", choices=("moca_mask_dense", "camotion"), required=True)
    parser.add_argument("--regime", choices=("D1", "S5"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnostic-seed", type=int, default=20260902)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dataset == "camotion" and args.regime != "S5":
        raise ValueError("CAMotion diagnostics require its public S5 regime")
    source_stride = 1 if args.regime == "D1" else 5
    released_stride = 1 if args.dataset == "camotion" else source_stride
    manifest_env = "MOCA_MASK_DENSE_MANIFEST" if args.dataset == "moca_mask_dense" else "CAMOTION_MANIFEST"
    release_profile = "moca_mask_dense_v1" if args.dataset == "moca_mask_dense" else "camotion_public_stride5_v1"
    boundary = "manual_target_hull_v1" if args.dataset == "moca_mask_dense" else "public_sequence_extent_v1"
    plans = [
        {"system": "VR", "order_mode": "repeated", "role": "repeated_target"},
        {"system": "DT", "order_mode": "shuffled", "role": "ordered_vs_shuffled"},
        {"system": "VV", "order_mode": "shuffled", "role": "ordered_vs_shuffled"},
    ]
    print(json.dumps(plans, indent=2))
    if args.dry_run:
        return
    for plan in tqdm(plans, desc="temporal diagnostics", unit="run", dynamic_ncols=True):
        config = (
            "configs/experiments/vcod_diagnostics.yaml"
            if plan["system"] == "VR" else "configs/experiments/vcod_primary_2x2.yaml"
        )
        command = [
            sys.executable, "scripts/train_probe.py", "--config", config,
            f"experiment.system_id={plan['system']}", "experiment.primary=false",
            f"experiment.seed={args.seed}", f"dataset.name={args.dataset}",
            f"dataset.regime={args.regime}", f"dataset.manifest_env={manifest_env}",
            f"dataset.release_profile={release_profile}", f"dataset.boundary_policy={boundary}",
            f"dataset.dense_intermediate_rgb_available={str(args.dataset == 'moca_mask_dense').lower()}",
            f"clip.stride={released_stride}", f"clip.released_stride={released_stride}",
            f"clip.source_frame_stride={source_stride}",
            f"clip.context_cadence={'dense_source_stride1' if source_stride == 1 else 'source_stride5'}",
            f"clip.order_mode={plan['order_mode']}", f"clip.diagnostic_seed={args.diagnostic_seed}",
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
