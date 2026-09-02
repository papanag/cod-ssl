#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from tqdm.auto import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the matched-target MoCA D1/S5 sampling/coverage ablation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-config", default="configs/experiments/vcod_primary_2x2.yaml")
    parser.add_argument("--systems", nargs="+", choices=("DT", "VV"), required=True)
    parser.add_argument("--source-frame-strides", nargs="+", type=int, choices=(1, 5), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ablation = yaml.safe_load(Path(args.config).read_text())["ablation"]
    if ablation.get("dataset") != "moca_mask_dense" or set(args.systems) - set(ablation["systems"]):
        raise ValueError("temporal-sampling request differs from the declared MoCA ablation")
    base = yaml.safe_load(Path(args.base_config).read_text())
    clip_length = int(base["clip"]["length"])
    plans = []
    for seed in args.seeds:
        for system in args.systems:
            for stride in args.source_frame_strides:
                cadence = "D1" if stride == 1 else "S5"
                plans.append({
                    "run_id": f"moca_mask_dense__{system}__{cadence}__seed{seed}",
                    "dataset": "moca_mask_dense", "system": system, "seed": seed,
                    "source_frame_stride": stride, "source_frame_span": (clip_length - 1) * stride,
                    "n_observations": clip_length,
                    "context_cadence": "dense_source_stride1" if stride == 1 else "source_stride5",
                    "reuse_primary": stride == 1,
                })
    print(json.dumps(plans, indent=2))
    if args.dry_run:
        return
    executions = [plan for plan in plans if not plan["reuse_primary"]]
    failures = []
    for plan in tqdm(executions, desc="MoCA temporal sampling", unit="run", dynamic_ncols=True):
        command = [
            sys.executable, "scripts/train_probe.py", "--config", args.base_config,
            f"experiment.system_id={plan['system']}", f"experiment.seed={plan['seed']}",
            "dataset.name=moca_mask_dense", "dataset.regime=S5",
            "dataset.manifest_env=MOCA_MASK_DENSE_MANIFEST",
            "dataset.release_profile=moca_mask_dense_v1",
            "dataset.boundary_policy=manual_target_hull_v1",
            "dataset.dense_intermediate_rgb_available=true",
            "clip.stride=5", "clip.released_stride=5", "clip.source_frame_stride=5",
            "clip.context_cadence=source_stride5",
        ]
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures.append(plan | {"returncode": result.returncode})
    if failures:
        raise SystemExit(f"{len(failures)} temporal-sampling runs failed: {failures}")


if __name__ == "__main__":
    main()
