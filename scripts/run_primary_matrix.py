#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml
from tqdm.auto import tqdm

from cod_ssl.utils.config import load_config
from cod_ssl.utils.vcod_config import configure_system, validate_vcod_config

REGIMES = {
    "moca_mask_dense": ("moca_mask_dense", "D1"),
    "camotion": ("camotion", "S5"),
}

MANIFEST_ENVS = {
    "moca_mask_dense": "MOCA_MASK_DENSE_MANIFEST",
    "camotion": "CAMOTION_MANIFEST",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--systems", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base = load_config(args.config)
    runs = []
    for dataset_key in args.datasets:
        if dataset_key not in REGIMES:
            raise ValueError(f"unknown dataset/regime: {dataset_key}")
        dataset, regime = REGIMES[dataset_key]
        for system in args.systems:
            for seed in args.seeds:
                config = configure_system(base, system)
                config["dataset"]["name"] = dataset
                config["dataset"]["regime"] = regime
                config["experiment"]["seed"] = seed
                if dataset == "camotion":
                    config["dataset"]["manifest_env"] = "CAMOTION_MANIFEST"
                    config["dataset"]["release_profile"] = "camotion_public_stride5_v1"
                    config["dataset"]["boundary_policy"] = "public_sequence_extent_v1"
                    config["dataset"]["dense_intermediate_rgb_available"] = False
                    config["dataset"]["expected_targets"] = 30_028
                    config["clip"].update({"stride": 1, "released_stride": 1,
                                           "source_frame_stride": 5,
                                           "context_cadence": "source_stride5"})
                else:
                    config["dataset"]["manifest_env"] = "MOCA_MASK_DENSE_MANIFEST"
                    config["dataset"]["release_profile"] = "moca_mask_dense_v1"
                    config["dataset"]["boundary_policy"] = "manual_target_hull_v1"
                    config["dataset"]["dense_intermediate_rgb_available"] = True
                    config["clip"].update({"stride": 1, "released_stride": 1,
                                           "source_frame_stride": 1,
                                           "context_cadence": "dense_source_stride1"})
                validate_vcod_config(config)
                digest = hashlib.sha256(yaml.safe_dump(config, sort_keys=True).encode()).hexdigest()
                run_id = f"{dataset_key}__{system}__{regime}__seed{seed}"
                runs.append({"run_id": run_id, "config_sha256": digest, "dataset": dataset,
                             "regime": regime, "system": system, "seed": seed,
                             "release_profile": config["dataset"]["release_profile"],
                             "boundary_policy": config["dataset"]["boundary_policy"],
                             "dense_intermediate_rgb_available": config["dataset"]["dense_intermediate_rgb_available"],
                             "expected_targets": config["dataset"]["expected_targets"],
                             "released_stride": config["clip"]["released_stride"],
                             "source_frame_stride": config["clip"]["source_frame_stride"],
                             "context_cadence": config["clip"]["context_cadence"]})
    if len({run["run_id"] for run in runs}) != len(runs):
        raise ValueError("duplicate run IDs")
    print(json.dumps(runs, indent=2))
    if args.dry_run:
        return
    failures = []
    progress = tqdm(runs, desc="primary matrix", unit="run", dynamic_ncols=True)
    for run in progress:
        progress.set_postfix(
            dataset=run["dataset"], regime=run["regime"],
            system=run["system"], seed=run["seed"], refresh=True,
        )
        command = [sys.executable, "scripts/train_probe.py", "--config", args.config,
                   f"experiment.system_id={run['system']}", f"experiment.seed={run['seed']}",
                   f"dataset.name={run['dataset']}", f"dataset.regime={run['regime']}",
                   f"dataset.manifest_env={MANIFEST_ENVS[run['dataset']]}",
                   f"dataset.release_profile={run['release_profile']}",
                   f"dataset.boundary_policy={run['boundary_policy']}",
                   f"dataset.dense_intermediate_rgb_available={str(run['dense_intermediate_rgb_available']).lower()}",
                   f"dataset.expected_targets={run['expected_targets']}",
                   f"clip.stride={run['released_stride']}",
                   f"clip.released_stride={run['released_stride']}",
                   f"clip.source_frame_stride={run['source_frame_stride']}",
                   f"clip.context_cadence={run['context_cadence']}"]
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures.append(run | {"returncode": result.returncode})
            progress.write(f"FAILED {run['run_id']} (exit {result.returncode})")
        else:
            progress.write(f"COMPLETED {run['run_id']}")
    if failures:
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/matrix_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise SystemExit(f"{len(failures)} matrix runs failed; successful runs were retained")


if __name__ == "__main__":
    main()
