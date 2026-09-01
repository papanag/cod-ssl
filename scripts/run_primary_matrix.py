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
    "moca_mask": ("moca_mask", "default"),
    "camovid60k_small": ("camovid60k", "small_displacement"),
    "camovid60k_large": ("camovid60k", "large_displacement"),
}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", nargs="+", required=True); parser.add_argument("--systems", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); base = load_config(args.config)
    runs = []
    for dataset_key in args.datasets:
        if dataset_key not in REGIMES: raise ValueError(f"unknown dataset/regime: {dataset_key}")
        dataset, regime = REGIMES[dataset_key]
        for system in args.systems:
            for seed in args.seeds:
                config = configure_system(base, system); config["dataset"]["name"] = dataset
                config["dataset"]["regime"] = regime; config["experiment"]["seed"] = seed
                if dataset == "camovid60k":
                    config["dataset"]["root_env"] = "CAMOVID60K_ROOT"
                    config["dataset"]["manifest_env"] = "CAMOVID60K_MANIFEST"
                validate_vcod_config(config)
                digest = hashlib.sha256(yaml.safe_dump(config, sort_keys=True).encode()).hexdigest()
                run_id = f"{dataset_key}__{system}__seed{seed}"
                runs.append({"run_id": run_id, "config_sha256": digest, "dataset": dataset,
                             "regime": regime, "system": system, "seed": seed})
    if len({run["run_id"] for run in runs}) != len(runs): raise ValueError("duplicate run IDs")
    print(json.dumps(runs, indent=2))
    if args.dry_run: return
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
                   f"dataset.manifest_env={'CAMOVID60K_MANIFEST' if run['dataset'] == 'camovid60k' else 'MOCA_MASK_MANIFEST'}"]
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures.append(run | {"returncode": result.returncode})
            progress.write(f"FAILED {run['run_id']} (exit {result.returncode})")
        else:
            progress.write(f"COMPLETED {run['run_id']}")
    if failures:
        Path("outputs").mkdir(exist_ok=True); Path("outputs/matrix_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise SystemExit(f"{len(failures)} matrix runs failed; successful runs were retained")


if __name__ == "__main__": main()
