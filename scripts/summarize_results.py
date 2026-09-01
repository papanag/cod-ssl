#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from tqdm.auto import tqdm

from cod_ssl.evaluation.statistics import (
    paired_regime_interaction,
    paired_video_bootstrap,
)

PRIMARY_CELLS = {
    ("moca_mask", "default"),
    ("camovid60k", "small_displacement"),
    ("camovid60k", "large_displacement"),
}
SYSTEMS = ("DS", "VI", "DT", "VV")
METRICS = ("S", "weightedF", "MAE", "E_adapt", "E_mean", "E_max")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--runs-root", required=True)
    parser.add_argument("--matrix", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); root, output = Path(args.runs_root), Path(args.output)
    records, video_tables = [], {}
    summary_paths = sorted(root.rglob("summary.json"))
    for summary_path in tqdm(
        summary_paths, desc="loading completed runs", unit="run", dynamic_ncols=True
    ):
        summary = json.loads(summary_path.read_text()); run = summary["run"]
        key = (run["dataset"], run.get("regime") or "default", run["system_id"], int(run["seed"]))
        if key in video_tables: raise ValueError(f"duplicate completed cell: {key}")
        video_path = summary_path.parent / "per_video.csv"
        if not video_path.is_file(): raise FileNotFoundError(f"missing paired video artifact: {video_path}")
        video_tables[key] = pd.read_csv(video_path)
        metrics = summary["metrics"]["minmax"]
        records.append({**dict(zip(("dataset", "regime", "system_id", "seed"), key)),
                        **{f"frame_{name}": value for name, value in metrics["frame_weighted"].items()},
                        **{f"video_{name}": value for name, value in metrics["video_weighted"].items()}})
    if not records: raise ValueError("no completed VCOD summaries found")
    seeds = sorted({record["seed"] for record in records})
    required = {(dataset, regime, system, seed) for dataset, regime in PRIMARY_CELLS
                for system in SYSTEMS for seed in seeds}
    missing = required - video_tables.keys()
    if missing: raise ValueError(f"missing primary cells: {sorted(missing)}")
    comparisons = []
    comparison_jobs = [
        (dataset, regime, seed, left, right, label, metric)
        for dataset, regime in sorted(PRIMARY_CELLS)
        for seed in seeds
        for left, right, label in (("DS", "VI", "spatial_VI_minus_DS"),
                                   ("DT", "VV", "video_VV_minus_DT"),
                                   ("DS", "DT", "dino_temporal_gain"),
                                   ("VI", "VV", "vjepa_temporal_gain"))
        for metric in METRICS
    ]
    for dataset, regime, seed, left, right, label, metric in tqdm(
        comparison_jobs, desc="paired video bootstraps", unit="comparison",
        dynamic_ncols=True,
    ):
        result = paired_video_bootstrap(
            video_tables[(dataset, regime, left, seed)],
            video_tables[(dataset, regime, right, seed)],
            metric,
        )
        comparisons.append(
            {
                "dataset": dataset,
                "regime": regime,
                "seed": seed,
                "comparison": label,
                **result,
            }
        )
    interactions = []
    interaction_jobs = [(seed, metric) for seed in seeds for metric in METRICS]
    for seed, metric in tqdm(
        interaction_jobs, desc="motion interactions", unit="interaction",
        dynamic_ncols=True,
    ):
        interactions.append({"seed": seed, **paired_regime_interaction(
                video_tables[("camovid60k", "small_displacement", "DT", seed)],
                video_tables[("camovid60k", "small_displacement", "VV", seed)],
                video_tables[("camovid60k", "large_displacement", "DT", seed)],
                video_tables[("camovid60k", "large_displacement", "VV", seed)], metric)})
    output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(records).sort_values(["dataset", "regime", "seed", "system_id"])
    results.to_csv(output / "primary_results.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "paired_comparisons.csv", index=False)
    pd.DataFrame(interactions).to_csv(output / "motion_interactions.csv", index=False)
    payload = {"primary_results": records, "paired_comparisons": comparisons,
               "motion_interactions": interactions,
               "within_family_warning": "DT-DS includes supervised GMMix adaptation; VV-VI switches pretrained pathways and the gains are not equivalent measures of temporal information."}
    (output / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    static = yaml.safe_load(Path("configs/static_reference_results.yaml").read_text())["results"]
    lines = ["# Controlled VCOD representation comparison", "", "## Protocol", "",
             "Primary endpoint: video-weighted, per-image-min-max S-measure. Video bootstrap is paired by source video.", "",
             "## Static COD reference", "", f"Registry: `{json.dumps(static)}`", "",
             "## Primary matrix", "", results.to_markdown(index=False), "", "## Paired comparisons", "",
             pd.DataFrame(comparisons).to_markdown(index=False), "", "## Motion-regime interaction", "",
             pd.DataFrame(interactions).to_markdown(index=False), "", "## Adaptation asymmetry", "",
             payload["within_family_warning"], "", "## Diagnostics, efficiency, qualitative review, and deviations", "",
             "Populate from completed optional runs and signed manual review artifacts; no values are copied by hand."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__": main()
