#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from tqdm.auto import tqdm

from cod_ssl.data.camotion_attributes import ATTRIBUTE_CODES
from cod_ssl.evaluation.statistics import (
    paired_system_attribute_summary,
    paired_video_bootstrap,
    temporal_sampling_summary,
)

PRIMARY_CELLS = {("moca_mask_dense", "D1"), ("camotion", "S5")}
SYSTEMS = ("DS", "VI", "DT", "VV")
METRICS = ("S", "weightedF", "MAE", "E_adapt", "E_mean", "E_max")
HEADLINE_ATTRIBUTE_METRICS = ("S", "weightedF", "MAE")
ATTRIBUTE_NARRATIVE_ORDER = ("OC", "OV", "MB", "SO", "MO", "UE", "SC", "BO")
ATTRIBUTE_SCOPE_WARNING = (
    "CAMotion challenge labels are overlapping sequence-level annotations. "
    "Attribute-conditioned scores summarize sequences carrying a label; they do not isolate "
    "that attribute from co-occurring challenges, and they should not be interpreted as "
    "frame-level onset labels or causal effects."
)
ADAPTATION_WARNING = (
    "DT-DS includes supervised GatedMambaMix adaptation; VV-VI switches pretrained pathways "
    "and the gains are not equivalent measures of temporal information."
)


def _metric_views(summary: dict) -> tuple[dict, dict]:
    metrics = summary["metrics"]["minmax"]
    if summary.get("schema_version", 1) >= 2:
        return (
            metrics["frame_weighted_official_compatible"],
            metrics["video_weighted_study_primary"],
        )
    return metrics["frame_weighted"], metrics["video_weighted"]


def _flatten_attribute_result(result: dict, *, seed: int) -> dict:
    row = {
        "seed": seed, "subset": result["subset"], "metric": result["metric"],
        "n_videos": result["n_videos"], "n_targets": result["n_targets"],
        "inferential_status": result["inferential_status"],
    }
    row.update(result["system_means"])
    for name, values in result["contrasts"].items():
        row[name] = values["estimate"]
        row[f"{name}_ci95_low"] = values["ci95"][0]
        row[f"{name}_ci95_high"] = values["ci95"][1]
    return row


def _frame_weighted_attribute_row(
    tables: dict[str, pd.DataFrame], metric: str, subset: str | None, seed: int
) -> dict:
    reference = tables["DS"]
    selected = pd.Series(True, index=reference.index)
    if subset is not None:
        column = f"attr_{subset}"
        if any(column not in table for table in tables.values()):
            raise ValueError(f"missing frame-level CAMotion attribute column: {column}")
        vectors = [table[column].astype(bool).reset_index(drop=True) for table in tables.values()]
        if any(not vector.equals(vectors[0]) for vector in vectors[1:]):
            raise ValueError(f"frame-level propagated attributes differ across systems: {subset}")
        selected = reference[column].astype(bool)
    selected_tables = {system: table[selected.to_numpy()] for system, table in tables.items()}
    means = {system: float(table[metric].mean()) for system, table in selected_tables.items()}
    direction = -1.0 if metric == "MAE" else 1.0
    dino_gain = direction * (means["DT"] - means["DS"])
    vjepa_gain = direction * (means["VV"] - means["VI"])
    return {
        "seed": seed, "subset": subset or "All", "metric": metric,
        "aggregation": "frame_weighted_official_compatible",
        "n_videos": int(selected_tables["DS"][
            "video_id" if "video_id" in selected_tables["DS"] else "source_video_id"
        ].nunique()),
        "n_targets": len(selected_tables["DS"]),
        **means,
        "VI_minus_DS": direction * (means["VI"] - means["DS"]),
        "VV_minus_DT": direction * (means["VV"] - means["DT"]),
        "dino_gain": dino_gain, "vjepa_gain": vjepa_gain,
        "gain_advantage": vjepa_gain - dino_gain,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--temporal-sampling-ablation")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--attributes", nargs="+", choices=ATTRIBUTE_CODES, default=ATTRIBUTE_NARRATIVE_ORDER
    )
    parser.add_argument("--minimum-attribute-videos", type=int, default=5)
    args = parser.parse_args()
    root, output = Path(args.runs_root), Path(args.output)
    records, video_tables, frame_tables, release_metadata = [], {}, {}, {}
    for summary_path in tqdm(
        sorted(root.rglob("summary.json")), desc="loading completed runs", unit="run",
        dynamic_ncols=True,
    ):
        summary = json.loads(summary_path.read_text())
        run = summary["run"]
        key = (run["dataset"], run.get("regime") or "default", run["system_id"], int(run["seed"]))
        is_sampling_cell = key[0] == "moca_mask_dense" and key[1] == "S5" and key[2] in {"DT", "VV"}
        if key[:2] not in PRIMARY_CELLS and not is_sampling_cell:
            continue
        if key in video_tables:
            raise ValueError(f"duplicate completed cell: {key}")
        video_path = summary_path.parent / "per_video.csv"
        if not video_path.is_file():
            raise FileNotFoundError(f"missing paired video artifact: {video_path}")
        video_tables[key] = pd.read_csv(video_path)
        frame_path = summary_path.parent / "per_frame.csv"
        if not frame_path.is_file():
            raise FileNotFoundError(f"missing paired target artifact: {frame_path}")
        frame_tables[key] = pd.read_csv(frame_path)
        frame_metrics, video_metrics = _metric_views(summary)
        release_metadata[key] = summary.get("dataset_release", {})
        records.append({
            **dict(zip(("dataset", "regime", "system_id", "seed"), key)),
            **{f"frame_{name}": value for name, value in frame_metrics.items()},
            **{f"video_{name}": value for name, value in video_metrics.items()},
        })
    if not records:
        raise ValueError("no completed VCOD summaries found")
    seeds = sorted({record["seed"] for record in records})
    required = {
        (dataset, regime, system, seed)
        for dataset, regime in PRIMARY_CELLS for system in SYSTEMS for seed in seeds
    }
    missing = required - video_tables.keys()
    if missing:
        raise ValueError(f"missing primary cells: {sorted(missing)}")
    for dataset, regime in PRIMARY_CELLS:
        for seed in seeds:
            reference_keys = None
            for system in SYSTEMS:
                table = frame_tables[(dataset, regime, system, seed)]
                benchmark_key = "video_id" if "video_id" in table else "source_video_id"
                keys = list(zip(table[benchmark_key].astype(str), table.frame_number.astype(int)))
                if len(keys) != len(set(keys)):
                    raise ValueError(f"duplicate target keys in {(dataset, regime, system, seed)}")
                if reference_keys is None:
                    reference_keys = keys
                elif keys != reference_keys:
                    raise ValueError(
                        f"systems have different ordered target keys for {(dataset, regime, seed)}"
                    )
            dt = frame_tables[(dataset, regime, "DT", seed)]
            vv = frame_tables[(dataset, regime, "VV", seed)]
            if "source_frame_indices" in dt and not dt["source_frame_indices"].equals(
                vv["source_frame_indices"]
            ):
                raise ValueError(f"DT/VV source-frame lists differ for {(dataset, regime, seed)}")

    comparisons = []
    jobs = [
        (dataset, regime, seed, left, right, label, metric)
        for dataset, regime in sorted(PRIMARY_CELLS) for seed in seeds
        for left, right, label in (
            ("DS", "VI", "spatial_VI_minus_DS"),
            ("DT", "VV", "video_VV_minus_DT"),
            ("DS", "DT", "dino_temporal_gain"),
            ("VI", "VV", "vjepa_temporal_gain"),
        ) for metric in METRICS
    ]
    for dataset, regime, seed, left, right, label, metric in tqdm(
        jobs, desc="paired video bootstraps", unit="comparison", dynamic_ncols=True,
    ):
        result = paired_video_bootstrap(
            video_tables[(dataset, regime, left, seed)],
            video_tables[(dataset, regime, right, seed)], metric,
        )
        comparisons.append({
            "dataset": dataset, "regime": regime, "seed": seed,
            "comparison": label, **result,
        })

    temporal_sampling_rows = []
    if args.temporal_sampling_ablation:
        for seed in seeds:
            sampling_tables = {
                "DT_D1": video_tables[("moca_mask_dense", "D1", "DT", seed)],
                "DT_S5": video_tables[("moca_mask_dense", "S5", "DT", seed)],
                "VV_D1": video_tables[("moca_mask_dense", "D1", "VV", seed)],
                "VV_S5": video_tables[("moca_mask_dense", "S5", "VV", seed)],
            }
            for metric in METRICS:
                result = temporal_sampling_summary(sampling_tables, metric)
                temporal_sampling_rows.append({
                    "seed": seed, "metric": metric, "n_sequences": result["n_sequences"],
                    **{
                        name: result[name]["estimate"]
                        for name in ("C_D", "C_V", "Delta_C")
                    },
                    **{
                        f"{name}_ci95_{side}": result[name]["ci95"][index]
                        for name in ("C_D", "C_V", "Delta_C")
                        for index, side in enumerate(("low", "high"))
                    },
                })

    attribute_rows, frame_attribute_rows = [], []
    attribute_jobs = [
        (seed, metric, subset)
        for seed in seeds for metric in METRICS for subset in (None, *args.attributes)
    ]
    for seed, metric, subset in tqdm(
        attribute_jobs, desc="CAMotion attribute bootstraps", unit="subset", dynamic_ncols=True,
    ):
        tables = {
            system: video_tables[("camotion", "S5", system, seed)] for system in SYSTEMS
        }
        result = paired_system_attribute_summary(
            tables, metric, attribute=subset,
            minimum_videos=args.minimum_attribute_videos,
        )
        attribute_rows.append(_flatten_attribute_result(result, seed=seed))
        frame_attribute_rows.append(_frame_weighted_attribute_row(
            {system: frame_tables[("camotion", "S5", system, seed)] for system in SYSTEMS},
            metric, subset, seed,
        ))

    output.mkdir(parents=True, exist_ok=True)
    all_results = pd.DataFrame(records).sort_values(["dataset", "regime", "seed", "system_id"])
    results = all_results[
        all_results.apply(lambda row: (row.dataset, row.regime) in PRIMARY_CELLS, axis=1)
    ].copy()
    comparison_frame = pd.DataFrame(comparisons)
    attribute_frame = pd.DataFrame(attribute_rows)
    frame_attribute_frame = pd.DataFrame(frame_attribute_rows)
    results.to_csv(output / "primary_results.csv", index=False)
    all_results[
        (all_results.dataset == "moca_mask_dense")
        & (all_results.system_id.isin(["DT", "VV"]))
    ].to_csv(output / "moca_temporal_sampling_results.csv", index=False)
    pd.DataFrame(temporal_sampling_rows).to_csv(
        output / "moca_temporal_sampling_interactions.csv", index=False
    )
    comparison_frame.to_csv(output / "paired_comparisons.csv", index=False)
    attribute_frame.to_csv(output / "camotion_attribute_comparisons.csv", index=False)
    frame_attribute_frame.to_csv(
        output / "camotion_attribute_frame_weighted.csv", index=False
    )
    foreground_diagnostics = []
    for seed in seeds:
        for system in SYSTEMS:
            table = frame_tables[("camotion", "S5", system, seed)]
            if "foreground_fraction" not in table:
                continue
            foreground_diagnostics.append({
                "seed": seed, "system_id": system,
                "n_targets": len(table),
                "mean_foreground_fraction": float(table.foreground_fraction.mean()),
                "median_foreground_fraction": float(table.foreground_fraction.median()),
                "foreground_S_spearman": float(
                    table[["foreground_fraction", "S"]].corr(method="spearman").iloc[0, 1]
                ),
                "continuous_motion_status": "unavailable_dense_rgb_frames_absent_from_public_archive",
            })
    pd.DataFrame(foreground_diagnostics).to_csv(
        output / "camotion_foreground_motion_diagnostics.csv", index=False
    )
    qualitative_candidates = []
    for seed in seeds:
        tables = {
            system: frame_tables[("camotion", "S5", system, seed)].reset_index(drop=True)
            for system in SYSTEMS
        }
        base = tables["DS"][["source_video_id", "frame_id", "frame_number"]].copy()
        for system, table in tables.items():
            base[f"S_{system}"] = table["S"]
        for code in args.attributes:
            column = f"attr_{code}"
            if column not in tables["DS"]:
                continue
            subset = base[tables["DS"][column].astype(bool).to_numpy()].copy()
            subset["VV_minus_DT"] = subset["S_VV"] - subset["S_DT"]
            for direction, selected in (
                ("vjepa_relative_win", subset.nlargest(min(3, len(subset)), "VV_minus_DT")),
                ("vjepa_relative_failure", subset.nsmallest(min(3, len(subset)), "VV_minus_DT")),
            ):
                for row in selected.to_dict(orient="records"):
                    qualitative_candidates.append({
                        "seed": seed, "attribute": code, "selection": direction, **row
                    })
    pd.DataFrame(qualitative_candidates).to_csv(
        output / "camotion_qualitative_candidates.csv", index=False
    )
    payload = {
        "schema_version": 3,
        "primary_results": records,
        "paired_comparisons": comparisons,
        "camotion_attribute_comparisons": attribute_rows,
        "camotion_attribute_frame_weighted": frame_attribute_rows,
        "camotion_foreground_motion_diagnostics": foreground_diagnostics,
        "camotion_qualitative_candidates": qualitative_candidates,
        "moca_temporal_sampling_interactions": temporal_sampling_rows,
        "attribute_scope_warning": ATTRIBUTE_SCOPE_WARNING,
        "within_family_warning": ADAPTATION_WARNING,
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    static = yaml.safe_load(Path("configs/static_reference_results.yaml").read_text())["results"]
    moca_release = next(
        (value for key, value in release_metadata.items() if key[:2] == ("moca_mask_dense", "D1")), {}
    )
    camotion_release = next(
        (value for key, value in release_metadata.items() if key[:2] == ("camotion", "S5")), {}
    )
    derived_count = moca_release.get("derived_dense_context_frames", "generated from declared bounds")
    camotion_unique = camotion_release.get("released_unique_rgb", 30_028)
    camotion_targets = camotion_release.get("manual_targets", 30_028)
    moca_targets = moca_release.get("manual_targets", 4_691)
    original_moca_frames = moca_release.get("original_moca_rgb_frames", 37_250)
    lines = [
        "# Controlled VCOD representation comparison", "", "## Protocol", "",
        "Primary endpoint: benchmark-sequence-weighted, per-image-min-max S-measure. Bootstrap draws are paired by benchmark sequence.", "",
        "## Public release audit", "",
        "| Dataset/release | Paper-described frames | Public unique RGB | Manual masks | Public cadence |",
        "|---|---:|---:|---:|---|",
        f"| Original MoCA | about 37K | {original_moca_frames:,} | none | consecutive |",
        f"| MoCA-Mask public ZIP | 22,939 selected frames | {moca_targets:,} | {moca_targets:,} | sparse targets, generally every five |",
        f"| MoCA-Mask dense derived | not a separate publication | {derived_count:,} | {moca_targets:,} | consecutive context |" if isinstance(derived_count, int) else f"| MoCA-Mask dense derived | not a separate publication | {derived_count} | {moca_targets:,} | consecutive context |",
        f"| CAMotion public ZIP | 149,319 collected frames | {camotion_unique:,} | {camotion_targets:,} | every fifth original frame |", "",
        "The current MoCA-Mask archive exposes only manual target RGB/GT pairs. Dense context was reconstructed by content-verified alignment to Original MoCA and constrained by a versioned conservative subsequence-boundary policy. This is not claimed to reproduce an unavailable 22,939-frame release exactly.", "",
        "The public CAMotion archive contains 30,028 unique sequence-organized RGB/GT pairs sampled every five original frames. The advertised 149,319 frames describe the collected source material, not publicly available dense RGB context. Therefore CAMotion results measure coarse ordered temporal context.", "",
        "## Static COD reference", "", f"Registry: `{json.dumps(static)}`", "",
        "## Primary matrix", "", results.to_markdown(index=False), "",
        "## Paired comparisons", "", comparison_frame.to_markdown(index=False), "",
        "## CAMotion attribute scope", "", ATTRIBUTE_SCOPE_WARNING, "",
    ]
    if temporal_sampling_rows:
        lines.extend([
            "## MoCA source-stride-1 versus source-stride-5 temporal-sampling/coverage ablation", "",
            pd.DataFrame(temporal_sampling_rows).to_markdown(index=False), "",
            "At fixed clip length, D1 versus S5 changes both observation spacing and source-frame coverage; it is not a pure cadence or pretraining-effect estimate.", "",
        ])
    for metric in HEADLINE_ATTRIBUTE_METRICS:
        table = attribute_frame[attribute_frame.metric == metric]
        lines.extend([
            f"## CAMotion per-attribute comparison — {metric}", "",
            table.to_markdown(index=False), "",
        ])
        official_table = frame_attribute_frame[frame_attribute_frame.metric == metric]
        lines.extend([
            f"### CAMotion frame-weighted official-compatible — {metric}", "",
            official_table.to_markdown(index=False), "",
        ])
    lines.extend([
        "## CAMotion foreground-size and continuous-motion diagnostics", "",
        (pd.DataFrame(foreground_diagnostics).to_markdown(index=False)
         if foreground_diagnostics else "No foreground diagnostic columns were present."), "",
        "Continuous source-stride-1 motion diagnostics are unavailable because the public release contains only source-stride-5 observations; no intermediate values are fabricated.", "",
        "## Qualitative attribute-conditioned wins and failures", "",
        "`camotion_qualitative_candidates.csv` deterministically selects the three largest and smallest per-target VV-minus-DT S-measure gaps for every declared attribute and seed.", "",
        "## Adaptation asymmetry", "", ADAPTATION_WARNING, "",
        "## Diagnostics, efficiency, qualitative review, and deviations", "",
        "Populate from completed optional runs and signed manual review artifacts; no values are copied by hand.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
