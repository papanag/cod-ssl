#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cod_ssl.data import CODDataset
from cod_ssl.data.clip_sampler import ClipSpec
from cod_ssl.data.collate import video_collate
from cod_ssl.data.camotion_attributes import ATTRIBUTE_CODES
from cod_ssl.data.video_manifest import ManifestVideoCODDataset
from cod_ssl.engine import Evaluator
from cod_ssl.engine.train import select_amp
from cod_ssl.evaluation.video_predictions import logits_to_float_views
from cod_ssl.metrics import CODMetrics
from cod_ssl.metrics.aggregation import aggregate_frame_and_video, per_video_table
from cod_ssl.models import build_frozen_cod_model, build_video_cod_model
from cod_ssl.utils.reproducibility import seed_everything
from cod_ssl.utils.run import file_sha256


def evaluate_video_run(
    run_dir: Path, checkpoint_arg: str | None, split: str, save_logits: bool
) -> None:
    config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text())
    seed_everything(int(config["experiment"]["seed"]))
    manifest = os.environ.get(
        config["dataset"]["manifest_env"],
        config["dataset"].get("split_manifest", ""),
    )
    if not manifest:
        raise FileNotFoundError(f"set {config['dataset']['manifest_env']} for video evaluation")
    if config["dataset"]["name"] == "moca_mask_dense":
        from cod_ssl.data.preprocessing.prepare_moca_mask_dense import verify_moca_mask_dense
        manifest_path = Path(manifest).resolve()
        if manifest_path.name != "runtime_manifest.csv" or manifest_path.parent.name != "manifest":
            raise ValueError("dense MoCA evaluation requires the verified moca_mask_dense_v1 runtime manifest")
        verify_moca_mask_dense(manifest_path.parent.parent)
        run_manifest_hash = file_sha256(manifest_path.parent / "manifest_checksums.sha256")
    else:
        run_manifest_hash = file_sha256(manifest)
    clip = config["clip"]
    system = config["experiment"]["system_id"]
    sample_spec = (
        ClipSpec(1, 1, 0)
        if system in {"DS", "VI"}
        else ClipSpec(
            int(clip["length"]), int(clip["stride"]), int(clip["target_index"])
        )
    )
    dataset = ManifestVideoCODDataset(
        manifest,
        split=split,
        clip_spec=sample_spec,
        training=False,
        size=int(config["backbone"]["input_size"][0]),
        regime=config["dataset"]["regime"],
        context_cadence=clip.get("context_cadence"),
        source_frame_step=clip.get("source_frame_stride"),
        release_profile=config["dataset"].get("release_profile"),
        boundary_policy=config["dataset"].get("boundary_policy"),
        temporal_order=(
            "repeated" if config.get("input_treatment") == "repeated_target"
            else clip.get("order_mode", "ordered")
        ),
        diagnostic_seed=int(clip.get("diagnostic_seed", config["experiment"]["seed"])),
        context_direction=clip.get("context_direction", "bidirectional"),
        filter_regime=config["dataset"]["name"] not in {"moca_mask_dense", "camotion"},
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=video_collate
    )
    if not len(loader):
        raise ValueError("video evaluation manifest contains no targets for this split/regime")
    model = build_video_cod_model(config)
    if checkpoint_arg:
        checkpoint = Path(checkpoint_arg)
    else:
        best = run_dir / "checkpoints" / "best.pt"
        checkpoint = best if best.is_file() else run_dir / "checkpoints" / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"video checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload = state.get("readout", state)
    missing, unexpected = model.load_state_dict(payload, strict=False)
    if unexpected or any(not key.startswith("backbone.") for key in missing):
        raise RuntimeError(f"invalid readout checkpoint keys: missing={missing}, unexpected={unexpected}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    model.assert_gradient_contract()
    prediction_dir = run_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "evaluation_progress.json"
    evaluation_started = time.perf_counter()
    rows, keys = [], set()
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"evaluate {system} {config['dataset']['name']}/{config['dataset']['regime']}",
        unit="target",
        dynamic_ncols=True,
    )
    for batch in progress:
        for key in ("frames", "target_mask", "valid_temporal_mask"):
            batch[key] = batch[key].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000
        mask_path = batch["metadata"]["mask_path"][0]
        with Image.open(mask_path) as raw_mask:
            ground_truth = np.asarray(raw_mask.convert("L"), dtype=np.uint8)
        views = logits_to_float_views(output["logits"], ground_truth.shape)
        metric = CODMetrics()
        metric.step(views["minmax_uint8"], ground_truth)
        values = metric.compute()
        source_id, frame_id = batch["source_video_id"][0], batch["frame_id"][0]
        key = (batch["video_id"][0], int(batch["frame_number"][0]))
        if key in keys:
            raise ValueError(f"duplicate evaluation target key: {key}")
        keys.add(key)
        if save_logits or config["evaluation"]["save_logits"]:
            np.savez_compressed(
                prediction_dir / f"{source_id}__{frame_id}.npz",
                logits=output["logits"].float().cpu().numpy(),
                sigmoid_raw=views["sigmoid_raw"],
                minmax=views["minmax"],
            )
        target_binary = ground_truth > 0
        attribute_vector = {
            code: bool(batch["attributes"].get(code, torch.tensor([False]))[0])
            for code in ATTRIBUTE_CODES
        }
        attribute_scope = batch["metadata"].get("attribute_scope")
        attribute_scope = None if attribute_scope is None else attribute_scope[0]
        row = {
            "run_id": run_dir.name, "system_id": config["experiment"]["system_id"],
            "dataset": config["dataset"]["name"], "regime": config["dataset"]["regime"],
            "split": split, "seed": config["experiment"]["seed"],
            "video_id": batch["video_id"][0], "source_video_id": source_id,
            "benchmark_sequence_id": batch["video_id"][0],
            "source_sequence_id": source_id,
            "frame_id": frame_id, "frame_number": int(batch["frame_number"][0]),
            "source_frame_number": int(batch["source_frame_number"][0]),
            "sequence_position": int(batch["sequence_position"][0]),
            "annotation_type": batch["annotation_type"][0],
            "target_index": int(batch["target_index"][0]),
            "source_frame_indices": json.dumps([int(value[0]) for value in batch["source_frame_indices"]]),
            "source_sequence_positions": json.dumps(
                [int(value[0]) for value in batch["source_sequence_positions"]]
            ),
            "release_profile": batch["release_profile"][0],
            "context_cadence": batch["context_cadence"][0],
            "released_frame_step": int(batch["released_frame_step"][0]),
            "source_frame_step": int(batch["source_frame_step"][0]),
            "dense_intermediate_rgb_available": bool(batch["dense_intermediate_rgb_available"][0]),
            "boundary_policy": batch["boundary_policy"][0],
            "context_direction": batch["context_direction"][0],
            "preprocessing_manifest_hash": run_manifest_hash,
            "foreground_fraction": float(target_binary.mean()), "motion_proxy": np.nan,
            "S": values["s_measure"], "E_adapt": values["e_adaptive"],
            "weightedF": values["weighted_f"], "MAE": values["mae"],
            "E_mean": values["e_mean"], "E_max": values["e_max"],
            "raw_mean_probability": float(views["sigmoid_raw"].mean()),
            "raw_max_probability": float(views["sigmoid_raw"].max()),
            "inference_ms": inference_ms,
            "attributes": json.dumps(attribute_vector, sort_keys=True),
            "attribute_scope": attribute_scope,
            **{f"attr_{code}": value for code, value in attribute_vector.items()},
        }
        rows.append(row)
        completed = len(rows)
        if completed % 25 == 0 or completed == len(loader):
            elapsed = time.perf_counter() - evaluation_started
            receipt = {
                "status": "running",
                "completed_targets": completed,
                "total_targets": len(loader),
                "current_video_id": source_id,
                "current_frame_number": row["frame_number"],
                "elapsed_seconds": elapsed,
                "eta_seconds": elapsed / completed * (len(loader) - completed),
                "mean_inference_ms": float(
                    np.mean([item["inference_ms"] for item in rows])
                ),
            }
            temporary = progress_path.with_suffix(".json.part")
            temporary.write_text(json.dumps(receipt, indent=2) + "\n")
            temporary.replace(progress_path)
        progress.set_postfix(
            video=source_id,
            frame=row["frame_number"],
            S=f"{row['S']:.3f}",
            ms=f"{inference_ms:.1f}",
            refresh=False,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "per_frame.csv", index=False)
    videos = per_video_table(frame)
    videos.to_csv(run_dir / "per_video.csv", index=False)
    aggregates = aggregate_frame_and_video(frame)
    attribute_results = {}
    for code in ATTRIBUTE_CODES:
        subset = frame[frame[f"attr_{code}"]]
        if not subset.empty:
            subset_aggregates = aggregate_frame_and_video(subset)
            attribute_results[code] = {
                "n_videos": int(subset.source_video_id.nunique()),
                "n_targets": len(subset),
                "frame_weighted_official_compatible": subset_aggregates["frame_weighted"],
                "video_weighted_study_primary": subset_aggregates["video_weighted"],
            }
    attribute_manifest = Path(manifest).with_name("camotion.attributes.json")
    source_stride = int(clip.get("source_frame_stride", 1))
    preprocessing_hash = None
    derived_context_frames = None
    original_moca_rgb_frames = None
    if config["dataset"]["name"] == "moca_mask_dense":
        preprocessing_hash = file_sha256(Path(manifest).resolve().parent / "manifest_checksums.sha256")
        release_manifest = json.loads((Path(manifest).resolve().parent / "release_manifest.json").read_text())
        derived_context_frames = release_manifest["derived"]["derived_dense_context_frames"]
        original_moca_rgb_frames = release_manifest["original_moca"]["discovered_rgb_frames"]
    dataset_release = {
        "release_profile": config["dataset"].get("release_profile"),
        "paper_reported_frames": 149_319 if config["dataset"]["name"] == "camotion" else 22_939,
        "released_unique_rgb": 30_028 if config["dataset"]["name"] == "camotion" else 4_691,
        "manual_targets": 30_028 if config["dataset"]["name"] == "camotion" else 4_691,
        "dense_intermediate_rgb_available": config["dataset"]["name"] == "moca_mask_dense",
        "flattened_rgb_gt_are_duplicates": config["dataset"]["name"] == "camotion",
        "preprocessing_manifest_hash": preprocessing_hash,
        "boundary_policy": config["dataset"].get("boundary_policy"),
        "derived_dense_context_frames": derived_context_frames,
        "original_moca_rgb_frames": original_moca_rgb_frames,
    }
    n_observations = 1 if system in {"DS", "VI"} else int(clip["length"])
    clip_summary = dict(clip) | {
        "source_frame_span": (n_observations - 1) * source_stride,
        "n_observations": n_observations,
    }
    summary = {
        "schema_version": 3,
        "run": {"run_id": run_dir.name, "system_id": config["experiment"]["system_id"],
                "dataset": config["dataset"]["name"], "regime": config["dataset"]["regime"],
                "seed": config["experiment"]["seed"]},
        "representation": {"backbone": config["backbone"]["name"],
                           "pathway": config["pathway"], "feature_layer": config["backbone"]["feature_layer"],
                           "temporal_adapter": config["temporal_adapter"]["name"],
                           "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                           "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad)},
        "clip": clip_summary,
        "dataset_release": dataset_release,
        "dataset_metadata": {
            "attribute_scope": "sequence" if config["dataset"]["name"] == "camotion" else None,
            "attribute_codes": list(ATTRIBUTE_CODES) if config["dataset"]["name"] == "camotion" else [],
            "attribute_manifest_sha256": (
                file_sha256(attribute_manifest) if attribute_manifest.is_file() else None
            ),
            "usage": "academic_research_only" if config["dataset"]["name"] == "camotion" else None,
        },
        "metrics": {"minmax": {
            "frame_weighted_official_compatible": aggregates["frame_weighted"],
            "video_weighted_study_primary": aggregates["video_weighted"],
            "by_attribute": attribute_results,
        }, "sigmoid_raw_diagnostics": {
            "mean_probability": float(frame.raw_mean_probability.mean())}},
        "prediction_view": "minmax",
        "timing": {"mode": "cold", "ms_per_output_frame": float(frame.inference_ms.mean()),
                   "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0)},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    final_progress = json.loads(progress_path.read_text())
    final_progress["status"] = "complete"
    final_progress["eta_seconds"] = 0.0
    progress_path.write_text(json.dumps(final_progress, indent=2) + "\n")
    (run_dir / "EVALUATION_COMPLETE").write_text("complete\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run")
    parser.add_argument("--run-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset", choices=["camo_test", "cod10k_test", "chameleon", "nc4k"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--save-logits", action="store_true")
    args = parser.parse_args()

    if args.run_dir:
        evaluate_video_run(Path(args.run_dir), args.checkpoint, args.split, args.save_logits)
        return
    if not args.run:
        parser.error("one of --run or --run-dir is required")

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
    for dataset_name in tqdm(
        dataset_names, desc="evaluation datasets", unit="dataset", dynamic_ncols=True
    ):
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
