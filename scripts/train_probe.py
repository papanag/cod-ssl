#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from cod_ssl.data.clip_sampler import ClipSpec
from cod_ssl.data.collate import video_collate
from cod_ssl.data.video_manifest import ManifestVideoCODDataset, video_balanced_indices
from cod_ssl.losses import BCEDiceLoss
from cod_ssl.models import build_video_cod_model
from cod_ssl.utils.config import load_config
from cod_ssl.utils.reproducibility import seed_everything
from cod_ssl.utils.run import file_sha256, git_commit
from cod_ssl.utils.vcod_config import configure_system, validate_vcod_config


def _parse_value(raw: str):
    return yaml.safe_load(raw)


def _override(config: dict, expression: str) -> None:
    key, separator, raw = expression.partition("=")
    if not separator: raise ValueError(f"override must be key=value: {expression}")
    target = config
    parts = key.split(".")
    for part in parts[:-1]: target = target.setdefault(part, {})
    target[parts[-1]] = _parse_value(raw)


def _run_id(config: dict) -> str:
    dataset = config["dataset"]["name"]
    regime = config["dataset"]["regime"]
    prefix = dataset if regime in {None, "default"} else f"{dataset}_{regime}"
    system = config["experiment"]["system_id"]
    adapter = config["temporal_adapter"]["name"]
    clip = config["clip"]
    treatment = "single" if system in {"DS", "VI"} else f"{adapter}__T{clip['length']}_S{clip['stride']}_target{clip['target_index']}"
    cadence = config["dataset"].get("regime") or config["clip"].get("context_cadence", "unspecified")
    return f"{prefix}__{system}__{config['backbone']['name']}__{treatment}__{cadence}__seed{config['experiment']['seed']}"


def _config_sha256(config: dict) -> str:
    return hashlib.sha256(yaml.safe_dump(config, sort_keys=True).encode()).hexdigest()


def _resume_compatibility_sha256(config: dict) -> str:
    """Hash state-affecting configuration while allowing a larger step target."""
    compatible = deepcopy(config)
    compatible["training"].pop("max_steps", None)
    return _config_sha256(compatible)


def _save_checkpoint(
    target: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    config_sha256: str,
    resume_compatibility_sha256: str,
) -> None:
    readout = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("backbone.")
    }
    payload = {
        "readout": readout,
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "config_sha256": config_sha256,
        "resume_compatibility_sha256": resume_compatibility_sha256,
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_states": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }
    temporary = target.with_suffix(target.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--run-dir")
    parser.add_argument("--resume"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config)
    for expression in args.overrides: _override(config, expression)
    system = config["experiment"]["system_id"]
    config = configure_system(config, system); validate_vcod_config(config)
    config_sha256 = _config_sha256(config)
    resume_compatibility_sha256 = _resume_compatibility_sha256(config)
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        checkpoint_compatibility = resume_state.get("resume_compatibility_sha256")
        if checkpoint_compatibility != resume_compatibility_sha256:
            raise ValueError(
                "resume checkpoint is incompatible with this run; only training.max_steps "
                "may change between tuning stages"
            )
    seed_everything(int(config["experiment"]["seed"]))
    manifest_value = os.environ.get(config["dataset"]["manifest_env"], config["dataset"].get("split_manifest", ""))
    if not manifest_value or not Path(manifest_value).is_file():
        raise FileNotFoundError(f"set {config['dataset']['manifest_env']} to a validated canonical manifest")
    if config["dataset"]["name"] == "moca_mask_dense":
        from cod_ssl.data.preprocessing.prepare_moca_mask_dense import (
            verify_moca_mask_dense,
        )
        manifest_path = Path(manifest_value).resolve()
        if manifest_path.name != "runtime_manifest.csv" or manifest_path.parent.name != "manifest":
            raise ValueError(
                "dense MoCA requires processed/moca_mask_dense_v1/manifest/runtime_manifest.csv; "
                "run scripts/prepare_moca_mask_dense.py first"
            )
        # Notebook 05 performs the one-time linked-asset audit. Per-run checks
        # validate its checksummed manifests without rehashing Drive files.
        verify_moca_mask_dense(manifest_path.parent.parent, verify_linked_targets=False)
    clip = config["clip"]
    sample_spec = (ClipSpec(1, 1, 0) if system in {"DS", "VI"} else
                   ClipSpec(int(clip["length"]), int(clip["stride"]), int(clip["target_index"])))
    dataset = ManifestVideoCODDataset(
        manifest_value, split="train",
        clip_spec=sample_spec,
        training=True, size=int(config["backbone"]["input_size"][0]), regime=config["dataset"]["regime"],
        context_cadence=clip.get("context_cadence"),
        source_frame_step=clip.get("source_frame_stride"),
        release_profile=config["dataset"].get("release_profile"),
        boundary_policy=config["dataset"].get("boundary_policy"),
        temporal_order=(
            "repeated" if config.get("input_treatment") == "repeated_target"
            else config["clip"].get("order_mode", "ordered")
        ),
        diagnostic_seed=int(config["clip"].get("diagnostic_seed", config["experiment"]["seed"])),
        context_direction=config["clip"].get("context_direction", "bidirectional"),
        filter_regime=config["dataset"]["name"] not in {"moca_mask_dense", "camotion"},
    )
    training_subset_receipt = None
    if args.smoke:
        dataset = Subset(dataset, range(min(4, len(dataset))))
    elif training_limit := config["training"].get("limit_targets"):
        training_limit = int(training_limit)
        if training_limit < 1:
            raise ValueError("training.limit_targets must be positive")
        subset_seed = int(config["training"].get("subset_seed", config["experiment"]["seed"]))
        target_video_ids = dataset.target_video_ids
        selected_indices = video_balanced_indices(
            target_video_ids, training_limit, seed=subset_seed
        )
        selected_video_ids = [target_video_ids[index] for index in selected_indices]
        training_subset_receipt = {
            "sampling": "deterministic_video_balanced",
            "seed": subset_seed,
            "limit": training_limit,
            "selected_indices": selected_indices,
            "selected_targets": len(selected_indices),
            "selected_source_videos": len(set(selected_video_ids)),
        }
        dataset = Subset(dataset, selected_indices)
    training = config["training"]
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True,
                        num_workers=(0 if args.smoke else int(training["num_workers"])),
                        pin_memory=torch.cuda.is_available(), collate_fn=video_collate)
    model = build_video_cod_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).train(); model.assert_gradient_contract()
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad),
                      lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    run_dir = Path(args.run_dir or Path(config["experiment"]["output_root"]) / _run_id(config))
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    splits = {split: sorted(set(map(str, group.source_video_id))) for split, group in __import__("pandas").read_csv(manifest_value).groupby("split")}
    (run_dir / "split_ids.json").write_text(json.dumps({"ids": splits, "manifest_sha256": file_sha256(manifest_value)}, indent=2) + "\n")
    (run_dir / "environment.json").write_text(json.dumps({"torch": torch.__version__, "device": str(device),
        "git_commit": git_commit(Path.cwd()), "seed": config["experiment"]["seed"]}, indent=2) + "\n")
    if training_subset_receipt is not None:
        training_subset_receipt["manifest_sha256"] = file_sha256(manifest_value)
        (run_dir / "training_subset.json").write_text(
            json.dumps(training_subset_receipt, indent=2) + "\n"
        )
    loss_fn = BCEDiceLoss(); max_steps = 1 if args.smoke else int(training["max_steps"])
    global_step, gradient_checked = 0, False
    log_path = run_dir / "train_log.jsonl"
    if resume_state is not None:
        missing, unexpected = model.load_state_dict(resume_state["readout"], strict=False)
        if unexpected or any(not key.startswith("backbone.") for key in missing):
            raise RuntimeError(f"invalid readout checkpoint keys: missing={missing}, unexpected={unexpected}")
        optimizer.load_state_dict(resume_state["optimizer"])
        global_step = int(resume_state["global_step"])
        torch.set_rng_state(resume_state["torch_random_state"].cpu())
        if device.type == "cuda" and resume_state.get("cuda_random_states"):
            torch.cuda.set_rng_state_all(resume_state["cuda_random_states"])
    checkpoint_every = int(training.get("checkpoint_every_steps", 250))
    if checkpoint_every < 1:
        raise ValueError("training.checkpoint_every_steps must be positive")
    if resume_state is not None and global_step > max_steps:
        raise ValueError(
            f"checkpoint step {global_step} exceeds requested target step {max_steps}"
        )
    gradient_accumulation = int(training.get("gradient_accumulation", 1))
    if gradient_accumulation < 1:
        raise ValueError("training.gradient_accumulation must be positive")
    started = time.perf_counter()
    session_start_step = global_step
    running_loss = 0.0
    micro_step = 0
    accumulated_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        total=max_steps,
        initial=global_step,
        desc=f"{system} {config['dataset']['name']}/{config['dataset']['regime']}",
        unit="step",
        dynamic_ncols=True,
    )
    try:
        while global_step < max_steps:
            for batch in loader:
                for key in ("frames", "target_mask", "valid_temporal_mask"): batch[key] = batch[key].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    output = model(batch); loss = loss_fn(output["logits"], batch["target_mask"])
                if not torch.isfinite(loss): raise FloatingPointError("non-finite VCOD training loss")
                (loss / gradient_accumulation).backward()
                if not gradient_checked:
                    model.assert_gradient_contract(after_backward=True); gradient_checked = True
                micro_step += 1
                accumulated_loss += float(loss.detach())
                if micro_step % gradient_accumulation:
                    continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                loss_value = accumulated_loss / gradient_accumulation
                accumulated_loss = 0.0
                running_loss += loss_value
                elapsed = time.perf_counter() - started
                completed_this_session = max(1, global_step - session_start_step)
                seconds_per_step = elapsed / completed_this_session
                eta_seconds = seconds_per_step * (max_steps - global_step)
                peak_memory_mb = (
                    torch.cuda.max_memory_allocated(device) / 2**20
                    if device.type == "cuda" else 0.0
                )
                log_row = {
                    "global_step": global_step,
                    "loss": loss_value,
                    "mean_session_loss": running_loss / completed_this_session,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta_seconds,
                    "peak_gpu_memory_mb": peak_memory_mb,
                }
                with log_path.open("a") as handle:
                    handle.write(json.dumps(log_row) + "\n")
                progress.update(1)
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    mean=f"{log_row['mean_session_loss']:.4f}",
                    lr=f"{log_row['learning_rate']:.2e}",
                    gpu=f"{peak_memory_mb:.0f}MiB" if device.type == "cuda" else "cpu",
                    refresh=False,
                )
                if global_step % checkpoint_every == 0:
                    _save_checkpoint(
                        run_dir / "checkpoints" / "last.pt",
                        model,
                        optimizer,
                        global_step,
                        config_sha256,
                        resume_compatibility_sha256,
                    )
                    progress.write(f"Checkpoint saved at optimizer step {global_step}")
                    if hasattr(os, "sync"):
                        os.sync()
                if global_step >= max_steps: break
    finally:
        progress.close()
    checkpoint = run_dir / "checkpoints" / "last.pt"
    _save_checkpoint(
        checkpoint, model, optimizer, global_step, config_sha256,
        resume_compatibility_sha256,
    )
    (run_dir / "checkpoints.json").write_text(json.dumps({"last": str(checkpoint), "global_step": global_step}, indent=2) + "\n")
    stage_dir = run_dir / "stages" / f"step_{max_steps:06d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "TRAINING_COMPLETE").write_text("complete\n")
    (run_dir / "TRAINING_COMPLETE").write_text(
        json.dumps({"status": "complete", "global_step": global_step,
                    "target_step": max_steps}, indent=2) + "\n"
    )
    print(run_dir)


if __name__ == "__main__": main()
