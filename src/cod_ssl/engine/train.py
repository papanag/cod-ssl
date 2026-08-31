from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from cod_ssl.losses import BCESoftIoULoss
from cod_ssl.models import FrozenCODModel


@dataclass(frozen=True)
class TrainingOptions:
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    accumulation_steps: int = 4
    amp: bool = True
    amp_dtype: str = "auto"
    grad_clip_norm: float = 1.0
    checkpoint_every: int = 1

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.accumulation_steps < 1:
            raise ValueError("epochs and accumulation_steps must be positive")
        if self.amp_dtype not in {"auto", "fp16", "bf16", "fp32"}:
            raise ValueError(f"unsupported AMP dtype: {self.amp_dtype}")


def build_decoder_optimizer(
    model: FrozenCODModel, *, learning_rate: float, weight_decay: float
) -> AdamW:
    model.assert_backbone_frozen()
    optimizer = AdamW(
        model.decoder.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if backbone_ids & optimizer_ids:
        raise RuntimeError("optimizer contains a backbone parameter")
    decoder_ids = {id(parameter) for parameter in model.decoder.parameters()}
    if optimizer_ids != decoder_ids:
        raise RuntimeError("optimizer must contain every decoder parameter and nothing else")
    return optimizer


def select_amp(device: torch.device, enabled: bool, requested: str) -> tuple[bool, torch.dtype]:
    if not enabled or device.type != "cuda" or requested == "fp32":
        return False, torch.float32
    if requested == "bf16" or (requested == "auto" and torch.cuda.is_bf16_supported()):
        return True, torch.bfloat16
    return True, torch.float16


class Trainer:
    def __init__(
        self,
        model: FrozenCODModel,
        train_loader: DataLoader,
        run_dir: str | Path,
        options: TrainingOptions,
        *,
        device: torch.device | None = None,
        loss_fn: nn.Module | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).train()
        self.model.assert_backbone_frozen()
        self.train_loader = train_loader
        if len(train_loader) == 0:
            raise ValueError("training data loader is empty")
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.tensorboard_dir = self.run_dir / "tensorboard"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.options = options
        self.loss_fn = loss_fn or BCESoftIoULoss()
        self.optimizer = build_decoder_optimizer(
            self.model,
            learning_rate=options.learning_rate,
            weight_decay=options.weight_decay,
        )
        optimizer_steps = math.ceil(len(train_loader) / options.accumulation_steps)
        self.scheduler: LRScheduler = CosineAnnealingLR(
            self.optimizer, T_max=max(1, optimizer_steps * options.epochs)
        )
        self.amp_enabled, self.amp_dtype = select_amp(
            self.device, options.amp, options.amp_dtype
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16
        )
        self.writer = SummaryWriter(self.tensorboard_dir)
        self.start_epoch = 0
        self.global_step = 0

    def save_checkpoint(self, epoch: int, name: str | None = None) -> Path:
        target = self.checkpoint_dir / (name or f"epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "decoder": self.model.decoder.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "amp_dtype": str(self.amp_dtype),
            },
            target,
        )
        return target

    def resume(self, checkpoint: str | Path) -> None:
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.decoder.load_state_dict(state["decoder"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
        self.start_epoch = int(state["epoch"])
        self.global_step = int(state.get("global_step", 0))
        self.model.assert_backbone_frozen()

    def _append_log(self, row: dict[str, Any]) -> None:
        path = self.run_dir / "training_log.csv"
        exists = path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        sample_count = 0
        started = time.perf_counter()
        progress = tqdm(self.train_loader, desc=f"epoch {epoch}/{self.options.epochs}")
        remainder = len(self.train_loader) % self.options.accumulation_steps
        final_group_start = len(self.train_loader) - remainder
        for batch_index, batch in enumerate(progress):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                logits = self.model(images)
                raw_loss = self.loss_fn(logits, masks)
                divisor = (
                    remainder
                    if remainder and batch_index >= final_group_start
                    else self.options.accumulation_steps
                )
                loss = raw_loss / divisor
            if not torch.isfinite(raw_loss):
                raise FloatingPointError(f"non-finite loss at batch {batch_index}: {raw_loss}")
            self.scaler.scale(loss).backward()
            last_batch = batch_index + 1 == len(self.train_loader)
            should_step = (batch_index + 1) % self.options.accumulation_steps == 0 or last_batch
            if should_step:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.decoder.parameters(), self.options.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1
            count = images.shape[0]
            total_loss += raw_loss.detach().float().item() * count
            sample_count += count
            progress.set_postfix(loss=f"{raw_loss.item():.4f}")
        self.model.assert_backbone_frozen()
        metrics = {
            "epoch": float(epoch),
            "loss": total_loss / sample_count,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "wall_time_seconds": time.perf_counter() - started,
            "global_step": float(self.global_step),
        }
        self._append_log(metrics)
        self.writer.add_scalar("train/loss", metrics["loss"], epoch)
        self.writer.add_scalar("train/learning_rate", metrics["learning_rate"], epoch)
        return metrics

    def fit(self) -> list[dict[str, float]]:
        history = []
        try:
            for epoch in range(self.start_epoch + 1, self.options.epochs + 1):
                history.append(self.train_epoch(epoch))
                if epoch % self.options.checkpoint_every == 0:
                    self.save_checkpoint(epoch)
            self.save_checkpoint(self.options.epochs, "last.pt")
            self.model.assert_backbone_frozen()
            return history
        finally:
            self.writer.close()
