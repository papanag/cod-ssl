from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cod_ssl.metrics import CODMetrics
from cod_ssl.models import FrozenCODModel


def logits_to_prediction(
    logits: torch.Tensor,
    output_size: tuple[int, int],
    *,
    minmax_normalize: bool = True,
) -> np.ndarray:
    """Restore logits to GT size, sigmoid, normalize, then quantize to evaluator PNG."""
    if logits.ndim != 4 or logits.shape != (1, 1, *logits.shape[-2:]):
        raise ValueError(f"expected one [1,1,H,W] logit map, got {tuple(logits.shape)}")
    restored = F.interpolate(logits.float(), size=output_size, mode="bilinear", align_corners=False)
    prediction = restored.sigmoid()[0, 0]
    if minmax_normalize:
        minimum, maximum = prediction.amin(), prediction.amax()
        prediction = (prediction - minimum) / (maximum - minimum + 1e-8)
    return (
        prediction.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    )


class Evaluator:
    def __init__(
        self,
        model: FrozenCODModel,
        loader: DataLoader,
        prediction_dir: str | Path,
        *,
        device: torch.device | None = None,
        minmax_normalize: bool = True,
        save_predictions: bool = True,
        amp_enabled: bool = False,
        amp_dtype: torch.dtype = torch.float32,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.model.assert_backbone_frozen()
        self.loader = loader
        self.prediction_dir = Path(prediction_dir)
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        self.minmax_normalize = minmax_normalize
        self.save_predictions = save_predictions
        self.amp_enabled = amp_enabled and self.device.type == "cuda"
        self.amp_dtype = amp_dtype

    def evaluate(self) -> dict[str, Any]:
        metrics = CODMetrics()
        forward_seconds = 0.0
        seen_ids: set[str] = set()
        for batch in tqdm(self.loader, desc=self.prediction_dir.name):
            images = batch["image"].to(self.device, non_blocking=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                logits = self.model(images)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - started

            for index, (sample_id, mask_path) in enumerate(zip(batch["id"], batch["mask_path"])):
                safe_id = Path(str(sample_id)).name
                if safe_id in seen_ids:
                    raise ValueError(f"duplicate sample id would overwrite prediction: {safe_id}")
                seen_ids.add(safe_id)
                with Image.open(mask_path) as raw_mask:
                    ground_truth = np.asarray(raw_mask.convert("L"), dtype=np.uint8)
                prediction = logits_to_prediction(
                    logits[index : index + 1],
                    ground_truth.shape,
                    minmax_normalize=self.minmax_normalize,
                )
                if self.save_predictions:
                    Image.fromarray(prediction).save(
                        self.prediction_dir / f"{safe_id}.png"
                    )
                metrics.step(prediction, ground_truth)
        results = metrics.compute()
        results["inference_ms_per_image"] = 1000.0 * forward_seconds / results["num_images"]
        results["minmax_normalized"] = self.minmax_normalize
        return results
