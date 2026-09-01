from __future__ import annotations

import random
from collections.abc import Sequence

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from cod_ssl.data.transforms import MEAN, STD


class ClipPairedTransform:
    """Apply one sampled geometry realization to an entire clip and target mask."""

    def __init__(self, training: bool, size: int = 384):
        self.training, self.size = training, size

    def __call__(self, frames: Sequence[Image.Image], mask: Image.Image):
        flip = self.training and random.random() < 0.5
        output = []
        for frame in frames:
            frame = TF.resize(frame.convert("RGB"), [self.size, self.size], InterpolationMode.BILINEAR)
            if flip:
                frame = TF.hflip(frame)
            output.append(TF.normalize(TF.to_tensor(frame), MEAN, STD))
        mask = TF.resize(mask.convert("L"), [self.size, self.size], InterpolationMode.NEAREST)
        if flip:
            mask = TF.hflip(mask)
        mask_tensor = (TF.pil_to_tensor(mask) > 0).float()
        if not torch.all((mask_tensor == 0) | (mask_tensor == 1)):
            raise RuntimeError("mask transform produced non-binary values")
        return torch.stack(output), mask_tensor
