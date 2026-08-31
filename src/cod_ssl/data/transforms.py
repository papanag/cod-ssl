from __future__ import annotations

import random

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class PairedTransform:
    def __init__(self, training: bool, size: int = 384):
        self.training, self.size = training, size

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, [self.size, self.size], InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.size, self.size], InterpolationMode.NEAREST)
        if self.training and random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        image_tensor = TF.normalize(TF.to_tensor(image), MEAN, STD)
        mask_tensor = (TF.pil_to_tensor(mask.convert("L")) > 0).float()
        return image_tensor, mask_tensor

