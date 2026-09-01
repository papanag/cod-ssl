import random

import numpy as np
import torch
from PIL import Image

from cod_ssl.data.video_transforms import ClipPairedTransform


def test_clip_geometry_is_shared_and_mask_remains_binary():
    pixels = np.zeros((12, 12, 3), dtype=np.uint8)
    pixels[:, :4] = 255
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[:, :4] = 255
    random.seed(1)  # first draw triggers flip
    frames, transformed_mask = ClipPairedTransform(True, 16)(
        [Image.fromarray(pixels), Image.fromarray(pixels)], Image.fromarray(mask)
    )
    assert torch.equal(frames[0], frames[1])
    assert set(transformed_mask.unique().tolist()) == {0.0, 1.0}
    assert transformed_mask[:, :, 10:].sum() > transformed_mask[:, :, :6].sum()
