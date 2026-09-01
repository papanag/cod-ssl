from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BCESoftIoULoss(nn.Module):
    """Unweighted BCE-with-logits plus smoothed soft IoU loss."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if logits.shape != mask.shape:
            raise ValueError(f"logits and mask shapes differ: {logits.shape} vs {mask.shape}")
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"expected [B,1,H,W] tensors, got {tuple(logits.shape)}")
        if not torch.is_floating_point(mask):
            mask = mask.float()

        bce = F.binary_cross_entropy_with_logits(logits, mask)
        probability = logits.sigmoid()
        intersection = (probability * mask).sum(dim=(2, 3))
        union = (probability + mask - probability * mask).sum(dim=(2, 3))
        iou_loss = 1.0 - ((intersection + self.smooth) / (union + self.smooth)).mean()
        return bce + iou_loss


class BCEDiceLoss(nn.Module):
    """VCOD protocol loss: unweighted BCE-with-logits plus soft Dice loss."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if logits.shape != mask.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"expected matching [B,1,H,W] tensors, got {logits.shape}, {mask.shape}")
        mask = mask.float()
        bce = F.binary_cross_entropy_with_logits(logits, mask)
        probability = logits.sigmoid()
        intersection = (probability * mask).sum(dim=(2, 3))
        denominator = (probability + mask).sum(dim=(2, 3))
        dice_loss = 1.0 - ((2 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        return bce + dice_loss
