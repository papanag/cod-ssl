import pytest
import torch

from cod_ssl.losses import BCESoftIoULoss


def test_loss_is_finite_and_differentiable():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    mask = torch.randint(0, 2, logits.shape).float()
    loss = BCESoftIoULoss()(logits, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_loss_is_near_zero_for_near_perfect_logits():
    mask = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    logits = torch.where(mask.bool(), torch.tensor(30.0), torch.tensor(-30.0))
    assert BCESoftIoULoss()(logits, mask).item() < 1e-6


def test_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shapes differ"):
        BCESoftIoULoss()(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 4, 4))
