import pytest
import torch
from torch import nn

from cod_ssl.temporal import (
    GatedMambaMixAdapter,
    MeanTemporalAdapter,
    VJEPATargetSelector,
)


class IdentityMamba(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, inputs):
        return inputs * self.scale


def test_vjepa_even_tubelet_mapping_is_explicit():
    selector = VJEPATargetSelector(2)
    assert selector.token_index(32, 64) == 16
    assert selector.source_interval(32, 64) == (32, 33)
    with pytest.raises(ValueError):
        selector.token_index(2, 63)


def test_mean_ignores_replicate_padding():
    adapter = MeanTemporalAdapter()
    features = torch.tensor([1.0, 2.0, 9.0]).reshape(1, 3, 1, 1, 1)
    output, _ = adapter(features, torch.tensor([[True, True, False]]), 1)
    assert output.item() == 1.5


def test_gmmix_dense_shape_backward_and_stateless_reset():
    adapter = GatedMambaMixAdapter(
        input_dim=8, depth=1, spatial_heads=2, dropout=0.0,
        mamba_factory=lambda **kwargs: IdentityMamba(**kwargs),
    )
    features = torch.randn(2, 3, 8, 2, 2, requires_grad=True)
    output, state = adapter(features, torch.ones(2, 3, dtype=torch.bool), 1)
    assert output.shape == (2, 8, 2, 2) and state is None
    output.square().mean().backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in adapter.parameters())
    assert adapter.reset_state(2, (2, 2)) is None
