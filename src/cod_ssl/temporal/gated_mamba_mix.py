"""Dense GMMix adapter derived from the official reference implementation.

Copyright (c) 2026 Mobile Perception Systems Lab at TU/e, MIT License.
Adapted from commit 59249bf83311bc34bae277e2e8adec287ffe5d0f; see
docs/gmmix_provenance.md. The optional ``mamba-ssm`` package supplies Mamba.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from cod_ssl.temporal.base import TemporalAdapter, validate_temporal_inputs


class _MLP(nn.Sequential):
    def __init__(self, dim: int, ratio: int, dropout: float):
        super().__init__(nn.Linear(dim, dim * ratio), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(dim * ratio, dim), nn.Dropout(dropout))


class _SpatialBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ratio: int, dropout: float):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mlp = _MLP(dim, ratio, dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(inputs)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        inputs = inputs + attended
        return inputs + self.mlp(self.norm2(inputs))


class _MambaResidual(nn.Module):
    def __init__(self, dim: int, d_state: int, expand: int, dropout: float,
                 layer_index: int, factory: Callable[..., nn.Module]):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.core = factory(d_model=dim, d_state=d_state, expand=expand, layer_idx=layer_index)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.dropout(self.core(self.norm(inputs)))


def _official_mamba_factory(**kwargs):
    try:
        from mamba_ssm import Mamba
    except ImportError as error:
        raise RuntimeError(
            "GatedMambaMix requires the optional mamba-ssm dependency; "
            "install it on the CUDA training environment"
        ) from error
    return Mamba(**kwargs)


class GatedMambaMixAdapter(TemporalAdapter):
    """Reference GMMix over B,T,C,H,W, returning the requested dense frame.

    Boundary replication is deterministic. In accordance with the reference core,
    ``temporal_valid`` is validated but is not injected into the Mamba recurrence.
    """

    def __init__(
        self, input_dim: int = 768, depth: int = 1, d_state: int = 16,
        expand: int = 2, dropout: float = 0.1, spatial_heads: int = 12,
        spatial_mlp_ratio: int = 4, gate_hidden: int = 0,
        *, mamba_factory: Callable[..., nn.Module] | None = None,
    ):
        super().__init__()
        if depth < 1 or input_dim % spatial_heads:
            raise ValueError("depth must be positive and channels divisible by spatial_heads")
        factory = mamba_factory or _official_mamba_factory
        self.input_dim, self.depth = input_dim, depth
        self.spatial_blocks = nn.ModuleList(
            _SpatialBlock(input_dim, spatial_heads, spatial_mlp_ratio, dropout)
            for _ in range(depth)
        )
        self.mamba_blocks = nn.ModuleList(
            _MambaResidual(input_dim, d_state, expand, dropout, index, factory)
            for index in range(depth)
        )
        self.gates = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * input_dim, gate_hidden), nn.GELU(),
                          nn.Linear(gate_hidden, input_dim))
            if gate_hidden > 0 else nn.Linear(2 * input_dim, input_dim)
            for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(input_dim)

    def forward(self, features, temporal_valid, target_index, *, state=None):
        validate_temporal_inputs(features, temporal_valid)
        batch, time, channels, height, width = features.shape
        if channels != self.input_dim or not 0 <= target_index < time:
            raise ValueError("GMMix channel count or target index is invalid")
        if state is not None:
            raise ValueError("primary window mode does not accept recurrent state")
        tokens = features.permute(0, 1, 3, 4, 2).reshape(batch, time, height * width, channels)
        for spatial, mamba, gate in zip(self.spatial_blocks, self.mamba_blocks, self.gates):
            spatial_tokens = spatial(tokens.reshape(batch * time, height * width, channels))
            spatial_tokens = spatial_tokens.reshape(batch, time, height * width, channels)
            before = spatial_tokens.permute(0, 2, 1, 3).reshape(batch * height * width, time, channels)
            after = mamba(before)
            mixing = torch.sigmoid(gate(torch.cat([before, after], dim=-1)))
            mixed = (1.0 - mixing) * before + mixing * after
            tokens = mixed.reshape(batch, height * width, time, channels).permute(0, 2, 1, 3)
        tokens = self.output_norm(tokens)
        target = tokens[:, target_index].reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return target, None

    def reset_state(self, batch_size: int, spatial_size: tuple[int, int]):
        # Window-mode samples are stateless by construction. A future streaming
        # benchmark must use the reference InferenceParams path separately.
        return None
