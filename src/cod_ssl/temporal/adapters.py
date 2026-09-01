from __future__ import annotations

from cod_ssl.temporal.base import TemporalAdapter, validate_temporal_inputs


class TargetFrameAdapter(TemporalAdapter):
    def forward(self, features, temporal_valid, target_index, *, state=None):
        validate_temporal_inputs(features, temporal_valid)
        if not 0 <= target_index < features.shape[1]:
            raise IndexError("target temporal feature is outside the encoded sequence")
        if not temporal_valid[:, target_index].all():
            raise ValueError("target temporal feature is padding")
        return features[:, target_index], None

    def reset_state(self, batch_size: int, spatial_size: tuple[int, int]):
        return None


class MeanTemporalAdapter(TemporalAdapter):
    def forward(self, features, temporal_valid, target_index, *, state=None):
        validate_temporal_inputs(features, temporal_valid)
        weights = temporal_valid[:, :, None, None, None].to(features.dtype)
        return (features * weights).sum(1) / weights.sum(1).clamp_min(1), None

    def reset_state(self, batch_size: int, spatial_size: tuple[int, int]):
        return None


class VJEPATargetSelector(TemporalAdapter):
    """Select the tubelet containing the target source-frame index."""

    def __init__(self, tubelet_size: int = 2):
        super().__init__()
        self.tubelet_size = int(tubelet_size)

    def token_index(self, target_index: int, clip_length: int) -> int:
        if clip_length % self.tubelet_size:
            raise ValueError("clip length is not divisible by V-JEPA tubelet size")
        if not 0 <= target_index < clip_length:
            raise IndexError("target_index is outside source clip")
        return target_index // self.tubelet_size

    def source_interval(self, target_index: int, clip_length: int) -> tuple[int, int]:
        token = self.token_index(target_index, clip_length)
        start = token * self.tubelet_size
        return start, start + self.tubelet_size - 1

    def forward(self, features, temporal_valid, target_index, *, state=None):
        validate_temporal_inputs(features, temporal_valid)
        # Here target_index is an encoded tubelet index; source-to-token conversion
        # must be explicit in model assembly using token_index().
        if not 0 <= target_index < features.shape[1]:
            raise IndexError("target tubelet is outside encoded sequence")
        if not temporal_valid[:, target_index].all():
            raise ValueError("target tubelet contains boundary padding")
        return features[:, target_index], None

    def reset_state(self, batch_size: int, spatial_size: tuple[int, int]):
        return None
