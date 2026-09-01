import pytest

from cod_ssl.data.clip_sampler import ClipSampler, ClipSpec


def test_middle_even_clip_and_stride_are_exact():
    positions, valid = ClipSampler().source_indices([1, 3, 8, 10, 20, 30], 3, ClipSpec(4, 1, 2))
    assert positions == [1, 2, 3, 4]
    assert valid.tolist() == [True] * 4


def test_replication_marks_padding_at_both_boundaries():
    sampler = ClipSampler()
    left, left_valid = sampler.source_indices([10, 20, 40], 0, ClipSpec(5, 1, 2))
    right, right_valid = sampler.source_indices([10, 20, 40], 2, ClipSpec(5, 1, 2))
    assert left == [0, 0, 0, 1, 2] and left_valid.tolist() == [False, False, True, True, True]
    assert right == [0, 1, 2, 2, 2] and right_valid.tolist() == [True, True, True, False, False]


def test_sampler_rejects_non_chronological_sequence():
    with pytest.raises(ValueError, match="chronological"):
        ClipSampler().source_indices([1, 3, 3], 1, ClipSpec(1, 1, 0))
