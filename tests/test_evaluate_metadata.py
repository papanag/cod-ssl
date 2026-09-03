from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_single_frame_summary_uses_local_target_index_zero():
    module = _module()
    clip = {"length": 64, "target_index": 32, "source_frame_stride": 5}

    for system in ("DS", "VI"):
        summary = module._evaluation_clip_summary(clip, system)
        assert summary["n_observations"] == 1
        assert summary["target_index"] == 0
        assert summary["source_frame_span"] == 0


def test_temporal_summary_preserves_declared_geometry():
    module = _module()
    clip = {"length": 64, "target_index": 32, "source_frame_stride": 5}

    for system in ("DT", "VV"):
        summary = module._evaluation_clip_summary(clip, system)
        assert summary["n_observations"] == 64
        assert summary["target_index"] == 32
        assert summary["source_frame_span"] == 315
