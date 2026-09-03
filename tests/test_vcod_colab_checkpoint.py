import importlib.util
from pathlib import Path

import torch
from torch import nn


def _module():
    path = Path(__file__).parents[1] / "scripts" / "train_probe.py"
    spec = importlib.util.spec_from_file_location("train_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.readout = nn.Linear(2, 1)


def test_atomic_checkpoint_contains_only_readout_and_resume_identity(tmp_path):
    module = _module()
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.readout.parameters())
    target = tmp_path / "last.pt"
    module._save_checkpoint(
        target, model, optimizer, 250, "config-hash", "resume-compatible-hash"
    )
    assert target.is_file()
    assert not target.with_suffix(".pt.part").exists()
    state = torch.load(target, map_location="cpu", weights_only=False)
    assert state["global_step"] == 250
    assert state["config_sha256"] == "config-hash"
    assert state["resume_compatibility_sha256"] == "resume-compatible-hash"
    assert state["readout"]
    assert all(not key.startswith("backbone.") for key in state["readout"])


def test_resume_identity_allows_only_step_target_to_change():
    module = _module()
    base = {
        "experiment": {"seed": 42},
        "training": {"learning_rate": 3e-4, "max_steps": 250},
    }
    extended = {
        "experiment": {"seed": 42},
        "training": {"learning_rate": 3e-4, "max_steps": 1000},
    }
    changed_lr = {
        "experiment": {"seed": 42},
        "training": {"learning_rate": 1e-4, "max_steps": 1000},
    }
    assert module._resume_compatibility_sha256(base) == module._resume_compatibility_sha256(extended)
    assert module._resume_compatibility_sha256(base) != module._resume_compatibility_sha256(changed_lr)


def test_per_run_entrypoints_do_not_rehash_approved_moca_assets():
    root = Path(__file__).parents[1]
    for name in ("train_probe.py", "evaluate.py"):
        source = (root / "scripts" / name).read_text()
        assert "verify_moca_mask_dense(" in source
        assert "verify_linked_targets=False" in source
