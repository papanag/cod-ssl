import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "bootstrap_colab.py"
    spec = importlib.util.spec_from_file_location("bootstrap_colab", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_clone_or_update_clones_missing_repository(tmp_path, monkeypatch):
    module = _module()
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, check: calls.append(command))
    target = tmp_path / "upstream"
    module.clone_or_update("https://example.test/repo.git", target)
    assert calls == [["git", "clone", "https://example.test/repo.git", str(target)]]
    assert target.parent.is_dir()


def test_clone_or_update_fast_forwards_existing_repository(tmp_path, monkeypatch):
    module = _module()
    target = tmp_path / "upstream"
    (target / ".git").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, check: calls.append(command))
    module.clone_or_update("https://example.test/repo.git", target)
    assert calls == [["git", "-C", str(target), "pull", "--ff-only"]]


def test_every_operational_notebook_starts_with_fresh_kernel_launcher():
    root = Path(__file__).parents[1]
    for name in (
        "01_backbone_feature_smoke_test.ipynb",
        "02_frozen_baseline_smoke_train.ipynb",
        "03_full_frozen_comparison.ipynb",
        "04_all_layer_mixture.ipynb",
    ):
        notebook = json.loads((root / "notebooks" / name).read_text())
        launcher = "".join(notebook["cells"][1]["source"])
        assert "drive.mount('/content/drive')" in launcher
        assert "scripts/bootstrap_colab.py" in launcher
        assert "os.environ.update(state['environment'])" in launcher
    assert not (root / "notebooks" / "00_colab_runtime_check.ipynb").exists()
