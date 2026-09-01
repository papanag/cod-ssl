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
        "05_vcod_setup_validation.ipynb",
        "06_vcod_run_cell.ipynb",
        "07_vcod_summarize.ipynb",
    ):
        notebook = json.loads((root / "notebooks" / name).read_text())
        # A Colab form containing constants may precede the launcher; it does
        # not touch runtime state and lets the bootstrap choose optional extras.
        launcher = "".join(
            line
            for cell in notebook["cells"][1:3]
            if cell["cell_type"] == "code"
            for line in cell["source"]
        )
        assert "drive.mount('/content/drive')" in launcher
        assert "scripts/bootstrap_colab.py" in launcher
        assert "os.environ.update(state['environment'])" in launcher
    assert not (root / "notebooks" / "00_colab_runtime_check.ipynb").exists()


def test_vcod_notebooks_are_thin_valid_colab_orchestrators():
    root = Path(__file__).parents[1]
    names = (
        "05_vcod_setup_validation.ipynb",
        "06_vcod_run_cell.ipynb",
        "07_vcod_summarize.ipynb",
    )
    sources = {}
    for name in names:
        notebook = json.loads((root / "notebooks" / name).read_text())
        assert notebook["nbformat"] == 4
        assert notebook["metadata"]["accelerator"] == "GPU"
        assert notebook["metadata"]["colab"]["gpuType"] == "A100"
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                compile(source, f"{name}:cell-{index}", "exec")
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
        sources[name] = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
    assert "scripts/inspect_dataset.py" in sources[names[0]]
    assert "vcod_validation_approval.json" in sources[names[0]]
    assert "[dev,notebooks,vcod]" not in sources[names[0]]
    assert "scripts/train_probe.py" in sources[names[1]]
    assert "SYSTEM == 'DT'" in sources[names[1]]
    assert "--resume" in sources[names[1]]
    assert "RUN_KIND == 'tuning'" in sources[names[1]]
    assert "scripts/evaluate.py" in sources[names[1]]
    assert "scripts/summarize_results.py" in sources[names[2]]
    assert "RUNS_ROOT = VCOD_ROOT / 'runs'" in sources[names[2]]
    assert "VCOD_ROOT / 'tuning'" not in sources[names[2]]
