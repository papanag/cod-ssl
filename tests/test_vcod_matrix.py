import json
import subprocess
import sys
from pathlib import Path


def test_primary_dry_run_is_moca_plus_camotion_eight_cells_per_seed():
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable, "scripts/run_primary_matrix.py",
            "--config", "configs/experiments/vcod_primary_2x2.yaml",
            "--datasets", "moca_mask_dense", "camotion",
            "--systems", "DS", "VI", "DT", "VV",
            "--seeds", "42", "--dry-run",
        ],
        cwd=root, check=True, capture_output=True, text=True,
    )
    runs = json.loads(result.stdout)
    assert len(runs) == 8
    assert {(run["dataset"], run["regime"]) for run in runs} == {
        ("moca_mask_dense", "D1"), ("camotion", "S5")
    }
    assert {run["system"] for run in runs} == {"DS", "VI", "DT", "VV"}
