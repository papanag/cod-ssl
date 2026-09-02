import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

METRICS = ("S", "weightedF", "MAE", "E_adapt", "E_mean", "E_max")


def _run(root: Path, dataset: str, system: str, *, seed: int = 42) -> None:
    directory = root / f"{dataset}_{system}"
    directory.mkdir(parents=True)
    offset = {"DS": 0.0, "VI": 0.1, "DT": 0.2, "VV": 0.3}[system]
    rows = []
    for index, video_id in enumerate(("a", "b")):
        row = {
            "source_video_id": video_id, "n_evaluated_frames": index + 1,
            "attr_OC": index == 0,
        }
        row.update({metric: 0.4 + offset for metric in METRICS})
        rows.append(row)
    pd.DataFrame(rows).to_csv(directory / "per_video.csv", index=False)
    frame_rows = []
    for index, video_id in enumerate(("a", "b")):
        frame_rows.append({
            "source_video_id": video_id, "frame_id": str(index), "frame_number": index,
            "source_frame_indices": json.dumps([index]),
            "attr_OC": index == 0,
            **{metric: 0.4 + offset for metric in METRICS},
        })
    pd.DataFrame(frame_rows).to_csv(directory / "per_frame.csv", index=False)
    values = {metric: 0.4 + offset for metric in METRICS}
    summary = {
        "schema_version": 3,
        "run": {"dataset": dataset, "regime": "D1" if dataset == "moca_mask_dense" else "S5", "system_id": system, "seed": seed},
        "dataset_release": {
            "released_unique_rgb": 4691 if dataset == "moca_mask_dense" else 30028,
            "manual_targets": 4691 if dataset == "moca_mask_dense" else 30028,
            "derived_dense_context_frames": 10000 if dataset == "moca_mask_dense" else None,
        },
        "metrics": {"minmax": {
            "frame_weighted_official_compatible": values,
            "video_weighted_study_primary": values,
            "by_attribute": {},
        }},
    }
    (directory / "summary.json").write_text(json.dumps(summary))


def test_summarizer_generates_camotion_attribute_tables(tmp_path):
    runs = tmp_path / "runs"
    for dataset in ("moca_mask_dense", "camotion"):
        for system in ("DS", "VI", "DT", "VV"):
            _run(runs, dataset, system)
    output = tmp_path / "report"
    root = Path(__file__).parents[1]
    subprocess.run([
        sys.executable, "scripts/summarize_results.py", "--runs-root", str(runs),
        "--matrix", "configs/experiments/vcod_primary_2x2.yaml", "--output", str(output),
        "--attributes", "OC",
    ], cwd=root, check=True)
    attributes = pd.read_csv(output / "camotion_attribute_comparisons.csv")
    frame_attributes = pd.read_csv(output / "camotion_attribute_frame_weighted.csv")
    assert set(attributes["subset"]) == {"All", "OC"}
    assert {"n_videos", "n_targets", "gain_advantage"} <= set(attributes)
    assert frame_attributes.aggregation.eq("frame_weighted_official_compatible").all()
    report = (output / "report.md").read_text()
    assert "overlapping sequence-level annotations" in report
    assert "MoCA-Mask dense derived" in report


def test_summarizer_rejects_a_missing_primary_system(tmp_path):
    runs = tmp_path / "runs"
    for dataset in ("moca_mask_dense", "camotion"):
        for system in ("DS", "VI", "DT", "VV"):
            if (dataset, system) != ("camotion", "VV"):
                _run(runs, dataset, system)
    root = Path(__file__).parents[1]
    result = subprocess.run([
        sys.executable, "scripts/summarize_results.py", "--runs-root", str(runs),
        "--matrix", "configs/experiments/vcod_primary_2x2.yaml",
        "--output", str(tmp_path / "report"), "--attributes", "OC",
    ], cwd=root, capture_output=True, text=True)
    assert result.returncode != 0
    assert "missing primary cells" in result.stderr
