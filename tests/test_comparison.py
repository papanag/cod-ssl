import json
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from cod_ssl.evaluation import compare_runs
from cod_ssl.evaluation.comparison import DATASET_ORDER


def _fake_run(root: Path, name: str, manifests: dict[str, Path], value: int) -> Path:
    root.mkdir()
    config = {
        "model": {"backbone": {"name": name}},
        "evaluation": {"manifests": {key: str(path) for key, path in manifests.items()}},
    }
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    metrics = {}
    for dataset in DATASET_ORDER:
        prediction_dir = root / "predictions" / dataset
        prediction_dir.mkdir(parents=True)
        Image.new("L", (4, 4), value).save(prediction_dir / "sample.png")
        metrics[dataset] = {
            "s_measure": 0.7, "e_adaptive": 0.8, "weighted_f": 0.6,
            "mae": 0.2, "inference_ms_per_image": 3.0,
        }
    (root / "metrics.json").write_text(json.dumps(metrics))
    pd.DataFrame({"epoch": [1, 2], "loss": [1.0, 0.8]}).to_csv(
        root / "training_log.csv", index=False
    )
    (root / "run_summary.json").write_text(json.dumps({"decoder_parameters": 10}))
    return root


def test_compare_runs_writes_tables_plots_and_overlays(tmp_path):
    image = tmp_path / "image.jpg"; mask = tmp_path / "mask.png"
    Image.new("RGB", (4, 4), "gray").save(image)
    Image.new("L", (4, 4), 255).save(mask)
    manifests = {}
    for dataset in DATASET_ORDER:
        manifest = tmp_path / f"{dataset}.csv"
        pd.DataFrame([{"id": "sample", "source": dataset,
                       "image_path": image, "mask_path": mask}]).to_csv(manifest, index=False)
        manifests[dataset] = manifest
    dino = _fake_run(tmp_path / "dino", "dinov3", manifests, 255)
    vjepa = _fake_run(tmp_path / "vjepa", "vjepa21", manifests, 0)
    output = compare_runs(dino, vjepa, tmp_path / "comparison", qualitative_count=12)
    assert (output / "comparison_metrics.csv").is_file()
    assert (output / "compute_comparison.csv").is_file()
    assert (output / "metric_comparison.png").is_file()
    assert len(list((output / "qualitative_panels").glob("*.png"))) == 4
