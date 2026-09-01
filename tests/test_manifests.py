import pandas as pd
from cod_ssl.data.manifests import (
    cod10k_category,
    create_dev_split,
    create_stratified_smoke_manifest,
)


def test_dev_split_is_deterministic_and_stratified(tmp_path):
    frame = pd.DataFrame([{"id": f"{source}{i}", "source": source,
                           "image_path": "i", "mask_path": "m"}
                          for source in ("camo", "cod10k") for i in range(20)])
    source = tmp_path / "all.csv"; frame.to_csv(source, index=False)
    train, val = create_dev_split(source, tmp_path / "train.csv", tmp_path / "val.csv")
    assert len(train) == 36 and len(val) == 4
    assert val.groupby("source").size().to_dict() == {"camo": 2, "cod10k": 2}
    _, val_again = create_dev_split(source, tmp_path / "train2.csv", tmp_path / "val2.csv")
    assert val.id.tolist() == val_again.id.tolist()


def test_cod10k_category_parses_published_identifier():
    assert (
        cod10k_category("COD10K-CAM-1-Aquatic-13-Pipefish-624")
        == "Aquatic/Pipefish"
    )


def test_smoke_manifest_is_deterministic_source_and_category_stratified(tmp_path):
    rows = [
        {"id": f"CAMO-{index}", "source": "camo", "image_path": "i", "mask_path": "m"}
        for index in range(1000)
    ]
    for category in range(20):
        rows.extend(
            {
                "id": f"COD10K-CAM-1-Aquatic-{category + 1}-Class{category}-{index}",
                "source": "cod10k", "image_path": "i", "mask_path": "m",
            }
            for index in range(152)
        )
    source = tmp_path / "train_all.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    first, report = create_stratified_smoke_manifest(
        source, tmp_path / "smoke.csv", size=256, seed=42
    )
    second, _ = create_stratified_smoke_manifest(
        source, tmp_path / "smoke_again.csv", size=256, seed=42
    )
    assert first.id.tolist() == second.id.tolist()
    assert first.groupby("source").size().to_dict() == {"camo": 63, "cod10k": 193}
    cod10k_report = report[report.source == "cod10k"]
    assert len(cod10k_report) == 20
    assert cod10k_report.selected.min() >= 1
    assert not first.id.duplicated().any()
