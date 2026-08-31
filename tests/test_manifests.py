import pandas as pd
from cod_ssl.data.manifests import create_dev_split


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
