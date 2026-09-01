import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from cod_ssl.data.exclusions import exclude_manifest_rows, load_exclusion_policy


def _validator_module():
    path = Path(__file__).parents[1] / "scripts" / "validate_dataset.py"
    spec = importlib.util.spec_from_file_location("validate_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_known_overlap_policy_is_initialized_with_seven_split_aware_pairs():
    policy = Path(__file__).parents[1] / "configs" / "dataset_exclusions.csv"
    exclusions = load_exclusion_policy(policy)
    shared = {
        "COD10K-CAM-1-Aquatic-3-Crab-32",
        "COD10K-CAM-2-Terrestrial-23-Cat-1506",
    }
    assert exclusions["train_all"] == shared | {
        "COD10K-CAM-2-Terrestrial-26-Chameleon-1694",
        "COD10K-CAM-2-Terrestrial-28-Deer-1796",
        "COD10K-CAM-3-Flying-53-Bird-3205",
        "COD10K-CAM-2-Terrestrial-32-Giraffe-1930",
        "COD10K-CAM-2-Terrestrial-31-Gecko-1928",
    }
    assert exclusions["cod10k_test"] == shared | {
        "COD10K-CAM-2-Terrestrial-31-Gecko-1892",
        "COD10K-CAM-2-Terrestrial-28-Deer-1762",
        "COD10K-CAM-3-Flying-65-Owl-4633",
        "COD10K-CAM-2-Terrestrial-32-Giraffe-1932",
        "COD10K-CAM-2-Terrestrial-31-Gecko-1895",
    }


def test_same_known_ids_are_removed_from_training_and_test_manifests():
    exclusions = {"duplicate-a", "duplicate-b"}
    frame = pd.DataFrame({"id": ["keep", "duplicate-a.png", "duplicate-b.jpg"]})
    for dataset in ("train_all", "cod10k_test"):
        clean, removed = exclude_manifest_rows(
            frame, exclusions, dataset_name=dataset, require_all=True
        )
        assert clean.id.tolist() == ["keep"]
        assert removed == ["duplicate-a", "duplicate-b"]


def test_required_exclusion_fails_if_one_side_does_not_contain_both_ids():
    with pytest.raises(ValueError, match="missing configured overlap exclusions"):
        exclude_manifest_rows(
            pd.DataFrame({"id": ["duplicate-a"]}),
            {"duplicate-a", "duplicate-b"},
            dataset_name="train_all",
            require_all=True,
        )


def test_validation_receipt_requires_exact_version_hash_counts_and_settings():
    validator = _validator_module()
    hashes = {"train_all": "abc"}
    counts = {"train_all": 4033}
    settings = {"exclusions_sha256": "policy"}
    receipt = {
        "validation_passed": True,
        "validator_version": validator.VALIDATOR_VERSION,
        "manifest_hashes": hashes,
        "dataset_counts": counts,
        "validation_settings": settings,
        "completion_timestamp_utc": "2026-09-01T00:00:00+00:00",
    }
    assert validator.receipt_matches(receipt, hashes, counts, settings)
    changed = json.loads(json.dumps(receipt))
    changed["validation_settings"]["exclusions_sha256"] = "changed"
    assert not validator.receipt_matches(changed, hashes, counts, settings)
