import pytest

from cod_ssl.utils.config import load_config
from cod_ssl.utils.vcod_config import configure_system, validate_vcod_config


@pytest.mark.parametrize("system", ["DS", "VI", "DT", "VV"])
def test_primary_system_constraints(system):
    config = configure_system(load_config("configs/experiments/vcod_primary_2x2.yaml"), system)
    validate_vcod_config(config)


def test_diagnostic_cannot_be_labeled_primary():
    config = configure_system(load_config("configs/experiments/vcod_primary_2x2.yaml"), "DM")
    with pytest.raises(ValueError, match="diagnostics"):
        validate_vcod_config(config)
