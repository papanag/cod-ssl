from cod_ssl.utils.config import load_config


def test_config_inheritance():
    config = load_config("configs/frozen_dinov3_vitb16.yaml")
    assert config["experiment"]["input_size"] == 384
    assert config["model"]["backbone"]["name"] == "dinov3_vitb16"
    assert config["model"]["backbone"]["layers"] == [2, 5, 8, 11]

