import torch
import pytest
from torch import nn
from cod_ssl.backbones.base import FrozenBackbone
from cod_ssl.backbones.vjepa21 import VJEPA21ViTB16


class MockBackbone(FrozenBackbone):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.ones(())); self.freeze()
    @property
    def feature_dims(self): return [8, 8, 8, 8]
    def forward_features(self, images):
        with torch.inference_mode(): outputs = [torch.zeros(images.shape[0], 8, 24, 24) for _ in range(4)]
        return self.validate_features(images, outputs)


def test_contract_and_freeze():
    model = MockBackbone(); outputs = model(torch.zeros(2, 3, 384, 384))
    assert len(outputs) == 4 and all(x.shape == (2, 8, 24, 24) for x in outputs)
    assert not any(p.requires_grad for p in model.parameters())


def test_layer_selection_requires_four_unique_increasing_transformer_indices():
    model = MockBackbone()
    model.configure_layers([1, 3, 5, 7])
    assert model.layer_indices == (1, 3, 5, 7)
    for invalid in ([1, 3, 5], [1, 3, 3, 7], [3, 1, 5, 7], [1, 3, 5, 12]):
        with pytest.raises(ValueError):
            model.configure_layers(invalid)


def test_all_twelve_layers_are_a_valid_extraction_contract():
    model = MockBackbone()
    model.configure_layers(range(12))
    assert model.layer_indices == tuple(range(12))


def test_vjepa_rejects_non_official_intermediate_outputs():
    with pytest.raises(ValueError, match="official hierarchical layers"):
        VJEPA21ViTB16(layers=list(range(12)))
