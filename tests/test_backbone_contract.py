import torch
from torch import nn
from cod_ssl.backbones.base import FrozenBackbone


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

