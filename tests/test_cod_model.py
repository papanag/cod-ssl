import torch
from torch import nn

from cod_ssl.backbones.base import FrozenBackbone
from cod_ssl.models import FrozenCODModel


class MockFrozenBackbone(FrozenBackbone):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.freeze()

    @property
    def feature_dims(self):
        return [8] * 4

    def forward_features(self, images):
        with torch.inference_mode():
            outputs = [
                torch.ones(images.shape[0], 8, 24, 24, device=images.device) * self.scale
                for _ in range(4)
            ]
        return self.validate_features(images, outputs)


def test_model_keeps_backbone_frozen_during_decoder_backward():
    model = FrozenCODModel(MockFrozenBackbone()).train()
    logits = model(torch.zeros(1, 3, 384, 384))
    logits.mean().backward()
    model.assert_backbone_frozen()
    assert model.backbone.training is False
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())
