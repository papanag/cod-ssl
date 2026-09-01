import torch
from torch import nn

from cod_ssl.backbones.base import DenseFeatureBatch
from cod_ssl.models.video_cod_model import VideoCODModel
from cod_ssl.temporal import TargetFrameAdapter, VJEPATargetSelector


class MockDenseBackbone(nn.Module):
    def __init__(self, tubelets=False):
        super().__init__(); self.weight = nn.Parameter(torch.ones(())); self.tubelets = tubelets; self.freeze()

    def freeze(self):
        for parameter in self.parameters(): parameter.requires_grad_(False)
        self.eval()

    def encode_image(self, image):
        features = image.mean(1, keepdim=True).repeat(1, 4, 1, 1)
        return DenseFeatureBatch(features[:, None], torch.ones(image.shape[0], 1, dtype=torch.bool),
                                 tuple(image.shape[-2:]), ((0, 0),), {"pathway": "image"})

    def encode_video(self, frames, valid):
        features = frames.mean(2, keepdim=True).repeat(1, 1, 4, 1, 1)
        if self.tubelets:
            features = features.reshape(frames.shape[0], frames.shape[1] // 2, 2, 4, *frames.shape[-2:]).mean(2)
            valid = valid.reshape(frames.shape[0], -1, 2).any(2)
            intervals = tuple((2*i, 2*i+1) for i in range(features.shape[1]))
        else: intervals = tuple((i, i) for i in range(features.shape[1]))
        return DenseFeatureBatch(features, valid, tuple(frames.shape[-2:]), intervals, {})


def _batch(time=1, target=0):
    return {"frames": torch.randn(2, time, 3, 8, 8), "target_mask": torch.zeros(2, 1, 8, 8),
            "target_index": torch.tensor([target, target]), "valid_temporal_mask": torch.ones(2, time, dtype=torch.bool)}


def test_image_model_outputs_target_mask_shape_and_freezes_backbone():
    model = VideoCODModel(MockDenseBackbone(), TargetFrameAdapter(), pathway="image", feature_dim=4,
                          projected_channels=8, hidden_channels=8)
    output = model(_batch())["logits"]
    assert output.shape == (2, 1, 8, 8)
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())


def test_native_video_maps_source_target_to_tubelet():
    model = VideoCODModel(MockDenseBackbone(tubelets=True), VJEPATargetSelector(2),
                          pathway="native_video", feature_dim=4, projected_channels=8, hidden_channels=8)
    assert model(_batch(4, 2))["logits"].shape == (2, 1, 8, 8)
