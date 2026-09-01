from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from cod_ssl.backbones.base import FrozenBackbone
from cod_ssl.engine.train import Trainer, TrainingOptions, build_decoder_optimizer, select_amp
from cod_ssl.models import FrozenCODModel


class TinyBackbone(FrozenBackbone):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.ones(()))
        self.freeze()

    @property
    def feature_dims(self):
        return [4] * 4

    def forward_features(self, images):
        with torch.inference_mode():
            result = [torch.zeros(images.shape[0], 4, 24, 24) for _ in range(4)]
        return self.validate_features(images, result)


class AllLayerTinyBackbone(TinyBackbone):
    layer_indices = tuple(range(12))

    @property
    def feature_dims(self):
        return [4] * 12

    def forward_features(self, images):
        with torch.inference_mode():
            result = [torch.full((images.shape[0], 4, 24, 24), float(i)) for i in range(12)]
        return self.validate_features(images, result)


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Conv2d(4, 1, 1)

    def forward(self, features):
        return F.interpolate(
            self.classifier(features[0]), size=(384, 384), mode="bilinear", align_corners=False
        )


class OneSampleDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "image": torch.zeros(3, 384, 384),
            "mask": torch.zeros(1, 384, 384),
        }


def test_optimizer_contains_decoder_only():
    model = FrozenCODModel(TinyBackbone())
    optimizer = build_decoder_optimizer(model, learning_rate=1e-3, weight_decay=1e-4)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in model.decoder.parameters()}
    assert not optimized & {id(p) for p in model.backbone.parameters()}


def test_amp_is_disabled_on_cpu():
    assert select_amp(torch.device("cpu"), True, "auto") == (False, torch.float32)


def test_optimizer_includes_mixer_but_never_backbone():
    model = FrozenCODModel(
        AllLayerTinyBackbone(), TinyDecoder(), learned_layer_mixtures=4
    )
    optimizer = build_decoder_optimizer(model, learning_rate=1e-3, weight_decay=1e-4)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in model.readout_parameters()}
    assert id(model.layer_mixer.logits) in optimized
    assert not optimized & {id(p) for p in model.backbone.parameters()}


def test_trainer_writes_logs_checkpoint_and_resumes(tmp_path):
    options = TrainingOptions(epochs=1, accumulation_steps=1, amp=False)
    model = FrozenCODModel(TinyBackbone(), TinyDecoder())
    trainer = Trainer(
        model,
        DataLoader(OneSampleDataset(), batch_size=1),
        tmp_path,
        options,
        device=torch.device("cpu"),
    )
    initial = model.decoder.classifier.weight.detach().clone()
    history = trainer.fit()
    assert len(history) == 1 and torch.isfinite(torch.tensor(history[0]["loss"]))
    assert not torch.equal(initial, model.decoder.classifier.weight)
    assert (tmp_path / "training_log.csv").is_file()
    checkpoint = tmp_path / "checkpoints" / "last.pt"
    assert checkpoint.is_file()

    restored = FrozenCODModel(TinyBackbone(), TinyDecoder())
    resumed = Trainer(
        restored,
        DataLoader(OneSampleDataset(), batch_size=1),
        tmp_path / "resumed",
        options,
        device=torch.device("cpu"),
    )
    resumed.resume(checkpoint)
    resumed.writer.close()
    assert resumed.start_epoch == 1 and resumed.global_step == 1
    assert torch.equal(restored.decoder.classifier.weight, model.decoder.classifier.weight)
    restored.assert_backbone_frozen()
