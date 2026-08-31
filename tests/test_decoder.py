import pytest
import torch

from cod_ssl.models.decoder import CommonCODDecoder


def test_decoder_shape_and_gradients():
    decoder = CommonCODDecoder([16, 24, 32, 40])
    features = [torch.randn(2, channels, 24, 24) for channels in (16, 24, 32, 40)]
    logits = decoder(features)
    assert logits.shape == (2, 1, 384, 384)
    logits.mean().backward()
    assert all(parameter.grad is not None for parameter in decoder.parameters())


def test_decoder_rejects_invalid_contract():
    decoder = CommonCODDecoder([8] * 4)
    with pytest.raises(ValueError, match="exactly four"):
        decoder([torch.zeros(1, 8, 24, 24)] * 3)
    with pytest.raises(ValueError, match="feature 0"):
        decoder([torch.zeros(1, 8, 12, 12)] + [torch.zeros(1, 8, 24, 24)] * 3)

