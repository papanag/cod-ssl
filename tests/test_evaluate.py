import numpy as np
import pytest
import torch

from cod_ssl.engine.evaluate import logits_to_prediction


def test_postprocessing_restores_size_and_minmax_normalizes():
    logits = torch.linspace(-3, 3, 16).reshape(1, 1, 4, 4)
    prediction = logits_to_prediction(logits, (7, 9), minmax_normalize=True)
    assert prediction.shape == (7, 9) and prediction.dtype == np.uint8
    assert prediction.min() == 0 and prediction.max() == 255


def test_postprocessing_rejects_batches():
    with pytest.raises(ValueError, match="expected one"):
        logits_to_prediction(torch.zeros(2, 1, 4, 4), (4, 4))
