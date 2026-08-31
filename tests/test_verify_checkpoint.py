import torch


def test_training_checkpoint_is_decoder_only(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"epoch": 2, "global_step": 4, "decoder": {"weight": torch.ones(1)}}, checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    assert "backbone" not in payload
    assert payload["epoch"] == 2 and payload["global_step"] == 4
