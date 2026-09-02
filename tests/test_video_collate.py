from cod_ssl.data.collate import video_collate


def test_video_collate_preserves_uniform_optional_metadata():
    result = video_collate([
        {"fps": None, "metadata": {"class": None, "scope": "sequence"}},
        {"fps": None, "metadata": {"class": None, "scope": "sequence"}},
    ])
    assert result["fps"] is None
    assert result["metadata"]["class"] is None
    assert result["metadata"]["scope"] == ["sequence", "sequence"]
