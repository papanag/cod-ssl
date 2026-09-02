import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from cod_ssl.data.clip_sampler import ClipSpec
from cod_ssl.data.preprocessing.prepare_moca_mask_dense import build_moca_mask_dense, verify_moca_mask_dense
from cod_ssl.data.video_manifest import ManifestVideoCODDataset


def _bootstrap_module():
    path = Path(__file__).parents[1] / "scripts" / "bootstrap_vcod_data.py"
    spec = importlib.util.spec_from_file_location("bootstrap_vcod_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_download_resumes_undersized_archive_at_final_path(tmp_path, monkeypatch):
    module = _bootstrap_module()
    destination = tmp_path / "CAMotion.zip"
    destination.write_bytes(b"partial")

    def resume_download(*, id, output, quiet, resume):
        partial = Path(output)
        assert id == "file-id" and resume is True
        assert partial.read_bytes() == b"partial"
        partial.write_bytes(b"partial-complete")
        return output

    monkeypatch.setattr(module.gdown, "download", resume_download)
    module._download("file-id", destination, expected_bytes=len(b"partial-complete"))
    assert destination.read_bytes() == b"partial-complete"
    assert not destination.with_suffix(".zip.part").exists()


def test_download_keeps_incomplete_part_for_next_resume(tmp_path, monkeypatch):
    module = _bootstrap_module()
    destination = tmp_path / "CAMotion.zip"
    destination.write_bytes(b"partial")
    monkeypatch.setattr(
        module.gdown, "download",
        lambda **kwargs: kwargs["output"],
    )
    with pytest.raises(RuntimeError, match="rerun to resume"):
        module._download("file-id", destination, expected_bytes=99)
    assert destination.with_suffix(".zip.part").read_bytes() == b"partial"


def _original_sequence(root: Path, sequence_id: str, count: int = 21) -> Path:
    directory = root / "MoCA" / "JPEGImages" / sequence_id
    directory.mkdir(parents=True)
    for number in range(count):
        Image.new("RGB", (8, 6), (number, 20, 30)).save(directory / f"{number:05d}.jpg")
    return directory


def _mask_sequence(mask_root: Path, original: Path, split: str, benchmark_id: str, targets: tuple[int, ...]) -> None:
    partition = "TrainDataset_per_sq" if split == "train" else "TestDataset_per_sq"
    base = mask_root / "MoCA_Video" / partition / benchmark_id
    (base / "Imgs").mkdir(parents=True)
    (base / "GT").mkdir()
    for number in targets:
        shutil.copy2(original / f"{number:05d}.jpg", base / "Imgs" / f"{number:05d}.jpg")
        Image.new("L", (8, 6), 255).save(base / "GT" / f"{number:05d}.png")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    original_root, mask_root = tmp_path / "original", tmp_path / "mask"
    snow = _original_sequence(original_root, "snow_leopard_4")
    direct = _original_sequence(original_root, "direct_test")
    _original_sequence(original_root, "unused")
    _mask_sequence(mask_root, snow, "train", "snow_leopard_4.1", (0, 5))
    _mask_sequence(mask_root, snow, "train", "snow_leopard_4.2", (10, 15))
    _mask_sequence(mask_root, direct, "test", "direct_test", (0, 5))
    return original_root, mask_root


def _config() -> dict:
    return {
        "name": "moca_mask_dense_v1", "boundary_policy": "manual_target_hull_v1",
        "materialization": "manifest_only", "require_binary_masks": True,
        "validation_fraction": 0.5, "validation_seed": 7,
        "expected": {"original_sequences": 3, "original_rgb_frames": 63,
                     "benchmark_train_sequences": 2, "benchmark_test_sequences": 1,
                     "manual_target_images": 6, "manual_masks": 6},
    }


def _build(tmp_path: Path) -> tuple[Path, Path]:
    original, mask = _fixture(tmp_path)
    output = tmp_path / "processed" / "moca_mask_dense_v1"
    build_moca_mask_dense(
        _config(), original_moca_root=original, moca_mask_root=mask,
        output_root=output, materialization="manifest_only", verify_counts=False,
    )
    return original, output


def test_moca_dense_build_preserves_manual_targets_bounds_and_identity(tmp_path):
    _, output = _build(tmp_path)
    result = verify_moca_mask_dense(output)
    assert result["targets"] == 6 and result["frames"] == 18
    manifest = pd.read_csv(output / "manifest" / "runtime_manifest.csv")
    assert set(manifest.dataset) == {"moca_mask_dense"}
    assert set(manifest[manifest.is_target].annotation_type) == {"official_manual"}
    assert manifest[~manifest.is_target].mask_path.isna().all()
    assert manifest.groupby("video_id").source_frame_number.apply(list).to_dict() == {
        "direct_test": list(range(6)), "snow_leopard_4.1": list(range(6)),
        "snow_leopard_4.2": list(range(10, 16)),
    }
    aliases = json.loads((output / "manifest" / "alias_manifest.json").read_text())
    assert aliases["resolved_mapping"]["snow_leopard_4.2"] == "snow_leopard_4"
    assert len(list((output / "audit" / "overlays").glob("*.png"))) == 6
    assert len(list((output / "audit" / "dense_clip_strips").glob("*.png"))) == 3
    second = build_moca_mask_dense(
        _config(), original_moca_root=tmp_path / "original", moca_mask_root=tmp_path / "mask",
        output_root=output, materialization="manifest_only", verify_counts=False,
    )
    assert second["targets"] == 6


def test_moca_dense_runtime_d1_and_s5_never_cross_subsegments(tmp_path):
    _, output = _build(tmp_path)
    manifest = output / "manifest" / "runtime_manifest.csv"
    frame = pd.read_csv(manifest)
    split = frame[frame.video_id == "snow_leopard_4.2"].split.iloc[0]
    dense = ManifestVideoCODDataset(manifest, split=split, clip_spec=ClipSpec(3, 1, 1), size=16)
    coarse = ManifestVideoCODDataset(
        manifest, split=split, clip_spec=ClipSpec(3, 5, 1), size=16,
        context_cadence="source_stride5", source_frame_step=5,
    )
    dense_sample = next(dense[index] for index in range(len(dense))
                        if dense[index]["video_id"] == "snow_leopard_4.2")
    coarse_sample = next(coarse[index] for index in range(len(coarse))
                         if coarse[index]["video_id"] == "snow_leopard_4.2")
    assert set(dense_sample["source_frame_indices"]) <= set(range(10, 16))
    assert set(coarse_sample["source_frame_indices"]) <= set(range(10, 16))
    assert coarse_sample["source_frame_step"] == 5


def test_moca_dense_verification_detects_changed_source_target(tmp_path):
    original, output = _build(tmp_path)
    Image.new("RGB", (8, 6), (255, 0, 0)).save(original / "MoCA" / "JPEGImages" / "direct_test" / "00000.jpg")
    with pytest.raises(ValueError, match="linked MoCA target changed"):
        verify_moca_mask_dense(output)
