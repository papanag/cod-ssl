from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from cod_ssl.data.base_video_cod import VideoCODDataset, validate_video_sample
from cod_ssl.data.clip_sampler import ClipSampler, ClipSpec
from cod_ssl.data.video_transforms import ClipPairedTransform

REQUIRED_COLUMNS = {
    "dataset", "split", "video_id", "source_video_id", "frame_id", "frame_number",
    "image_path", "mask_path", "annotation_type",
}


def _optional(row: pd.Series, name: str, default: Any = None) -> Any:
    value = row.get(name, default)
    return default if pd.isna(value) else value


class ManifestVideoCODDataset(VideoCODDataset):
    """Release-agnostic adapter over a canonical, explicitly serialized video manifest."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        split: str,
        clip_spec: ClipSpec,
        training: bool = False,
        size: int = 384,
        regime: str | None = None,
    ):
        self.manifest_path = Path(manifest)
        frame = pd.read_csv(self.manifest_path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"video manifest missing columns: {sorted(missing)}")
        frame = frame[frame["split"] == split].copy()
        if regime is not None and "regime" in frame:
            frame = frame[frame["regime"].fillna("default") == regime].copy()
        if frame.empty:
            raise ValueError(f"manifest contains no rows for split={split!r}, regime={regime!r}")
        keys = ["source_video_id", "frame_number"]
        if frame.duplicated(keys).any():
            raise ValueError("duplicate source_video_id/frame_number keys in manifest")
        self.frame = frame.sort_values(keys, kind="stable").reset_index(drop=True)
        self.split, self.regime, self.clip_spec = split, regime, clip_spec
        self.transform = ClipPairedTransform(training=training, size=size)
        self.sampler = ClipSampler()
        self._groups = {
            str(key): group.sort_values("frame_number", kind="stable").reset_index(drop=True)
            for key, group in self.frame.groupby("video_id", sort=True)
        }
        self._lookup = [
            (video_id, position)
            for video_id, group in self._groups.items()
            for position in range(len(group))
        ]

    @property
    def video_ids(self) -> tuple[str, ...]:
        return tuple(self._groups)

    def frames_for_video(self, video_id: str) -> tuple[int, ...]:
        group = self._groups[video_id]
        return tuple(map(int, group["frame_number"]))

    def __len__(self) -> int:
        return len(self._lookup)

    def __getitem__(self, index: int) -> dict[str, Any]:
        video_id, target_position = self._lookup[index]
        group = self._groups[video_id]
        positions, valid = self.sampler.source_indices(
            list(map(int, group["frame_number"])), target_position, self.clip_spec
        )
        target = group.iloc[target_position]
        context = [group.iloc[position] for position in positions]
        with Image.open(str(target["mask_path"])) as raw_mask:
            mask = raw_mask.copy()
        images = []
        for row in context:
            with Image.open(str(row["image_path"])) as raw_image:
                images.append(raw_image.copy())
        frames, target_mask = self.transform(images, mask)
        fps = _optional(target, "fps")
        source_numbers = [int(row["frame_number"]) for row in context]
        timestamps = None if fps is None else [number / float(fps) for number in source_numbers]
        attributes = _optional(target, "attributes", {})
        if isinstance(attributes, str):
            attributes = json.loads(attributes)
        sample = {
            "frames": frames,
            "target_mask": target_mask,
            "target_index": self.clip_spec.target_index,
            "video_id": str(target["video_id"]),
            "source_video_id": str(target["source_video_id"]),
            "frame_id": str(target["frame_id"]),
            "frame_number": int(target["frame_number"]),
            "source_frame_indices": source_numbers,
            "timestamps_sec": timestamps,
            "fps": None if fps is None else float(fps),
            "dataset": str(target["dataset"]),
            "regime": _optional(target, "regime", self.regime),
            "split": str(target["split"]),
            "annotation_type": str(target["annotation_type"]),
            "valid_temporal_mask": valid,
            "attributes": attributes,
            "metadata": {"manifest": str(self.manifest_path),
                         "image_path": str(target["image_path"]),
                         "mask_path": str(target["mask_path"])},
        }
        validate_video_sample(sample)
        return sample


def assert_disjoint_video_splits(frame: pd.DataFrame) -> None:
    by_split = {
        split: set(map(str, group["source_video_id"]))
        for split, group in frame.groupby("split")
    }
    names = sorted(by_split)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = by_split[left] & by_split[right]
            if overlap:
                raise ValueError(f"source video leakage between {left} and {right}: {sorted(overlap)[:5]}")
