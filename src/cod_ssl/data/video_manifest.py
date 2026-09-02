from __future__ import annotations

import json
import hashlib
import random
from itertools import pairwise
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
        context_cadence: str | None = None,
        source_frame_step: int | None = None,
        release_profile: str | None = None,
        boundary_policy: str | None = None,
        temporal_order: str = "ordered",
        diagnostic_seed: int = 0,
        context_direction: str = "bidirectional",
        filter_regime: bool = True,
    ):
        self.manifest_path = Path(manifest)
        frame = pd.read_csv(self.manifest_path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"video manifest missing columns: {sorted(missing)}")
        frame = frame[frame["split"] == split].copy()
        if filter_regime and regime is not None and "regime" in frame:
            frame = frame[frame["regime"].fillna("default") == regime].copy()
        if frame.empty:
            raise ValueError(f"manifest contains no rows for split={split!r}, regime={regime!r}")
        keys = ["source_video_id", "frame_number"]
        if frame.duplicated(keys).any():
            raise ValueError("duplicate source_video_id/frame_number keys in manifest")
        ordering = ["video_id", "sequence_position"] if "sequence_position" in frame else keys
        self.frame = frame.sort_values(ordering, kind="stable").reset_index(drop=True)
        self.split, self.regime, self.clip_spec = split, regime, clip_spec
        self.context_cadence = context_cadence
        self.source_frame_step = source_frame_step
        self.release_profile = release_profile
        self.boundary_policy = boundary_policy
        if temporal_order not in {"ordered", "repeated", "shuffled"}:
            raise ValueError(f"unsupported temporal order: {temporal_order}")
        self.temporal_order = temporal_order
        self.diagnostic_seed = diagnostic_seed
        if context_direction not in {"causal", "bidirectional"}:
            raise ValueError(f"unsupported context direction: {context_direction}")
        self.context_direction = context_direction
        self.transform = ClipPairedTransform(training=training, size=size)
        self.sampler = ClipSampler()
        self._groups = {
            str(key): group.sort_values(
                "sequence_position" if "sequence_position" in group else "frame_number", kind="stable"
            ).reset_index(drop=True)
            for key, group in self.frame.groupby("video_id", sort=True)
        }
        self._lookup = []
        for video_id, group in self._groups.items():
            if "is_target" not in group:
                target_flags = [True] * len(group)
            elif pd.api.types.is_bool_dtype(group["is_target"]):
                target_flags = group["is_target"].fillna(False).tolist()
            else:
                target_flags = group["is_target"].fillna("").astype(str).str.lower().eq("true").tolist()
            self._lookup.extend(
                (video_id, position)
                for position, is_target in enumerate(target_flags)
                if is_target
            )

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
        if self.context_direction == "causal":
            for slot, position in enumerate(positions):
                if position > target_position:
                    positions[slot] = target_position
                    valid[slot] = False
        if self.temporal_order == "repeated":
            positions = [target_position] * self.clip_spec.length
            valid = valid.new_ones(self.clip_spec.length)
        elif self.temporal_order == "shuffled":
            slots = [index for index in range(self.clip_spec.length) if index != self.clip_spec.target_index]
            digest = hashlib.sha256(
                f"{video_id}/{target_position}/{self.diagnostic_seed}".encode()
            ).digest()
            shuffled = slots.copy(); random.Random(int.from_bytes(digest[:8], "big")).shuffle(shuffled)
            original_positions, original_valid = positions.copy(), valid.clone()
            for destination, source in zip(slots, shuffled):
                positions[destination] = original_positions[source]
                valid[destination] = original_valid[source]
        target = group.iloc[target_position]
        if pd.isna(target["mask_path"]) or not str(target["mask_path"]).strip():
            raise ValueError(f"supervised target has no mask: {video_id}/{target['frame_id']}")
        context = [group.iloc[position] for position in positions]
        with Image.open(str(target["mask_path"])) as raw_mask:
            mask = raw_mask.copy()
        images = []
        for row in context:
            with Image.open(str(row["image_path"])) as raw_image:
                images.append(raw_image.copy())
        frames, target_mask = self.transform(images, mask)
        fps = _optional(target, "fps")
        source_numbers = [int(_optional(row, "source_frame_number", row["frame_number"])) for row in context]
        expected_step = self.source_frame_step
        if expected_step is not None and self.temporal_order == "ordered":
            for index, (left, right) in enumerate(pairwise(source_numbers)):
                if bool(valid[index]) and bool(valid[index + 1]) and right - left != expected_step:
                    raise ValueError(
                        f"selected source-frame step differs from declared cadence for {video_id}: "
                        f"{left}->{right}, expected {expected_step}"
                    )
        if self.context_direction == "causal" and any(
            number > int(_optional(target, "source_frame_number", target["frame_number"]))
            for number in source_numbers
        ):
            raise ValueError("causal clip contains source frames after the target")
        source_positions = [int(_optional(row, "sequence_position", position)) for position, row in zip(positions, context)]
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
            "source_frame_number": int(_optional(target, "source_frame_number", target["frame_number"])),
            "sequence_position": int(_optional(target, "sequence_position", target_position)),
            "source_frame_indices": source_numbers,
            "source_sequence_positions": source_positions,
            "timestamps_sec": timestamps,
            "fps": None if fps is None else float(fps),
            "dataset": str(target["dataset"]),
            "regime": self.regime if self.regime is not None else _optional(target, "regime"),
            "split": str(target["split"]),
            "annotation_type": str(target["annotation_type"]),
            "release_profile": str(self.release_profile or _optional(target, "release_profile", "legacy_unspecified")),
            "context_cadence": str(
                "repeated_target" if self.temporal_order == "repeated"
                else f"shuffled_{'stride5' if (self.source_frame_step or 1) == 5 else 'dense'}"
                if self.temporal_order == "shuffled"
                else self.context_cadence or _optional(target, "context_cadence", "legacy_unspecified")
            ),
            "released_frame_step": int(self.clip_spec.stride),
            "source_frame_step": int(
                self.source_frame_step
                if self.source_frame_step is not None
                else _optional(target, "source_frame_step", self.clip_spec.stride)
            ),
            "dense_intermediate_rgb_available": bool(
                _optional(target, "dense_intermediate_rgb_available", False)
            ),
            "boundary_policy": self.boundary_policy or _optional(target, "boundary_policy"),
            "context_direction": self.context_direction,
            "valid_temporal_mask": valid,
            "attributes": attributes,
            "metadata": {
                "manifest": str(self.manifest_path),
                "image_path": str(target["image_path"]),
                "mask_path": str(target["mask_path"]),
                "attribute_scope": _optional(target, "attribute_scope"),
                "foreground_fraction": _optional(target, "foreground_fraction"),
                "bbox_available": bool(_optional(target, "bbox_available", False)),
                "bbox_path": _optional(target, "bbox_path"),
                "class": _optional(target, "class"),
                "subclass": _optional(target, "subclass"),
                "species": _optional(target, "species"),
                "preprocessing_manifest_hash": _optional(target, "preprocessing_manifest_hash"),
            },
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
