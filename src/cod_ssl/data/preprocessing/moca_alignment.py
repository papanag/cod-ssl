from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

from cod_ssl.data.preprocessing.moca_release_inventory import MaskSequence, OriginalSequence

MOCA_SEQUENCE_ALIASES = {
    "snow_leopard_4.1": "snow_leopard_4",
    "snow_leopard_4.2": "snow_leopard_4",
    "snow_leopard_5.1": "snow_leopard_5",
    "snow_leopard_5.2": "snow_leopard_5",
    "snow_leopard_5.3": "snow_leopard_5",
}


@dataclass(frozen=True)
class LegalRange:
    benchmark_sequence_id: str
    source_sequence_id: str
    start_source_frame: int
    end_source_frame: int

    @property
    def n_context_frames(self) -> int:
        return self.end_source_frame - self.start_source_frame + 1


def map_sequences(
    benchmark: dict[str, MaskSequence],
    original: dict[str, OriginalSequence],
    *,
    aliases: dict[str, str] = MOCA_SEQUENCE_ALIASES,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for benchmark_id in sorted(benchmark):
        source_id = benchmark_id if benchmark_id in original else aliases.get(benchmark_id)
        if source_id is None or source_id not in original:
            raise ValueError(f"no exact or explicit Original MoCA mapping for {benchmark_id}")
        if source_id != benchmark_id and aliases.get(benchmark_id) != source_id:
            raise ValueError(f"unlisted many-to-one mapping for {benchmark_id}")
        mapping[benchmark_id] = source_id
    return mapping


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_target_alignment(
    benchmark: dict[str, MaskSequence],
    original: dict[str, OriginalSequence],
    mapping: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = sum(len(sequence.images) for sequence in benchmark.values())
    progress = tqdm(total=total, desc="verify MoCA target alignment", unit="target", dynamic_ncols=True)
    for benchmark_id in sorted(benchmark):
        sequence, source = benchmark[benchmark_id], original[mapping[benchmark_id]]
        for frame_number, target_path in sequence.images.items():
            source_path = source.frames.get(frame_number)
            if source_path is None:
                raise ValueError(f"missing Original MoCA target candidate: {benchmark_id}/{frame_number:05d}")
            with Image.open(target_path) as target_image, Image.open(source_path) as source_image, Image.open(
                sequence.masks[frame_number]
            ) as mask_image:
                if target_image.size != source_image.size:
                    raise ValueError(f"MoCA target dimensions differ: {benchmark_id}/{frame_number:05d}")
                if target_image.size != mask_image.size:
                    raise ValueError(f"MoCA target/mask dimensions differ: {benchmark_id}/{frame_number:05d}")
            target_digest, source_digest = sha256(target_path), sha256(source_path)
            if target_digest != source_digest:
                raise ValueError(f"MoCA target content mismatch: {benchmark_id}/{frame_number:05d}")
            rows.append({
                "benchmark_sequence_id": benchmark_id,
                "source_sequence_id": source.sequence_id,
                "official_split": sequence.official_split,
                "source_frame_number": frame_number,
                "source_rgb_path": str(source_path),
                "target_rgb_path": str(target_path),
                "manual_mask_path": str(sequence.masks[frame_number]),
                "target_rgb_sha256": target_digest,
                "manual_mask_sha256": sha256(sequence.masks[frame_number]),
                "verified": True,
            })
            progress.update(1)
    progress.close()
    return rows


def resolve_manual_target_hulls(
    benchmark: dict[str, MaskSequence],
    original: dict[str, OriginalSequence],
    mapping: dict[str, str],
) -> dict[str, LegalRange]:
    ranges: dict[str, LegalRange] = {}
    for benchmark_id, sequence in benchmark.items():
        ids = list(sequence.images)
        legal = LegalRange(benchmark_id, mapping[benchmark_id], min(ids), max(ids))
        source_ids = original[legal.source_sequence_id].frames
        missing = set(range(legal.start_source_frame, legal.end_source_frame + 1)) - set(source_ids)
        if missing:
            raise ValueError(f"missing Original MoCA frames inside legal range for {benchmark_id}")
        ranges[benchmark_id] = legal
    by_source: dict[str, list[LegalRange]] = {}
    for legal in ranges.values():
        by_source.setdefault(legal.source_sequence_id, []).append(legal)
    for source_id, source_ranges in by_source.items():
        ordered = sorted(source_ranges, key=lambda value: value.start_source_frame)
        for left, right in zip(ordered, ordered[1:]):
            if right.start_source_frame <= left.end_source_frame:
                raise ValueError(
                    f"overlapping benchmark subsequence hulls in {source_id}: "
                    f"{left.benchmark_sequence_id}, {right.benchmark_sequence_id}"
                )
    return ranges


def assert_no_source_split_leakage(
    benchmark: dict[str, MaskSequence], mapping: dict[str, str], ranges: dict[str, LegalRange]
) -> None:
    source_splits: dict[str, set[str]] = {}
    source_frame_splits: dict[tuple[str, int], set[str]] = {}
    for benchmark_id, sequence in benchmark.items():
        source_id, legal = mapping[benchmark_id], ranges[benchmark_id]
        source_splits.setdefault(source_id, set()).add(sequence.official_split)
        for number in range(legal.start_source_frame, legal.end_source_frame + 1):
            source_frame_splits.setdefault((source_id, number), set()).add(sequence.official_split)
    mixed_sources = {key: value for key, value in source_splits.items() if len(value) > 1}
    mixed_frames = {key: value for key, value in source_frame_splits.items() if len(value) > 1}
    if mixed_sources or mixed_frames:
        raise ValueError(
            f"Original MoCA leakage across official splits: sources={sorted(mixed_sources)}, "
            f"frame_keys={sorted(mixed_frames)[:5]}"
        )
