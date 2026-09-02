from __future__ import annotations

from collections.abc import Iterable, Mapping

ATTRIBUTE_CODES = ("MO", "BO", "SO", "UE", "OC", "SC", "OV", "MB")


def parse_camotion_attributes(lines: Iterable[str]) -> dict[str, dict[str, bool]]:
    """Parse the official sequence-level attribute file without fuzzy matching."""
    parsed: dict[str, dict[str, bool]] = {}
    allowed = set(ATTRIBUTE_CODES)
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split(maxsplit=1)
        sequence_id = fields[0]
        if sequence_id in parsed:
            raise ValueError(f"duplicate CAMotion attribute row: {sequence_id!r} (line {line_number})")
        present = set()
        if len(fields) == 2 and fields[1].strip():
            codes = [code.strip() for code in fields[1].split(",")]
            if any(not code for code in codes):
                raise ValueError(f"empty CAMotion attribute code on line {line_number}")
            present = set(codes)
        unknown = present - allowed
        if unknown:
            raise ValueError(f"unknown CAMotion attribute codes for {sequence_id}: {sorted(unknown)}")
        parsed[sequence_id] = {code: code in present for code in ATTRIBUTE_CODES}
    return parsed


def align_camotion_attributes(
    sequence_ids: Iterable[str],
    attributes: Mapping[str, dict[str, bool]],
    *,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, dict[str, bool]]:
    """Apply only explicit metadata-ID -> dataset-ID aliases and require a bijection."""
    aliases = dict(aliases or {})
    unknown_alias_sources = set(aliases) - set(attributes)
    if unknown_alias_sources:
        raise ValueError(f"CAMotion aliases reference absent metadata IDs: {sorted(unknown_alias_sources)}")
    normalized: dict[str, dict[str, bool]] = {}
    for metadata_id, values in attributes.items():
        dataset_id = aliases.get(metadata_id, metadata_id)
        if dataset_id in normalized:
            raise ValueError(f"CAMotion alias collision for dataset sequence: {dataset_id}")
        normalized[dataset_id] = dict(values)
    dataset_ids = set(sequence_ids)
    missing = dataset_ids - set(normalized)
    extra = set(normalized) - dataset_ids
    if missing or extra:
        raise ValueError(
            f"CAMotion sequence/attribute mismatch: missing_metadata={sorted(missing)}, "
            f"extra_metadata={sorted(extra)}"
        )
    return {sequence_id: normalized[sequence_id] for sequence_id in sorted(dataset_ids)}
