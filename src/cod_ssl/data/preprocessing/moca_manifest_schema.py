from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class SequenceIdentity:
    benchmark_sequence_id: str
    source_sequence_id: str
    official_split: Literal["train", "test"]
    derived_split: Literal["train", "val", "test"] | None
    is_source_subsegment: bool


@dataclass(frozen=True)
class FrameIdentity:
    source_frame_number: int
    benchmark_sequence_position: int
    source_rgb_relpath: str
    is_manual_target: bool
    target_rgb_relpath: str | None
    target_mask_relpath: str | None


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[Any]) -> None:
    with path.open("w") as handle:
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                row = asdict(row)
            handle.write(canonical_json(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(manifest_dir: Path) -> dict[str, str]:
    names = sorted(
        path.name for path in manifest_dir.iterdir()
        if path.is_file() and path.name != "manifest_checksums.sha256"
    )
    checksums = {name: file_sha256(manifest_dir / name) for name in names}
    (manifest_dir / "manifest_checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items())
    )
    return checksums


def verify_checksums(manifest_dir: Path) -> dict[str, str]:
    checksum_path = manifest_dir / "manifest_checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"missing manifest checksum file: {checksum_path}")
    expected: dict[str, str] = {}
    for line in checksum_path.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        if Path(name).name != name:
            raise ValueError(f"unsafe manifest checksum path: {name}")
        expected[name] = digest
    for name, digest in expected.items():
        path = manifest_dir / name
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"manifest checksum mismatch: {name}")
    return expected
