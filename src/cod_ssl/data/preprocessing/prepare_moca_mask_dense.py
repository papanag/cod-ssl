from __future__ import annotations

import json
import os
import platform
import random
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from cod_ssl.data.preprocessing.moca_alignment import (
    MOCA_SEQUENCE_ALIASES,
    assert_no_source_split_leakage,
    map_sequences,
    resolve_manual_target_hulls,
    sha256,
    verify_target_alignment,
)
from cod_ssl.data.preprocessing.moca_manifest_schema import (
    file_sha256,
    read_jsonl,
    verify_checksums,
    write_checksums,
    write_json,
    write_jsonl,
)
from cod_ssl.data.preprocessing.moca_release_inventory import (
    inventory_moca_mask,
    inventory_original_moca,
)
from cod_ssl.utils.run import git_commit

BUILD_ID = "moca_mask_dense_v1"
BOUNDARY_POLICY = "manual_target_hull_v1"


def _load_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return json.loads(json.dumps(config))
    value = yaml.safe_load(Path(config).read_text())
    return value["dataset_build"] if "dataset_build" in value else value


def _derived_splits(sequence_ids: list[str], benchmark, mapping: dict[str, str], fraction: float, seed: int) -> dict[str, str]:
    train_ids = sorted(key for key in sequence_ids if benchmark[key].official_split == "train")
    source_groups: dict[str, list[str]] = {}
    for key in train_ids:
        source_groups.setdefault(mapping[key], []).append(key)
    shuffled = sorted(source_groups)
    random.Random(seed).shuffle(shuffled)
    n_val_sources = 0 if fraction <= 0 else max(1, round(len(shuffled) * fraction))
    validation_sources = set(shuffled[:n_val_sources])
    validation = {key for key in train_ids if mapping[key] in validation_sources}
    return {
        key: "test" if benchmark[key].official_split == "test" else "val" if key in validation else "train"
        for key in sequence_ids
    }


def _relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _materialize(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(os.path.relpath(source, destination.parent))
        if destination.resolve() != source.resolve():
            raise ValueError(f"created symlink does not resolve to source: {destination}")
    elif mode == "hardlink":
        if source.stat().st_dev != destination.parent.stat().st_dev:
            raise ValueError("hardlink materialization requires source and output on the same device")
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    elif mode != "manifest_only":
        raise ValueError(f"unsupported materialization: {mode}")


def _source_inventory_rows(original, benchmark) -> tuple[list[dict], list[dict]]:
    original_rows = [
        {"source_sequence_id": key, "n_rgb_frames": len(value.frames), "first_frame": min(value.frames),
         "last_frame": max(value.frames), "zero_based_consecutive": True}
        for key, value in sorted(original.items())
    ]
    mask_rows = [
        {"benchmark_sequence_id": key, "official_split": value.official_split,
         "n_target_rgb": len(value.images), "n_manual_masks": len(value.masks),
         "first_target": min(value.images), "last_target": max(value.images),
         "cadence_exception_count": len(value.cadence_exceptions)}
        for key, value in sorted(benchmark.items())
    ]
    return original_rows, mask_rows


def _write_visual_audit(
    audit_dir: Path, target_rows: list[dict[str, Any]], frame_rows: list[dict[str, Any]],
    original_root: Path, mask_root: Path, *, seed: int,
) -> None:
    overlays, strips = audit_dir / "overlays", audit_dir / "dense_clip_strips"
    overlays.mkdir(); strips.mkdir()
    rng = random.Random(seed)
    forced_ids = sorted({
        row["benchmark_sequence_id"] for row in target_rows
        if row["benchmark_sequence_id"].startswith("snow_leopard_")
    })
    selected = []
    for benchmark_id in forced_ids:
        candidates = [row for row in target_rows if row["benchmark_sequence_id"] == benchmark_id]
        if candidates:
            selected.append(candidates[len(candidates) // 2])
    remaining = [row for row in target_rows if row not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, min(20, len(target_rows)) - len(selected))])
    for index, row in enumerate(tqdm(
        selected, desc="render MoCA overlays", unit="overlay", dynamic_ncols=True
    )):
        with Image.open(original_root / row["source_rgb_relpath"]) as raw_image, Image.open(
            mask_root / row["manual_mask_relpath"]
        ) as raw_mask:
            image, mask = raw_image.convert("RGB"), raw_mask.convert("L")
            red = Image.new("RGB", image.size, (255, 0, 0))
            image = Image.composite(red, image, mask.point(lambda value: 100 if value else 0))
        image.thumbnail((480, 320))
        canvas = Image.new("RGB", (500, 360), "white"); canvas.paste(image, (10, 10))
        ImageDraw.Draw(canvas).text(
            (10, 335),
            f"{row['official_split']} {row['benchmark_sequence_id']} source={row['source_frame_number']}",
            fill="black",
        )
        canvas.save(overlays / f"target_{index:02d}.png")

    by_sequence: dict[str, list[dict[str, Any]]] = {}
    for row in frame_rows:
        by_sequence.setdefault(row["benchmark_sequence_id"], []).append(row)
    strip_ids = forced_ids + [key for key in sorted(by_sequence) if key not in forced_ids]
    strip_ids = strip_ids[: min(10, len(strip_ids))]
    for index, benchmark_id in enumerate(tqdm(
        strip_ids, desc="render MoCA dense strips", unit="strip", dynamic_ncols=True
    )):
        rows = by_sequence[benchmark_id]; center = len(rows) // 2
        chosen = rows[max(0, center - 2): min(len(rows), center + 3)]
        panels = []
        for row in chosen:
            with Image.open(original_root / row["source_rgb_relpath"]) as raw:
                panel = raw.convert("RGB"); panel.thumbnail((180, 130))
            canvas = Image.new("RGB", (190, 155), "white")
            canvas.paste(panel, ((190 - panel.width) // 2, 0))
            ImageDraw.Draw(canvas).text((4, 134), f"source={row['source_frame_number']}", fill="black")
            panels.append(canvas)
        strip = Image.new("RGB", (190 * len(panels), 155), "white")
        for panel_index, panel in enumerate(panels):
            strip.paste(panel, (190 * panel_index, 0))
        strip.save(strips / f"clip_{index:02d}_{benchmark_id}.png")


def _build_into(
    destination: Path,
    config: dict[str, Any],
    original_root: Path,
    mask_root: Path,
    *,
    published_output: Path,
    materialization: str,
    verify_counts: bool,
) -> dict[str, Any]:
    manifest_dir, audit_dir = destination / "manifest", destination / "audit"
    manifest_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)
    expected = config.get("expected", {})
    _, original = inventory_original_moca(
        original_root,
        expected_sequences=int(expected.get("original_sequences", 141)),
        expected_frames=int(expected.get("original_rgb_frames", 37_250)),
        verify_counts=verify_counts,
    )
    annotation_candidates = sorted(original_root.rglob("annotations.csv"))
    if verify_counts and len(annotation_candidates) != 1:
        raise ValueError(
            f"expected one Original MoCA annotations.csv, found {len(annotation_candidates)}"
        )
    annotations_metadata = None
    if len(annotation_candidates) == 1:
        annotations_path = annotation_candidates[0].resolve()
        annotations_metadata = {
            "relpath": _relpath(annotations_path, original_root),
            "bytes": annotations_path.stat().st_size,
            "sha256": sha256(annotations_path),
        }
    benchmark, mask_quality = inventory_moca_mask(
        mask_root,
        expected_train_sequences=int(expected.get("benchmark_train_sequences", 71)),
        expected_test_sequences=int(expected.get("benchmark_test_sequences", 16)),
        expected_targets=int(expected.get("manual_target_images", 4_691)),
        verify_counts=verify_counts,
        require_binary_masks=bool(config.get("require_binary_masks", True)),
    )
    original_inventory, mask_inventory = _source_inventory_rows(original, benchmark)
    write_json(audit_dir / "source_inventory.json", {
        "original_moca": original_inventory, "original_annotations": annotations_metadata,
        "moca_mask": mask_inventory,
    })
    mapping = map_sequences(benchmark, original)
    alignment = verify_target_alignment(benchmark, original, mapping)
    ranges = resolve_manual_target_hulls(benchmark, original, mapping)
    assert_no_source_split_leakage(benchmark, mapping, ranges)
    derived = _derived_splits(
        list(benchmark), benchmark, mapping,
        float(config.get("validation_fraction", 0.1)), int(config.get("validation_seed", 42)),
    )

    sequence_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    aligned_by_key = {
        (row["benchmark_sequence_id"], int(row["source_frame_number"])): row for row in alignment
    }
    materialize_total = sum(value.n_context_frames for value in ranges.values())
    progress = tqdm(total=materialize_total, desc="write MoCA dense view", unit="frame", dynamic_ncols=True)
    ordered_ids = sorted(benchmark, key=lambda key: (benchmark[key].official_split != "train", key))
    for benchmark_id in ordered_ids:
        mask_sequence, source = benchmark[benchmark_id], original[mapping[benchmark_id]]
        legal = ranges[benchmark_id]
        sequence_rows.append({
            "schema_version": 1, "dataset_build_id": BUILD_ID,
            "benchmark_sequence_id": benchmark_id, "source_sequence_id": source.sequence_id,
            "official_split": mask_sequence.official_split, "derived_split": derived[benchmark_id],
            "is_source_subsegment": benchmark_id != source.sequence_id,
            "boundary_policy": BOUNDARY_POLICY,
            "start_source_frame": legal.start_source_frame, "end_source_frame": legal.end_source_frame,
            "n_context_frames": legal.n_context_frames, "n_manual_targets": len(mask_sequence.images),
            "source_is_consecutive": True,
        })
        for position, number in enumerate(range(legal.start_source_frame, legal.end_source_frame + 1)):
            source_path = source.frames[number]
            match = aligned_by_key.get((benchmark_id, number))
            target_path = None if match is None else Path(str(match["target_rgb_path"]))
            mask_path = None if match is None else Path(str(match["manual_mask_path"]))
            source_rel = _relpath(source_path, original_root)
            target_rel = None if target_path is None else _relpath(target_path, mask_root)
            mask_rel = None if mask_path is None else _relpath(mask_path, mask_root)
            frame_rows.append({
                "schema_version": 1, "dataset_build_id": BUILD_ID,
                "benchmark_sequence_id": benchmark_id, "source_sequence_id": source.sequence_id,
                "official_split": mask_sequence.official_split, "derived_split": derived[benchmark_id],
                "benchmark_sequence_position": position, "source_frame_number": number,
                "source_rgb_relpath": source_rel, "is_manual_target": match is not None,
                "target_rgb_relpath": target_rel, "target_mask_relpath": mask_rel,
                "boundary_policy": BOUNDARY_POLICY,
            })
            if materialization != "manifest_only":
                linked_image = destination / derived[benchmark_id] / benchmark_id / "frames" / source_path.name
                _materialize(source_path, linked_image, materialization)
                if mask_path is not None:
                    linked_mask = destination / derived[benchmark_id] / benchmark_id / "masks" / mask_path.name
                    _materialize(mask_path, linked_mask, materialization)
            runtime_rows.append({
                "dataset": "moca_mask_dense", "regime": "D1", "split": derived[benchmark_id],
                "video_id": benchmark_id, "benchmark_sequence_id": benchmark_id,
                "source_video_id": source.sequence_id, "source_sequence_id": source.sequence_id,
                "frame_id": f"{number:05d}", "frame_number": number, "source_frame_number": number,
                "sequence_position": position, "image_path": str(source_path.resolve()),
                "mask_path": "" if mask_path is None else str(mask_path.resolve()),
                "annotation_type": "official_manual" if match is not None else "context_only",
                "is_target": match is not None, "release_profile": BUILD_ID,
                "context_cadence": "dense_source_stride1", "released_frame_step": 1,
                "source_frame_step": 1, "dense_intermediate_rgb_available": True,
                "boundary_policy": BOUNDARY_POLICY, "official_partition": mask_sequence.official_split,
            })
            if match is not None:
                target_rows.append({
                    "schema_version": 1, "dataset_build_id": BUILD_ID,
                    "target_key": f"{mask_sequence.official_split}/{benchmark_id}/{number:05d}",
                    "benchmark_sequence_id": benchmark_id, "source_sequence_id": source.sequence_id,
                    "official_split": mask_sequence.official_split, "derived_split": derived[benchmark_id],
                    "source_frame_number": number, "benchmark_sequence_position": position,
                    "source_rgb_relpath": source_rel, "moca_mask_rgb_relpath": target_rel,
                    "manual_mask_relpath": mask_rel, "target_rgb_sha256": match["target_rgb_sha256"],
                    "manual_mask_sha256": match["manual_mask_sha256"],
                    "annotation_type": "manual", "boundary_policy": BOUNDARY_POLICY,
                })
            progress.update(1)
    progress.close()
    if len(target_rows) != int(expected.get("manual_target_images", 4_691)) and verify_counts:
        raise ValueError(f"derived target count differs: {len(target_rows)}")

    write_jsonl(manifest_dir / "sequence_manifest.jsonl", sequence_rows)
    write_jsonl(manifest_dir / "frame_manifest.jsonl", frame_rows)
    write_jsonl(manifest_dir / "target_manifest.jsonl", target_rows)
    split_manifest = {
        "schema_version": 1, "official_train": sorted(key for key in benchmark if benchmark[key].official_split == "train"),
        "official_test": sorted(key for key in benchmark if benchmark[key].official_split == "test"),
        "derived_train": sorted(key for key in benchmark if derived[key] == "train"),
        "derived_val": sorted(key for key in benchmark if derived[key] == "val"),
        "derived_test": sorted(key for key in benchmark if derived[key] == "test"),
        "validation_fraction": float(config.get("validation_fraction", 0.1)),
        "validation_seed": int(config.get("validation_seed", 42)),
    }
    write_json(manifest_dir / "split_manifest.json", split_manifest)
    write_json(manifest_dir / "alias_manifest.json", {
        "schema_version": 1, "exact_name_mapping": True,
        "aliases": MOCA_SEQUENCE_ALIASES, "resolved_mapping": mapping,
    })
    resolved = config | {
        "name": BUILD_ID, "original_moca_root": str(original_root.resolve()),
        "moca_mask_root": str(mask_root.resolve()), "output_root": str(published_output.resolve()),
        "materialization": materialization, "boundary_policy": BOUNDARY_POLICY,
    }
    (manifest_dir / "build_config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True))
    pd.DataFrame(runtime_rows).sort_values(
        ["split", "benchmark_sequence_id", "sequence_position"], kind="stable"
    ).to_csv(manifest_dir / "runtime_manifest.csv", index=False)

    pd.DataFrame(original_inventory).to_csv(audit_dir / "source_inventory_original_moca.csv", index=False)
    pd.DataFrame(mask_inventory).to_csv(audit_dir / "source_inventory_moca_mask.csv", index=False)
    pd.DataFrame([{"benchmark_sequence_id": key, "source_sequence_id": value} for key, value in sorted(mapping.items())]).to_csv(
        audit_dir / "sequence_mapping.csv", index=False
    )
    pd.DataFrame(alignment).to_csv(audit_dir / "target_alignment.csv", index=False)
    pd.DataFrame(sequence_rows).to_csv(audit_dir / "boundary_report.csv", index=False)
    pd.DataFrame([
        {"benchmark_sequence_id": key, "left_source_frame": left, "right_source_frame": right,
         "observed_step": right - left}
        for key, sequence in sorted(benchmark.items()) for left, right in sequence.cadence_exceptions
    ]).to_csv(audit_dir / "cadence_exceptions.csv", index=False)
    pd.DataFrame(mask_quality).to_csv(audit_dir / "mask_quality.csv", index=False)
    write_json(audit_dir / "leakage_report.json", {
        "train_test_benchmark_overlap": [], "train_test_source_frame_overlap": [],
        "source_sequences_in_multiple_official_splits": [], "passed": True,
    })
    write_json(audit_dir / "unmatched.json", {"unmatched": [], "ambiguous": [], "hash_mismatches": []})
    _write_visual_audit(
        audit_dir, target_rows, frame_rows, original_root, mask_root,
        seed=int(config.get("audit_sample_seed", 42)),
    )

    release = {
        "schema_version": 1, "dataset_build_id": BUILD_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "original_moca": {"source": "official", "archive_sha256": None, "archive_size_bytes": None,
            "discovered_sequences": len(original),
            "discovered_rgb_frames": sum(len(value.frames) for value in original.values()),
            "all_sequences_zero_based_consecutive": True,
            "annotations": annotations_metadata},
        "moca_mask": {"source": "official", "archive_sha256": None, "archive_size_bytes": None,
            "paper_reported_total_frames": 22_939,
            "discovered_target_rgb_frames": len(target_rows), "discovered_manual_masks": len(target_rows),
            "dense_rgb_present_in_release": False},
        "derived": {"boundary_policy": BOUNDARY_POLICY, "derived_dense_context_frames": len(frame_rows),
            "verified_target_matches": len(alignment), "exact_reconstruction_of_paper_22939": False,
            "materialization": materialization},
        "preprocessing": {"python": sys.version.split()[0], "platform": platform.platform(),
                          "code_commit": git_commit(Path.cwd())},
    }
    write_json(manifest_dir / "release_manifest.json", release)
    checksums = write_checksums(manifest_dir)
    summary = {
        "dataset_build_id": BUILD_ID, "boundary_policy": BOUNDARY_POLICY,
        "benchmark_sequences": len(benchmark), "manual_targets": len(target_rows),
        "derived_dense_context_frames": len(frame_rows), "verified_target_matches": len(alignment),
        "manifest_checksums": checksums,
    }
    write_json(audit_dir / "summary.json", summary)
    (audit_dir / "summary.md").write_text(
        "# MoCA-Mask dense preprocessing audit\n\n"
        f"- Benchmark sequences: {len(benchmark)}\n- Manual targets: {len(target_rows)}\n"
        f"- Derived dense context frames: {len(frame_rows)}\n- Boundary policy: `{BOUNDARY_POLICY}`\n\n"
        "The public MoCA-Mask release contains 4,691 target RGB/mask pairs, not the "
        "paper-described 22,939 dense RGB frames. Dense temporal context in this derived "
        "product comes from byte-verified alignment to Original MoCA. Legal subsequence "
        f"bounds use `{BOUNDARY_POLICY}` and are not claimed to reconstruct an unpublished release exactly.\n"
    )
    return summary


def verify_moca_mask_dense(output_root: str | Path, *, verify_linked_targets: bool = True) -> dict[str, Any]:
    output = Path(output_root).resolve()
    manifest_dir = output / "manifest"
    verify_checksums(manifest_dir)
    release = json.loads((manifest_dir / "release_manifest.json").read_text())
    if release.get("dataset_build_id") != BUILD_ID:
        raise ValueError("processed dataset is not moca_mask_dense_v1")
    if release["derived"].get("boundary_policy") != BOUNDARY_POLICY:
        raise ValueError("processed MoCA boundary policy mismatch")
    sequences = read_jsonl(manifest_dir / "sequence_manifest.jsonl")
    frames = read_jsonl(manifest_dir / "frame_manifest.jsonl")
    targets = read_jsonl(manifest_dir / "target_manifest.jsonl")
    if len(targets) != release["derived"]["verified_target_matches"]:
        raise ValueError("target manifest count differs from verified release metadata")
    sequence_keys = {row["benchmark_sequence_id"] for row in sequences}
    frame_keys = {(row["benchmark_sequence_id"], row["source_frame_number"]) for row in frames}
    if any(row["benchmark_sequence_id"] not in sequence_keys for row in targets):
        raise ValueError("target references an unknown benchmark sequence")
    if any((row["benchmark_sequence_id"], row["source_frame_number"]) not in frame_keys for row in targets):
        raise ValueError("target references an unknown dense frame")
    runtime = pd.read_csv(manifest_dir / "runtime_manifest.csv")
    if set(runtime["annotation_type"].unique()) - {"official_manual", "context_only"}:
        raise ValueError("dense MoCA runtime manifest contains non-manual target semantics")
    if verify_linked_targets:
        target_runtime = runtime[runtime["is_target"].astype(str).str.lower().eq("true")]
        digest_by_key = {
            (row["benchmark_sequence_id"], int(row["source_frame_number"])): row["target_rgb_sha256"]
            for row in targets
        }
        mask_digest_by_key = {
            (row["benchmark_sequence_id"], int(row["source_frame_number"])): row["manual_mask_sha256"]
            for row in targets
        }
        for row in tqdm(target_runtime.itertuples(), total=len(target_runtime),
                        desc="verify linked MoCA targets", unit="target", dynamic_ncols=True):
            if sha256(Path(row.image_path)) != digest_by_key[(row.benchmark_sequence_id, int(row.source_frame_number))]:
                raise ValueError(f"linked MoCA target changed: {row.benchmark_sequence_id}/{row.frame_id}")
            if not Path(row.mask_path).is_file():
                raise FileNotFoundError(f"missing linked manual mask: {row.mask_path}")
            if sha256(Path(row.mask_path)) != mask_digest_by_key[(row.benchmark_sequence_id, int(row.source_frame_number))]:
                raise ValueError(f"linked MoCA manual mask changed: {row.benchmark_sequence_id}/{row.frame_id}")
    materialization = release["derived"].get("materialization", "manifest_only")
    if materialization != "manifest_only":
        view_paths = [path for split in ("train", "val", "test") for path in (output / split).rglob("*")]
        broken = [str(path) for path in view_paths if path.is_symlink() and not path.exists()]
        if broken:
            raise ValueError(f"processed MoCA view contains broken links: {broken[:5]}")
    return {
        "dataset_build_id": BUILD_ID, "sequences": len(sequences), "frames": len(frames),
        "targets": len(targets), "manifest_checksums_sha256": file_sha256(manifest_dir / "manifest_checksums.sha256"),
    }


def build_moca_mask_dense(
    config: str | Path | dict[str, Any],
    *,
    original_moca_root: str | Path,
    moca_mask_root: str | Path,
    output_root: str | Path,
    materialization: str | None = None,
    overwrite: bool = False,
    verify_counts: bool = True,
    audit_sample_seed: int | None = None,
) -> dict[str, Any]:
    resolved = _load_config(config)
    if audit_sample_seed is not None:
        resolved["audit_sample_seed"] = int(audit_sample_seed)
    if resolved.get("name", BUILD_ID) != BUILD_ID:
        raise ValueError(f"dataset build name must be {BUILD_ID}")
    if resolved.get("boundary_policy", BOUNDARY_POLICY) != BOUNDARY_POLICY:
        raise ValueError(f"unsupported boundary policy: {resolved.get('boundary_policy')}")
    mode = materialization or resolved.get("materialization", "symlink")
    if mode not in {"symlink", "hardlink", "manifest_only", "copy"}:
        raise ValueError(f"unsupported materialization: {mode}")
    output = Path(output_root).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        existing = yaml.safe_load((output / "manifest" / "build_config_resolved.yaml").read_text())
        requested = {
            "original_moca_root": str(Path(original_moca_root).resolve()),
            "moca_mask_root": str(Path(moca_mask_root).resolve()),
            "materialization": mode, "boundary_policy": BOUNDARY_POLICY,
        }
        if any(existing.get(key) != value for key, value in requested.items()):
            raise FileExistsError(
                "existing dense MoCA build has different provenance/configuration; pass --overwrite explicitly"
            )
        return verify_moca_mask_dense(output)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        summary = _build_into(
            temporary, resolved, Path(original_moca_root).resolve(), Path(moca_mask_root).resolve(),
            published_output=output, materialization=mode, verify_counts=verify_counts,
        )
        verify_moca_mask_dense(temporary)
        if output.exists():
            backup = output.with_name(f"{output.name}.backup-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}")
            output.rename(backup)
        temporary.rename(output)
        return summary | verify_moca_mask_dense(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
