#!/usr/bin/env python3
"""Cache the deterministic cross-dataset identity and representative-hash audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from cod_ssl.utils.run import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moca-manifest", type=Path, required=True)
    parser.add_argument("--camotion-manifest", type=Path, required=True)
    parser.add_argument("--camotion-attributes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    identity = {
        "schema_version": 1,
        "auditor_sha256": file_sha256(__file__),
        "moca_manifest_sha256": file_sha256(args.moca_manifest),
        "camotion_manifest_sha256": file_sha256(args.camotion_manifest),
        "camotion_attribute_manifest_sha256": file_sha256(args.camotion_attributes),
    }
    if args.output.is_file() and not args.force:
        try:
            cached = json.loads(args.output.read_text())
        except (json.JSONDecodeError, OSError):
            cached = {}
        if cached.get("complete") is True and cached.get("identity") == identity:
            print(f"Using cached cross-dataset audit: {args.output}")
            print(json.dumps(cached, indent=2))
            return
    moca = pd.read_csv(args.moca_manifest)
    camotion = pd.read_csv(args.camotion_manifest)

    def representative_hashes(frame: pd.DataFrame) -> dict[str, str]:
        representatives = (
            frame.sort_values(["source_video_id", "frame_number"])
            .groupby("source_video_id").head(1)
        )
        pairs = zip(representatives.image_path, representatives.source_video_id)
        return {
            file_sha256(path): str(source)
            for path, source in tqdm(
                pairs, total=len(representatives), desc="hash representative frames",
                unit="sequence", dynamic_ncols=True,
            )
        }

    moca_hashes = representative_hashes(moca)
    camotion_hashes = representative_hashes(camotion)
    audit = {
        "complete": True,
        "identity": identity,
        "source_name_overlap": sorted(
            set(moca.source_video_id.astype(str))
            & set(camotion.source_video_id.astype(str))
        ),
        "representative_image_hash_overlap": sorted(set(moca_hashes) & set(camotion_hashes)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
