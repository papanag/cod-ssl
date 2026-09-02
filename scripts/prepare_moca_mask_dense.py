#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from cod_ssl.data.preprocessing.prepare_moca_mask_dense import (
    build_moca_mask_dense,
    verify_moca_mask_dense,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the verified moca_mask_dense_v1 product")
    parser.add_argument("--config", required=True)
    parser.add_argument("--original-moca-root")
    parser.add_argument("--moca-mask-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--materialization", choices=("symlink", "hardlink", "manifest_only", "copy"), default=None)
    parser.add_argument("--boundary-policy", choices=("manual_target_hull_v1",), default="manual_target_hull_v1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--audit-sample-seed", type=int, default=42)
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_moca_mask_dense(args.output_root), indent=2))
        return
    if not args.original_moca_root or not args.moca_mask_root:
        parser.error("--original-moca-root and --moca-mask-root are required unless --verify-only is used")
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="moca-mask-dense-dry-run-") as directory:
            temporary = Path(directory) / "moca_mask_dense_v1"
            result = build_moca_mask_dense(
                args.config, original_moca_root=args.original_moca_root, moca_mask_root=args.moca_mask_root,
                output_root=temporary, materialization="manifest_only", overwrite=True,
                audit_sample_seed=args.audit_sample_seed,
            )
            print(json.dumps(result | {"dry_run": True, "published": False}, indent=2))
        return
    result = build_moca_mask_dense(
        args.config, original_moca_root=args.original_moca_root, moca_mask_root=args.moca_mask_root,
        output_root=args.output_root, materialization=args.materialization, overwrite=args.overwrite,
        audit_sample_seed=args.audit_sample_seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
