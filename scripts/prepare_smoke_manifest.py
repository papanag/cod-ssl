#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cod_ssl.data.manifests import create_stratified_smoke_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report")
    args = parser.parse_args()
    selected, report = create_stratified_smoke_manifest(
        args.train_manifest, args.output, size=args.size, seed=args.seed
    )
    report_path = Path(args.report) if args.report else Path(args.output).with_suffix(".selection.csv")
    report.to_csv(report_path, index=False)
    metadata = {
        "source_manifest": str(Path(args.train_manifest).resolve()),
        "output_manifest": str(Path(args.output).resolve()),
        "size": args.size,
        "seed": args.seed,
        "source_counts": selected.groupby("source").size().to_dict(),
        "cod10k_categories": int(
            selected.loc[selected.source == "cod10k", "smoke_stratum"].nunique()
        ),
    }
    Path(args.output).with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
