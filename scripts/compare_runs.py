#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from cod_ssl.evaluation import compare_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dino_run")
    parser.add_argument("vjepa_run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--qualitative-count", type=int, default=24)
    args = parser.parse_args()
    output = compare_runs(
        args.dino_run, args.vjepa_run, args.output,
        qualitative_count=args.qualitative_count,
    )
    if hasattr(os, "sync"):
        os.sync()
    print(f"Comparison written to {output}")


if __name__ == "__main__":
    main()
