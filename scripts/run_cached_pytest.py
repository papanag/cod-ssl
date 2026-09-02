#!/usr/bin/env python3
"""Run the repository test gate once per exact revision and environment."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from cod_ssl.utils.run import file_sha256, git_commit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    project = args.project_dir.resolve()
    tracked_inputs = [project / "pyproject.toml", project / "uv.lock"]
    identity = {
        "schema_version": 1,
        "git_commit": git_commit(project),
        "python": platform.python_version(),
        "pytest": pytest.__version__,
        "torch": torch.__version__,
        "input_sha256": {
            path.name: file_sha256(path) for path in tracked_inputs if path.is_file()
        },
    }
    if args.receipt.is_file() and not args.force:
        try:
            cached = json.loads(args.receipt.read_text())
        except (json.JSONDecodeError, OSError):
            cached = {}
        if cached.get("passed") is True and cached.get("identity") == identity:
            print(f"Using cached repository-test receipt: {args.receipt}")
            print(f"Passed at: {cached['completed_at_utc']}")
            return
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=project, check=True)
    receipt = {
        "passed": True,
        "identity": identity,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_suffix(args.receipt.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(args.receipt)
    print(f"Repository-test receipt written: {args.receipt}")


if __name__ == "__main__":
    main()
