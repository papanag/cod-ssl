#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys

from tqdm.auto import tqdm

if __name__ == "__main__":
    for system in tqdm(
        ("DM", "VR"), desc="diagnostic systems", unit="run", dynamic_ncols=True
    ):
        subprocess.run([sys.executable, "scripts/train_probe.py", "--config",
                        "configs/experiments/vcod_diagnostics.yaml",
                        f"experiment.system_id={system}"], check=True)
