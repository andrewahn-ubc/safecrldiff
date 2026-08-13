from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path


def archive(path: Path, timestamp: str) -> None:
    if path.exists():
        target = path.with_name(f"{path.name}.previous.{timestamp}")
        os.replace(path, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--from-stage", choices=("pilot-minus1", "pilot0", "pilots12", "aggregate"), required=True)
    parser.add_argument("--seeds", required=True)
    arguments = parser.parse_args()
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = arguments.run_root / "results"
    order = ("pilot-minus1", "pilot0", "pilots12", "aggregate")
    start = order.index(arguments.from_stage)
    if start <= 0:
        archive(results / "pilot_minus1", timestamp)
    if start <= 1:
        archive(results / "pilot0", timestamp)
    if start <= 2:
        for seed in arguments.seeds.split(","):
            archive(results / f"seed_{int(seed)}", timestamp)
    if start <= 3:
        archive(results / "aggregate", timestamp)
        for name in ("GO_NO_GO_REPORT.md", "summary.json", "all_metrics.parquet"):
            archive(results / name, timestamp)


if __name__ == "__main__":
    main()
