from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args()
    gate_path = arguments.run_root / "results" / "pilot0" / "gate.json"
    gate = json.loads(gate_path.read_text())
    if gate.get("proceed_gpu", False):
        return
    output = arguments.run_root / "results" / f"seed_{arguments.seed}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "SKIPPED_PILOT0_RED.json").write_text(
        json.dumps({"seed": arguments.seed, "gate": gate}, indent=2, sort_keys=True) + "\n"
    )
    raise SystemExit(3)


if __name__ == "__main__":
    main()

