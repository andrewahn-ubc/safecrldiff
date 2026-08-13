from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

from safe_diffusion_cl_pilots.utils.logging import configure_logging, write_json
from safe_diffusion_cl_pilots.utils.manifests import atomic_success, write_source_versions


def stage_parser(description: str, seed: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2")
    if seed:
        parser.add_argument("--seed", type=int, required=True)
    return parser


def parse_seeds(raw: str) -> list[int]:
    result = [int(value) for value in raw.split(",") if value.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("--seeds must contain unique comma-separated integers")
    return result


class Stage:
    def __init__(self, name: str, project_root: Path, run_root: Path):
        self.name = name
        self.project_root = project_root.resolve()
        self.run_root = run_root.resolve()
        self.result_dir = self.run_root / "results" / name
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.logger = configure_logging(self.result_dir / "logs" / "stage.log")
        self.last_successful_step = "initialization"

    def mark(self, step: str) -> None:
        self.last_successful_step = step
        self.logger.info("completed: %s", step)

    def succeed(self, status: dict[str, Any]) -> None:
        write_json(self.result_dir / "status.json", status)
        atomic_success(self.result_dir, status)

    def fail(self, error: BaseException, stop_pipeline: bool = False) -> None:
        details = {
            "stage": self.name,
            "status": "FAILED",
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
            "last_successful_step": self.last_successful_step,
        }
        try:
            write_source_versions(self.project_root, self.run_root)
        except Exception as version_error:
            details["version_capture_error"] = str(version_error)
        write_json(self.result_dir / "status.json", details)
        if stop_pipeline:
            write_json(self.result_dir / "STOP_PIPELINE.json", details)
        (self.result_dir / "report.md").write_text(
            f"# {self.name} failure\n\n"
            f"Last successful step: `{self.last_successful_step}`\n\n"
            f"Blocker: `{type(error).__name__}: {error}`\n\n"
            "See `status.json` and `logs/stage.log` for machine-readable details.\n"
        )
        self.logger.exception("stage failed")

