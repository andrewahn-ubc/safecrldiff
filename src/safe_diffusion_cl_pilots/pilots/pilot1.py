from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from safe_diffusion_cl_pilots.utils.logging import write_json
from safe_diffusion_cl_pilots.utils.manifests import atomic_success

from .common import Stage, stage_parser
from .training import run_policy_pipeline


def _scientific_summary(raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("skipped") or raw.get("skipped_rl"):
        return raw
    evaluations = raw["evaluations"]
    ordered = sorted(int(key) for key in evaluations)
    first, last = str(ordered[0]), str(ordered[-1])
    pre_a = evaluations[first]["A"]
    post_a = evaluations[last]["A"]
    pre_b = evaluations[first]["B"]["task_success"]
    post_b = evaluations[last]["B"]["task_success"]
    adjacent = False
    if len(ordered) >= 3:
        prior = evaluations[str(ordered[-2])]["A"]
        damage_pre = pre_a["p_damage_given_success"]
        adjacent = (
            prior["p_damage_given_success"] > damage_pre
            and post_a["p_damage_given_success"] > damage_pre
        )
    return {
        **raw,
        "a_pre": pre_a,
        "a_post": post_a,
        "b_pre": pre_b,
        "b_post": post_b,
        "adjacent_checkpoint_effect": adjacent,
        "extended": ordered[-1] > 150_000,
    }


def run(project_root: Path, run_root: Path, seed: int) -> None:
    stage = Stage(f"seed_{seed}/pilot1/diffusion", project_root, run_root)
    try:
        # Seed 0 alone applies the predeclared trigger. Other seeds receive the
        # extension only when seed 0 has already established it in the manifest.
        extension_manifest = run_root / "artifacts" / "adaptive_extension.json"
        maximum = 300_000 if seed == 0 else 150_000
        if seed != 0 and extension_manifest.exists():
            maximum = (
                300_000
                if json.loads(extension_manifest.read_text()).get("extend_other_seeds")
                else 150_000
            )
        raw = run_policy_pipeline(
            "diffusion", project_root, run_root, seed, stage.result_dir, maximum
        )
        summary = _scientific_summary(raw)
        if seed == 0:
            write_json(
                extension_manifest,
                {
                    "seed0_extended": bool(summary.get("extended", False)),
                    "extend_other_seeds": bool(
                        summary.get("extended", False)
                        and summary.get("b_competence_reached", False)
                    ),
                    "seed0_primitive_steps": summary.get("primitive_steps", 0),
                    "seed0_skipped": bool(raw.get("skipped") or raw.get("skipped_rl")),
                },
            )
        write_json(stage.result_dir / "summary.json", summary)
        for config_name in ("common", "diffusion_bc", "diffusion_dppo"):
            source = project_root / "configs" / f"{config_name}.yaml"
            target = stage.result_dir / ("config.yaml" if config_name == "common" else f"{config_name}.yaml")
            shutil.copyfile(source, target)
        atomic_success(stage.result_dir, {"seed": seed, "family": "diffusion"})
    except Exception as error:
        stage.fail(error)
        raise


def main() -> None:
    parser = stage_parser("Pilot 1 diffusion BC and task-only DPPO", seed=True)
    arguments = parser.parse_args()
    run(arguments.project_root, arguments.run_root, arguments.seed)


if __name__ == "__main__":
    main()
