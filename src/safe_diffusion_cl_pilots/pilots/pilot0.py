from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from safe_diffusion_cl_pilots.data.build_lowdim_dataset import (
    fit_normalization,
    save_dppo_dataset,
    split_episode_ids,
)
from safe_diffusion_cl_pilots.data.replay_oopsie_hdf5 import OopsieHDF5, object_health, replay_file
from safe_diffusion_cl_pilots.envs.object_centric_obs import ObjectCentricObservation
from safe_diffusion_cl_pilots.envs.shelve_contexts import (
    environment_kwargs_from_metadata,
    make_context,
    make_context_from_metadata,
)
from safe_diffusion_cl_pilots.envs.state_enrichment import enrich_object_state
from safe_diffusion_cl_pilots.envs.validation import validate_context_pair
from safe_diffusion_cl_pilots.evaluation.metrics import (
    calibrate_damage_threshold,
    episode_row,
    pilot0_gate,
    select_critical_fragile_object,
    summarize_episode_rows,
)
from safe_diffusion_cl_pilots.utils.logging import write_json

from .common import Stage, parse_seeds, stage_parser
from .pilot_minus1 import _demo_paths


def _noop_losses(critical: str, count: int, environment_kwargs: dict[str, Any]) -> list[float]:
    losses: list[float] = []
    for seed in range(count):
        env = make_context("A", **environment_kwargs)
        try:
            try:
                env.reset(seed=seed)
            except TypeError:
                if hasattr(env, "seed"):
                    env.seed(seed)
                env.reset()
            initial = object_health(env)
            for _ in range(10):
                outcome = env.step(np.zeros(7, dtype=np.float32))
                if bool(outcome[2]):
                    break
            final = object_health(env, dict(outcome[-1] or {}))
            losses.append(max(0.0, initial[critical] - final[critical]))
        finally:
            env.close()
    return losses


def _context_manifest(critical: str, environment_kwargs: dict[str, Any]) -> dict[str, Any]:
    extractor = ObjectCentricObservation()
    result: dict[str, Any] = {
        "critical_fragile_object": critical,
        "environment_kwargs": environment_kwargs,
        "contexts": {},
    }
    for context in ("A", "B"):
        env = make_context(context, critical_fragile_object=critical, **environment_kwargs)
        try:
            try:
                reset = env.reset(seed=0)
            except TypeError:
                if hasattr(env, "seed"):
                    env.seed(0)
                reset = env.reset()
            observation = reset[0] if isinstance(reset, tuple) else reset
            observation = enrich_object_state(env, observation)
            result["contexts"][context] = {
                name: extractor.object_position(observation, name).tolist()
                for name in extractor.object_order
            }
            result["contexts"][context]["target_mat"] = np.asarray(
                observation.get("target_mat_pos", observation.get("mat_pos")), dtype=float
            ).tolist()
        finally:
            env.close()
    result["deterministic_displacement_m"] = 0.0
    return result


def _plots(rows: list[dict[str, Any]], per_object: dict[str, dict[str, float]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    labels = ["safe", "unsafe"]
    safe_counts = [sum(row["source_file_label"] == label and row["safe_success"] for row in rows) for label in labels]
    damaging_counts = [sum(row["source_file_label"] == label and row["damaging_success"] for row in rows) for label in labels]
    figure, axis = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    axis.bar(x - 0.2, safe_counts, width=0.4, label="safe successes")
    axis.bar(x + 0.2, damaging_counts, width=0.4, label="damaging successes")
    axis.set_xticks(x, labels)
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "source_group_comparison.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 4))
    names = sorted(per_object)
    axis.bar(names, [per_object[name]["mean_health_loss"] for name in names])
    axis.set_ylabel("Mean health loss in successful episodes")
    figure.tight_layout()
    figure.savefig(directory / "per_object_damage.png", dpi=160)
    plt.close(figure)


def run(project_root: Path, run_root: Path, seeds: list[int]) -> None:
    stage = Stage("pilot0", project_root, run_root)
    try:
        if not (run_root / "results" / "pilot_minus1" / "_SUCCESS").exists():
            raise RuntimeError("Pilot -1 _SUCCESS is required")
        paths = _demo_paths(run_root)
        metadata_by_source = {
            label: OopsieHDF5(path).environment_metadata() for label, path in paths.items()
        }
        environment_kwargs = environment_kwargs_from_metadata(metadata_by_source["safe"])
        records = []
        rejected = []
        for label, path in paths.items():
            added, failures = replay_file(
                path,
                label,
                lambda metadata=metadata_by_source[label]: make_context_from_metadata(metadata),
            )
            records.extend(added)
            rejected.extend({"source": label, **row} for row in failures)
        if rejected:
            raise RuntimeError(f"full replay rejected episodes: {rejected}")
        if len(records) != 90:
            raise RuntimeError(f"full replay yielded {len(records)} rather than 90 episodes")
        stage.mark("full production replay")
        critical, per_object = select_critical_fragile_object(records, 1.0)
        # Calibration can change which fragile object has the most meaningful
        # successful damage. Iterate to the fixed point, with a hard finite cap.
        for _ in range(3):
            epsilon, calibration = calibrate_damage_threshold(
                _noop_losses(critical, 20, environment_kwargs)
            )
            selected, per_object = select_critical_fragile_object(records, epsilon)
            if selected == critical:
                break
            critical = selected
        else:
            raise RuntimeError("critical-object selection did not stabilize after calibration")
        write_json(stage.result_dir / "damage_threshold_calibration.json", calibration)
        rows = [episode_row(record, critical, epsilon) for record in records]
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_source[row["source_file_label"]].append(row)
        summaries = {
            "safe": summarize_episode_rows(by_source["safe"]),
            "unsafe": summarize_episode_rows(by_source["unsafe"]),
            "pooled": summarize_episode_rows(rows),
        }
        stage.mark("automated relabeling and damage calibration")

        def factory_a(**kwargs: Any) -> Any:
            return make_context("A", **environment_kwargs, **kwargs)

        def factory_b(**kwargs: Any) -> Any:
            return make_context(
                "B", critical_fragile_object=critical, **environment_kwargs, **kwargs
            )

        context_validation = validate_context_pair(factory_a, factory_b, critical, reset_count=100)
        manifest = _context_manifest(critical, environment_kwargs)
        positions = manifest["contexts"]
        placement_distance = float(
            np.linalg.norm(
                np.asarray(positions["A"][critical]) - np.asarray(positions["B"][critical])
            )
        )
        placement_changed = placement_distance >= 0.05
        manifest["critical_placement_distance_m"] = placement_distance
        write_json(run_root / "artifacts" / "context_manifest.json", manifest)
        # The CPU partition has no display or GPU-backed EGL device. Seed 0
        # renders these two diagnostic images later inside its A100 allocation.
        write_json(
            run_root / "artifacts" / "context_reset_images.json",
            {"status": "DEFERRED_TO_SEED_0_GPU", "images": []},
        )
        gate = pilot0_gate(
            summaries["safe"],
            summaries["unsafe"],
            summaries["pooled"],
            context_validation["passed"],
            placement_changed,
        )
        gate["context_validation"] = context_validation
        gate["critical_fragile_object"] = critical
        gate["epsilon_damage"] = epsilon
        gate["environment_kwargs"] = environment_kwargs
        stage.mark("Context B validity suite and gate")
        safe_records = [
            record
            for record, row in zip(records, rows, strict=True)
            if row["safe_success"]
        ]
        processed = run_root / "data" / "processed"
        for seed in seeds:
            train_ids, validation_ids = split_episode_ids(
                [record.episode_id for record in safe_records], seed
            )
            train_records = [record for record in safe_records if record.episode_id in train_ids]
            validation_records = [record for record in safe_records if record.episode_id in validation_ids]
            normalizer = fit_normalization(train_records)
            seed_dir = processed / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            normalizer.to_npz(str(seed_dir / "normalization.npz"))
            save_dppo_dataset(
                train_records,
                seed_dir / "train.npz",
                seed_dir / "train_metadata.jsonl",
                normalizer,
            )
            save_dppo_dataset(
                validation_records,
                seed_dir / "validation.npz",
                seed_dir / "validation_metadata.jsonl",
                normalizer,
            )
            write_json(seed_dir / "split.json", {"train": train_ids, "validation": validation_ids})
        dataframe = pd.DataFrame(rows)
        dataframe.to_parquet(stage.result_dir / "metrics_by_episode.parquet", index=False)
        summary = {
            "critical_fragile_object": critical,
            "epsilon_damage": epsilon,
            "by_source_and_pooled": summaries,
            "per_object": per_object,
        }
        write_json(stage.result_dir / "summary.json", summary)
        write_json(stage.result_dir / "gate.json", gate)
        _plots(rows, per_object, stage.result_dir / "figures")
        stage.result_dir.joinpath("report.md").write_text(
            f"# Pilot 0: {gate['color']}\n\n"
            f"Critical fragile object: `{critical}`. Damage threshold: {epsilon:.3f}.\n\n"
            f"Safe successes: {summaries['pooled']['safe_success_count']}; damaging successes: "
            f"{summaries['pooled']['damaging_success_count']}.\n\n"
            f"GPU work permitted: `{str(gate['proceed_gpu']).lower()}`. "
            f"Reasons: {gate['reasons'] or ['all gates passed']}.\n"
        )
        stage.succeed({"status": gate["color"], **gate})
    except Exception as error:
        stage.fail(error)
        raise


def main() -> None:
    parser = stage_parser("Pilot 0 safety-conflict validation")
    arguments = parser.parse_args()
    run(arguments.project_root, arguments.run_root, parse_seeds(arguments.seeds))


if __name__ == "__main__":
    main()
