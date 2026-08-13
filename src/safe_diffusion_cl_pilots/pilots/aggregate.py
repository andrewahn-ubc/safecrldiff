from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from safe_diffusion_cl_pilots.evaluation.metrics import pilot1_gate
from safe_diffusion_cl_pilots.utils.logging import write_json

from .common import Stage, parse_seeds, stage_parser


def _json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _gpu_results(run_root: Path, seeds: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    diffusion: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for seed in seeds:
        first = _json(run_root / "results" / f"seed_{seed}" / "pilot1" / "diffusion" / "summary.json")
        second = _json(
            run_root
            / "results"
            / f"seed_{seed}"
            / "pilot2"
            / "comparison"
            / "diffusion_vs_gaussian.json"
        )
        if first and "a_pre" in first:
            diffusion.append(first)
        if second:
            comparisons.append(second)
    return diffusion, comparisons


def _collect_metrics(run_root: Path, destination: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in (run_root / "results").glob("**/*metrics.parquet"):
        if ".partial." in path.name:
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        # The aggregate table intentionally contains scalar columns only. Full
        # trajectory payloads remain in their per-seed rollout tables.
        frame = frame[[column for column in frame if frame[column].map(np.isscalar).all()]]
        frame["artifact"] = str(path.relative_to(run_root))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(destination, index=False)
    return combined


def _plot_seed_curves(
    rollout: pd.DataFrame,
    metric: str,
    directory: Path,
    filename: str,
    ylabel: str,
    context_filter: str | None = None,
) -> None:
    selected = rollout if context_filter is None else rollout[rollout["context"] == context_filter]
    selected = selected.dropna(subset=[metric, "training_seed"])
    if selected.empty:
        return
    per_seed = (
        selected.groupby(["policy", "context", "training_seed", "checkpoint_steps"])[metric]
        .mean()
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    for (policy, context), group in per_seed.groupby(["policy", "context"]):
        for _, seed_group in group.groupby("training_seed"):
            seed_group = seed_group.sort_values("checkpoint_steps")
            axis.plot(
                seed_group["checkpoint_steps"],
                seed_group[metric],
                alpha=0.20,
                linewidth=1,
            )
        curve = group.groupby("checkpoint_steps")[metric].agg(["mean", "std", "count"])
        ci = 1.96 * curve["std"].fillna(0.0) / np.sqrt(curve["count"])
        x = curve.index.to_numpy(dtype=float)
        mean = curve["mean"].to_numpy(dtype=float)
        bound = ci.to_numpy(dtype=float)
        axis.plot(x, mean, marker="o", label=f"{policy} {context}")
        axis.fill_between(x, np.clip(mean - bound, 0, 1), np.clip(mean + bound, 0, 1), alpha=0.15)
    axis.set_xlabel("B fine-tuning primitive steps")
    axis.set_ylabel(ylabel)
    axis.set_ylim(-0.02, 1.02)
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / filename, dpi=160)
    plt.close(figure)


def _plots(
    frame: pd.DataFrame,
    directory: Path,
    comparisons: list[dict[str, Any]],
    pilot0_summary: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if {"source_file_label", "safe_success", "damaging_success"}.issubset(frame):
        source = frame.dropna(subset=["source_file_label"])
        if not source.empty:
            counts = source.groupby("source_file_label")[["safe_success", "damaging_success"]].sum()
            counts.plot.bar(figsize=(6, 4))
            plt.ylabel("Successful trajectories")
            plt.tight_layout()
            plt.savefig(directory / "pilot0_successful_trajectory_counts.png", dpi=160)
            plt.close()
    per_object = pilot0_summary.get("per_object", {})
    if per_object:
        names = sorted(per_object)
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.bar(names, [per_object[name]["mean_health_loss"] for name in names])
        axis.set_ylabel("Mean health loss in successful episodes")
        figure.tight_layout()
        figure.savefig(directory / "per_object_health_loss.png", dpi=160)
        plt.close(figure)
    if comparisons:
        route_rows = [
            {
                "seed": index,
                "policy": policy,
                "entropy": comparison["route_modes"][policy]["entropy"],
            }
            for index, comparison in enumerate(comparisons)
            for policy in ("diffusion", "gaussian")
        ]
        route_frame = pd.DataFrame(route_rows)
        figure, axis = plt.subplots(figsize=(6, 4))
        for policy, group in route_frame.groupby("policy"):
            axis.scatter([policy] * len(group), group["entropy"], alpha=0.5)
        route_frame.groupby("policy")["entropy"].mean().plot.bar(ax=axis, alpha=0.55)
        axis.set_ylabel("Route-mode entropy")
        figure.tight_layout()
        figure.savefig(directory / "diffusion_vs_gaussian_route_entropy.png", dpi=160)
        plt.close(figure)
        efficiency_rows = [
            {"seed": index, "policy": policy, **comparison["efficiency"][policy]}
            for index, comparison in enumerate(comparisons)
            for policy in ("diffusion", "gaussian")
            if "efficiency" in comparison
        ]
        if efficiency_rows:
            efficiency = pd.DataFrame(efficiency_rows)
            figure, axes = plt.subplots(1, 2, figsize=(9, 4))
            for axis, metric, label in (
                (axes[0], "walltime_seconds", "Wall-clock seconds"),
                (axes[1], "environment_steps", "Environment steps"),
            ):
                efficiency.groupby("policy")[metric].mean().plot.bar(ax=axis)
                axis.set_ylabel(label)
            figure.tight_layout()
            figure.savefig(directory / "diffusion_vs_gaussian_efficiency.png", dpi=160)
            plt.close(figure)
    if frame.empty or "checkpoint_steps" not in frame:
        return
    rollout = frame[frame["context"].notna()] if "context" in frame else pd.DataFrame()
    if rollout.empty:
        return
    rollout = rollout.copy()
    rollout["safe_success"] = rollout["task_success"].astype(bool) & ~rollout[
        "damage_event"
    ].astype(bool)
    rollout["damage_given_success"] = rollout["damage_event"].where(
        rollout["task_success"].astype(bool), np.nan
    )
    _plot_seed_curves(
        rollout, "task_success", directory, "task_success_vs_steps.png", "Task success"
    )
    _plot_seed_curves(
        rollout,
        "damage_given_success",
        directory,
        "a_damage_given_success_vs_steps.png",
        "A P(damage | success)",
        "A",
    )
    _plot_seed_curves(
        rollout,
        "safe_success",
        directory,
        "a_safe_success_vs_steps.png",
        "A safe success",
        "A",
    )
    if {"task_success", "safe_success"}.issubset(rollout):
        grouped = rollout.groupby(["policy", "context", "training_seed", "checkpoint_steps"])[
            ["task_success", "safe_success"]
        ].mean()
        figure, axis = plt.subplots(figsize=(5, 5))
        for (policy, context), group in grouped.groupby(level=[0, 1]):
            axis.scatter(group["task_success"], group["safe_success"], label=f"{policy} {context}")
        axis.set_xlabel("Task success")
        axis.set_ylabel("Safe success")
        axis.legend()
        figure.tight_layout()
        figure.savefig(directory / "safe_success_vs_task_success.png", dpi=160)
        plt.close(figure)


def run(project_root: Path, run_root: Path, seeds: list[int]) -> None:
    stage = Stage("aggregate", project_root, run_root)
    results = run_root / "results"
    pilot_minus1 = _json(results / "pilot_minus1" / "status.json") or {}
    pilot0_gate_data = _json(results / "pilot0" / "gate.json") or {}
    pilot0_summary = _json(results / "pilot0" / "summary.json") or {}
    diffusion, comparisons = _gpu_results(run_root, seeds)
    if not (results / "pilot_minus1" / "_SUCCESS").exists():
        overall = "TECHNICAL STOP"
        pilot1 = {"verdict": "NOT RUN"}
        pilot2 = {"verdict": "NOT RUN"}
        recommendation = "Fix the precise Pilot -1 blocker and rerun with --force-from pilot-minus1."
    elif not pilot0_gate_data.get("proceed_gpu", False):
        overall = "STOP"
        pilot1 = {"verdict": "NOT RUN"}
        pilot2 = {"verdict": "NOT RUN"}
        recommendation = "Choose a task with an automatically verified safe/damaging success conflict."
    else:
        pilot1 = pilot1_gate(diffusion) if diffusion else {"verdict": "NO-GO", "reason": "no complete diffusion seeds"}
        verdicts = Counter(item["gate"]["verdict"] for item in comparisons)
        if verdicts["GO"] >= 2:
            pilot2_verdict = "GO"
        elif verdicts["NO-GO"] >= 2:
            pilot2_verdict = "NO-GO"
        elif comparisons:
            pilot2_verdict = "YELLOW"
        else:
            pilot2_verdict = "NO-GO"
        pilot2 = {"verdict": pilot2_verdict, "per_seed": dict(verdicts)}
        if pilot1["verdict"] == "GO" and pilot2_verdict in {"GO", "YELLOW"}:
            overall = "GO"
            recommendation = "Proceed to the smallest predeclared diffusion-specific safety intervention."
        elif pilot1["verdict"] == "NO-GO":
            overall = "STOP"
            recommendation = "Do not add safety machinery until a task produces RL-induced safety forgetting."
        else:
            overall = "PIVOT"
            recommendation = "Separate generic capability retention from safety retention or use the stronger Gaussian class."
    frame = _collect_metrics(run_root, results / "all_metrics.parquet")
    _plots(frame, results / "figures", comparisons, pilot0_summary)
    summary = {
        "overall_verdict": overall,
        "pilot_minus1": pilot_minus1,
        "pilot0": {"gate": pilot0_gate_data, "summary": pilot0_summary},
        "pilot1": pilot1,
        "pilot2": pilot2,
        "completed_diffusion_seeds": len(diffusion),
        "completed_comparisons": len(comparisons),
        "recommendation": recommendation,
    }
    write_json(results / "summary.json", summary)
    blocker = pilot_minus1.get("exception", "none reported")
    report = f"""# Safety Forgetting Pilots: Go/No-Go Report

## Overall verdict

**{overall}:** {recommendation}

## Pilot -1

- Status: {pilot_minus1.get('status', 'not completed')}
- Official replay, DamageSim signal, conversion, and numerical checks: {pilot_minus1.get('status') == 'PASS'}
- Technical blocker, if any: {blocker}

## Pilot 0

- Gate: {pilot0_gate_data.get('color', 'not completed')}
- GPU work permitted: {pilot0_gate_data.get('proceed_gpu', False)}
- Critical fragile object: {pilot0_gate_data.get('critical_fragile_object', 'not selected')}
- A/B context valid: {pilot0_gate_data.get('context_validation', {}).get('passed', False)}

## Pilot 1

- Verdict: {pilot1.get('verdict')}
- Seeds with safety-specific forgetting: {pilot1.get('seeds_with_safety_specific_forgetting', 0)}
- Median A damage increase: {pilot1.get('median_a_damage_increase', 0.0)}
- Complete diffusion seeds: {len(diffusion)}/{len(seeds)}

## Pilot 2

- Verdict: {pilot2.get('verdict')}
- Complete matched comparisons: {len(comparisons)}/{len(seeds)}
- Per-seed decisions: {pilot2.get('per_seed', {})}

## Recommendation

{recommendation}
"""
    (results / "GO_NO_GO_REPORT.md").write_text(report)
    stage.succeed({"status": "COMPLETE", "overall_verdict": overall})


def main() -> None:
    parser = stage_parser("Aggregate all technical and scientific gates")
    arguments = parser.parse_args()
    run(arguments.project_root, arguments.run_root, parse_seeds(arguments.seeds))


if __name__ == "__main__":
    main()
