# Safe Continual RL Diffusion Pilots

This repository automates Pilot −1 and go/no-go Pilots 0–2 for safety forgetting in the OopsieBench RoboCasa `DamageableShelveItem` task. It replays and relabels the two official demonstration files through DamageSim, constructs fixed object-centric state datasets, trains matched action-chunk diffusion and Gaussian policies, performs task-only weight-level DPPO/PPO adaptation, and writes a local decision report. No episode labeling or gate decision requires human input.

## One command on Narval

```bash
# On a Narval login node:
cd /home/taegyoem/scratch/safe-continual-rl-diffusion
git pull --ff-only
bash run_pilots.sh
```

No user-defined environment variables are required for this checkout. The
launcher defaults to account `def-mijungp`, discovers the repository from its
own location, uses that same directory for run artifacts, and runs seeds
`0,1,2`. Every default remains overridable with the corresponding command-line
option.

A separate project root and run root are **not required**. Using this checkout
for both is supported: `.venv/`, `vendor/` checkouts, `data/`, `artifacts/`,
`logs/`, and `results/` are ignored by Git. Because this checkout lives on
Narval scratch, copy important final artifacts to persistent project storage
before the cluster's scratch-retention window expires.

The command creates or reuses a Python 3.10 virtual environment, resolves and records exact upstream source SHAs, installs the pinned OopsieVerse simulator stack and minimal DPPO dependencies, downloads only the two Shelve Item HDF5 files, runs preflight checks, and submits the complete dependency chain. It prints every job ID and returns; Slurm completes the pipeline without later commands or approvals.

Expected early feasibility artifact:

```text
/home/taegyoem/scratch/safe-continual-rl-diffusion/results/pilot_minus1/report.md
```

Expected final artifact:

```text
/home/taegyoem/scratch/safe-continual-rl-diffusion/results/GO_NO_GO_REPORT.md
```

To inspect submission commands without installing, downloading, or submitting:

```bash
bash run_pilots.sh --dry-run
```

Recovery is idempotent. Valid downloads and `_SUCCESS` stages are reused. A forced rerun archives affected result directories with a UTC timestamp rather than deleting them:

```bash
bash run_pilots.sh --force-from pilot-minus1
bash run_pilots.sh --force-from pilot0
bash run_pilots.sh --force-from pilots12
bash run_pilots.sh --force-from aggregate
```

CPU counts and walltimes can be overridden with `--pilot-minus1-cpus`, `--pilot0-cpus`, `--gpu-cpus`, `--aggregate-cpus`, and the corresponding `--*-walltime` options. Defaults live in `configs/narval.yaml`.

## Pipeline and gates

1. **Pilot −1 (CPU):** stable-hash selection, official `states[:-1]` / `actions[1:]` playback alignment, automated DamageSim relabeling, miniature DPPO NPZ conversion, real loaders, 20 BC steps per actor, short simulator rollouts, weight-level PPO/DPPO updates, and deterministic checkpoint reloads. A failure writes `STOP_PIPELINE.json` and prevents downstream compute.
2. **Pilot 0 (CPU, `afterok`):** complete 90-episode replay, no-op damage-threshold calibration, critical fragile-object selection, deterministic Context B construction, 100-reset validity checks, frozen A-safe per-seed datasets, and the exact green/yellow/red gate.
3. **Pilots 1–2 (A100 array, `afterok`):** three seed tasks. Each first runs the production 200-step-per-policy model/environment smoke, then trains/evaluates diffusion and the matched Gaussian comparator sequentially. Seed 0 declares the adaptive-extension decision; other array tasks wait for and apply it without racing. GPU jobs read Pilot 0's gate before importing Torch or initializing CUDA.
4. **Video postprocessing (seed 0 only):** after training, deterministically selects at most five A-pre, B-post, and A-post rollouts per policy and renders them separately. Rendering errors are diagnostic and never change metrics or gates.
5. **Aggregation (CPU, `afterany`):** always writes a technical failure, scientific stop, pivot, or go report—even when dependencies are canceled after an early failure.

Task reward is never modified by damage. Health, damage, source label, and context ID are excluded from actor/critic observations and optimization. Context is inferable only from fixed-slot object positions. A-safe training statistics are frozen for all B adaptation and later A evaluation.

## Reproducibility

`artifacts/source_versions.json` records the resolved commits for [OopsieVerse](https://github.com/UT-Austin-RobIn/oopsieverse), [DPPO](https://github.com/irom-princeton/dppo), the pinned RoboCasa/RoboSuite stack, and the inspected DSRL and RoboCasa Diffusion Policy references, plus the demonstration snapshot and hashes, Python/CUDA/package versions, `pip freeze`, hostname, and Slurm metadata. The local DPPO adapter records its exact upstream PPO-objective and RoboMimic-config provenance; existing checkouts are never silently advanced. A complete resolved environment is also copied to `artifacts/requirements.lock.txt` for each run.

The suite is intentionally limited to the four pilots in the specification. It does not implement safe denoising, unsafe memory, EWC, constrained PPO, cost critics, RGB policies, or extra tasks.

## Local checks

The simulator-dependent preflight runs automatically during the Narval command. Dependency-light unit tests cover the pure data, wrapper, policy-likelihood, and gate contracts:

```bash
pytest -q
ruff check .
python -m compileall src
```

`--local-smoke` is an explicit non-Narval mode for machines that already support the pinned RoboCasa simulator. It runs setup, static tests, preflight, Pilot −1, and Pilot 0 locally; it does not launch the A100 scientific stages.
