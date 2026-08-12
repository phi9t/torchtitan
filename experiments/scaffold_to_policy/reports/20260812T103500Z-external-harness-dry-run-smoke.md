# External Harness Dry-Run Smoke

Date: 2026-08-12

## Scope

This report records the first external-harness feasibility smoke for the
scaffold-to-policy program. It covers the two planned agentic harness families:

- Harbor plus Terminal-Bench;
- tau2-bench.

This is compatibility and artifact-ingestion evidence only. It is not a
Terminal-Bench, Harbor, tau2-bench, agentic, model, adapter, or benchmark score.
The actual external harness packages were not installed in the current rootfs,
so the smoke uses dry-run harness-owned score fixtures to validate pins,
trajectory ingestion, environment capture, and shared report-input rendering.

## Command

The run used the bwrap rootfs:

```bash
RUN_ID=20260812T103500Z-external-harness-dry-run-smoke \
RESULTS_ROOT=experiments/scaffold_to_policy/results/external_harness_dry_run_smoke \
experiments/scaffold_to_policy/run_external_harness_dry_run_smoke.sh
```

The entrypoint re-executes itself through:

```bash
scripts/rootfs/enter_rootfs.sh -- experiments/scaffold_to_policy/run_external_harness_dry_run_smoke.sh
```

## Pins

The smoke records explicit upstream pins:

| Harness | Repository | Revision | Installed in rootfs |
| --- | --- | --- | --- |
| Harbor | `https://github.com/harbor-framework/harbor.git` | `b7e2f71b4563618af3a42279740f5f412dcf7046` | false |
| Terminal-Bench 2.1 | `https://github.com/harbor-framework/terminal-bench-2-1.git` | `7131e4375048a0e408a8fb404b5f499d726b695b` | false |
| tau2-bench | `https://github.com/sierra-research/tau2-bench.git` | `668d3bcd135c02aa3438f987ef45735b7c163ee3` | false |

The rootfs import probe confirms these modules are absent:

```text
harbor=False
terminal_bench=False
tau2=False
```

## Artifacts

Generated artifacts are ignored by git and live under:

```text
experiments/scaffold_to_policy/results/external_harness_dry_run_smoke/
```

Important artifacts:

```text
experiments/scaffold_to_policy/results/external_harness_dry_run_smoke/raw/harbor_terminal.json
experiments/scaffold_to_policy/results/external_harness_dry_run_smoke/raw/tau2.json
experiments/scaffold_to_policy/results/external_harness_dry_run_smoke/ingested/harbor_terminal.json
experiments/scaffold_to_policy/results/external_harness_dry_run_smoke/ingested/tau2.json
experiments/scaffold_to_policy/results/external_harness_dry_run_smoke/manifests/report_input_20260812T103500Z-external-harness-dry-run-smoke.json
```

Report-input checks:

| Check | Result |
| --- | --- |
| `ingested_present` | true |
| `all_rootfs_selected` | true |
| `all_dry_run_labeled` | true |
| `all_pins_present` | true |

## Runtime Metadata

The smoke records rootfs execution metadata:

| Field | Value |
| --- | --- |
| `TORCHTITAN_IN_ROOTFS` | true |
| Python | `/usr/bin/python` |
| Python version | `3.12.3` |
| Platform | `Linux-5.15.152.bsk.9-amd64-x86_64-with-glibc2.39` |
| Git | `git version 2.43.0` |
| Docker inside rootfs | not found |
| bwrap inside rootfs | not found |

The absence of Docker and bwrap binaries inside the rootfs is not a failure for
this dry-run ingestion smoke, but it is a blocker to record before any real
Terminal-Bench/Harbor container execution claim.

## Ingested Results

| Harness family | Mode | Task subset | Metric | Score | Score source |
| --- | --- | --- | --- | ---: | --- |
| Harbor/Terminal-Bench | `dry_run` | `terminal-bench-dry-run` | `dry_run_compatibility` | 1.0 | synthetic dry-run fixture |
| tau2-bench | `dry_run` | `tau2-dry-run` | `dry_run_compatibility` | 1.0 | synthetic dry-run fixture |

The dry-run score means only that the repo-local ingestion contract can carry a
harness-owned score-like object through raw artifacts, ingested artifacts, and a
shared report input. It does not mean either upstream benchmark ran.

## Trajectory Shape

Each dry-run artifact records a minimal trajectory:

```text
step 0: torchtitan declares the external harness boundary
step 1: external_harness emits a dry-run score fixture for the selected subset
```

This validates that later real harness outputs can carry trajectories or event
logs without changing the report-input shape.

## Interpretation

This clears the dry-run external-harness ingestion gate:

- rootfs-managed launch works from one repo-local command;
- Harbor, Terminal-Bench, and tau2-bench pins are recorded;
- installed-package state is recorded instead of assumed;
- raw score and trajectory artifacts are ingested into a shared report input;
- the report input explicitly prevents capability overclaiming.

The next external-harness step is not another dry run. It is a real pinned
install/run smoke for one tiny Harbor/Terminal-Bench task or tau2 task, after
the rootfs has the required runtime tools and package dependencies. For
Terminal-Bench/Harbor specifically, container runtime availability must be
resolved or the result must be labeled as a harness import/preflight only.
