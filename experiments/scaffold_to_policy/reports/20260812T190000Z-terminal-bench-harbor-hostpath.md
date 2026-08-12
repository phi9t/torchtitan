# Terminal-Bench Harbor Host-Path Probe

Run ID: `20260812T190000Z-terminal-bench-harbor-hostpath`

This reruns the one-task Terminal-Bench / Harbor oracle probe after fixing the
rootfs/Docker path mismatch. It is an external-harness infrastructure result,
not a Terminal-Bench model capability result.

## Command

```bash
RUN_ID=20260812T190000Z-terminal-bench-harbor-hostpath \
RESULTS_ROOT=experiments/scaffold_to_policy/results/terminal_bench_harbor_hostpath \
TIMEOUT_SECONDS=240 \
experiments/scaffold_to_policy/run_terminal_bench_oracle_probe.sh
```

## Fix

The previous Docker/Compose probe failed because Harbor generated bind mount
sources under `/workspace/torchtitan/...`, which exists inside bwrap but not on
the host where the Docker daemon resolves bind sources.

When `TORCHTITAN_ROOTFS_BIND_DOCKER=1` is set, `scripts/rootfs/enter_rootfs.sh`
now also binds this checkout at its real host path and exports
`TORCHTITAN_ROOTFS_HOST_REPO_ROOT`. The Terminal-Bench runner uses that host path
as the Harbor subprocess working directory, so Docker receives host-visible bind
source paths while Python execution still occurs inside the rootfs.

## Result

Harbor's own job result recorded a successful one-trial oracle execution:

| Field | Value |
| --- | ---: |
| `n_total_trials` | 1 |
| `stats.n_completed_trials` | 1 |
| `stats.n_errored_trials` | 0 |
| `stats.evals.oracle__adhoc.n_trials` | 1 |
| `stats.evals.oracle__adhoc.n_errors` | 0 |
| `stats.evals.oracle__adhoc.metrics[0].mean` | 1.0 |

The trial result for `headless-terminal__6MShnbm` has
`verifier_result.rewards.reward=1.0` and `exception_info=null`.

The scaffold report input now marks:

```text
task_execution_probes_succeeded=true
task_score_smokes_succeeded=true
all_rootfs_selected=true
```

## Interpretation

This clears the Harbor/Docker/Compose/verifier-mount infrastructure blocker for
the pinned `headless-terminal` oracle probe. It does not evaluate Qwen3, a
TorchTitan adapter, or any agent harness policy. The next Terminal-Bench step is
to replace the oracle with a bounded model or deterministic baseline agent while
preserving Harbor's released task execution and verifier semantics.
