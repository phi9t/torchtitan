# Terminal-Bench Harbor Nop Baseline

Run ID: `20260812T111500Z-terminal-bench-harbor-nop-baseline`

This run replaces the previous Harbor oracle-only probe with a bounded
non-oracle baseline while preserving the same pinned Terminal-Bench task,
Harbor runner, Docker environment, and verifier semantics.

## Command

```bash
RUN_ID=20260812T111500Z-terminal-bench-harbor-nop-baseline \
RESULTS_ROOT=experiments/scaffold_to_policy/results/terminal_bench_harbor_nop_baseline \
HARBOR_AGENT=nop \
TIMEOUT_SECONDS=240 \
RECREATE_VENV=0 \
experiments/scaffold_to_policy/run_terminal_bench_oracle_probe.sh
```

The runner still re-execs through `scripts/rootfs/enter_rootfs.sh` with
`TORCHTITAN_ROOTFS_BIND_DOCKER=1`, runs Harbor from the host-visible checkout
path, and lets Harbor own task execution and scoring.

## Result

Raw scaffold summary:

```json
{
  "agent_name": "nop",
  "returncode": 0,
  "execution_completed": true,
  "num_trials": 1,
  "num_errors": 0,
  "num_trial_exceptions": 0,
  "mean_metric": 0.0,
  "score": 0.0,
  "num_tasks": 1
}
```

Harbor's own job result recorded:

| Field | Value |
| --- | ---: |
| `n_total_trials` | 1 |
| `stats.n_completed_trials` | 1 |
| `stats.n_errored_trials` | 0 |
| `stats.evals.nop__adhoc.n_trials` | 1 |
| `stats.evals.nop__adhoc.n_errors` | 0 |
| `stats.evals.nop__adhoc.metrics[0].mean` | 0.0 |

The scaffold report input records:

```json
{
  "task_execution_probes_completed": true,
  "task_execution_probes_succeeded": false,
  "all_rootfs_selected": true
}
```

## Interpretation

This clears a stronger Terminal-Bench/Harbor baseline than the oracle probe:
the external harness ran a non-oracle agent through the released task and
verifier, completed one trial, and produced the expected zero reward. It does
not evaluate Qwen3, a TorchTitan adapter, or a learned policy. The next
Terminal-Bench step is a small model or scaffold-policy agent run with the same
labels and verifier path.
