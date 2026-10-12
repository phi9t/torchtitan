# tau2 Noop Baseline

Run ID: `20260812T113000Z-tau2-noop-baseline`

This run replaces the tau2 oracle-style deterministic probe with a bounded
non-oracle baseline while preserving Sierra tau2-bench's pinned task loading,
runner, trajectory writer, and official evaluator.

## Command

```bash
RUN_ID=20260812T113000Z-tau2-noop-baseline \
RESULTS_ROOT=experiments/scaffold_to_policy/results/tau2_noop_baseline \
TAU2_AGENT=torchtitan_noop_agent \
TAU2_USER=torchtitan_static_user \
TAU2_MAX_STEPS=4 \
RECREATE_VENV=1 \
experiments/scaffold_to_policy/run_tau2_execution_probe.sh
```

The runner re-executes through the bwrap rootfs, creates an isolated tau2
virtualenv, installs Sierra tau2-bench from revision
`668d3bcd135c02aa3438f987ef45735b7c163ee3`, clones the same revision for its
`data/` directory, and dispatches to upstream `tau2.cli.main` through the local
registration wrapper.

## Registered Components

- `torchtitan_noop_agent`: a non-solo half-duplex tau2 agent that immediately
  emits tau2's stop token without tool calls.
- `torchtitan_static_user`: a static user that sends the task scenario once,
  then stops if asked again.

## Result

Raw scaffold summary:

```json
{
  "score": 0.0,
  "num_tasks": 1,
  "returncode": 0,
  "execution_completed": true,
  "num_simulations": 1,
  "num_evaluated": 1,
  "num_infra_errors": 0,
  "average_reward": 0.0,
  "termination_reasons": {
    "agent_stop": 1
  },
  "errors": []
}
```

The official tau2 reward breakdown for `create_task_1` was:

```json
{
  "reward": 0.0,
  "reward_breakdown": {
    "DB": 0.0,
    "COMMUNICATE": 1.0
  }
}
```

The scaffold report input records:

```json
{
  "task_execution_probes_completed": true,
  "task_execution_probes_succeeded": false,
  "all_rootfs_selected": true
}
```

## Interpretation

This clears a non-oracle tau2 baseline: the official tau2 runner produced a
trajectory, the evaluator scored it, and no infrastructure errors occurred.
The zero reward is expected because the noop agent does not perform the required
`create_task` write action. This does not evaluate Qwen3, a TorchTitan adapter,
or a learned scaffold policy.
