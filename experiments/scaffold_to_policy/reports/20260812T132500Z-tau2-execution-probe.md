# tau2 Execution Probe

Run ID: `20260812T132500Z-tau2-execution-probe`

This probe advances tau2-bench from fixture scorer ingestion to an upstream CLI
execution attempt inside the TorchTitan bwrap rootfs. It is blocker evidence,
not a tau2 benchmark result: tau2 wrote a real `results.json`, but no task was
evaluated.

## Command

```bash
RUN_ID=20260812T132500Z-tau2-execution-probe \
RESULTS_ROOT=experiments/scaffold_to_policy/results/tau2_execution_probe_20260812T132500Z \
experiments/scaffold_to_policy/run_tau2_execution_probe.sh
```

The script re-entered through `scripts/rootfs/enter_rootfs.sh`, created an
isolated virtualenv under the ignored results tree, installed Sierra
tau2-bench from the pinned git revision, cloned the same pinned repo for its
`data/` directory, ran `tau2 check-data`, launched one bounded mock-domain task
through upstream `tau2 run`, and ingested the saved tau2 result.

## Pinned Evaluation Object

| Field | Value |
| --- | --- |
| Benchmark repo | `https://github.com/sierra-research/tau2-bench.git` |
| Revision | `668d3bcd135c02aa3438f987ef45735b7c163ee3` |
| Package | `tau2==1.0.1` from the pinned repo |
| Domain | `mock` |
| Task set | `mock` |
| Task | `create_task_1` |
| Agent | `llm_agent_solo` |
| Agent model label | `fake` |
| User | `dummy_user` |
| Max steps | `2` |
| Max errors | `1` |
| tau2 task timeout | `20` seconds |

## Result

| Field | Value |
| --- | --- |
| Rootfs selected | `true` |
| CLI return code | `0` |
| Saved results present | `true` |
| Simulations | `1` |
| Evaluated tasks | `0` |
| Infrastructure errors | `1` |
| Probe score | `0.0` |
| `task_execution_probes_succeeded` | `false` |

The saved tau2 result is:

```text
experiments/scaffold_to_policy/results/tau2_execution_probe_20260812T132500Z/src/tau2-bench/data/simulations/20260812T132500Z-tau2-execution-probe/results.json
```

The raw and ingested TorchTitan artifacts are:

```text
experiments/scaffold_to_policy/results/tau2_execution_probe_20260812T132500Z/raw/tau2_execution_probe.json
experiments/scaffold_to_policy/results/tau2_execution_probe_20260812T132500Z/ingested/tau2_execution_probe.json
experiments/scaffold_to_policy/results/tau2_execution_probe_20260812T132500Z/manifests/report_input_20260812T132500Z-tau2-execution-probe.json
```

## Example Failure

The upstream tau2 result records:

```json
{
  "task_id": "create_task_1",
  "termination_reason": "infrastructure_error",
  "reward_info": null,
  "info": {
    "error_type": "TypeError",
    "error": "DummyUser.__init__() got an unexpected keyword argument 'tools'",
    "failed_after_attempts": 1
  }
}
```

The captured tau2 stdout summary agrees with the parsed result:

```text
Total Simulations         1
Infra Errors              1 (excluded from metrics below)
Evaluated                 0
Total Tasks               0
Average Reward            0.0000
```

## Interpretation

This clears a narrower gate than full tau2 execution: TorchTitan can now launch
upstream `tau2 run` hermetically, preserve tau2's own saved result file, ingest
its termination summary, and surface an explicit failed execution probe in the
shared external-harness report contract.

It does not clear the tau2 benchmark gate. The next step is to choose a
supported deterministic agent/user pairing or wire a local model provider so
that at least one mock-domain tau2 task reaches a non-infrastructure
termination and receives tau2 evaluator rewards.
