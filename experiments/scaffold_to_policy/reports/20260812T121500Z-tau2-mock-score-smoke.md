# tau2 Mock Score Smoke

Run ID: `20260812T121500Z-tau2-mock-score-smoke`

This smoke proves that TorchTitan can invoke Sierra tau2-bench's own mock-domain
task definitions and reward computation inside the bwrap rootfs. It is not a
model or agent benchmark result: the trajectory is a deterministic fixture, and
no tau2 agent, user simulator, Harbor agent, Terminal-Bench task, or external
LLM provider was run.

## Command

```bash
RUN_ID=20260812T121500Z-tau2-mock-score-smoke \
RESULTS_ROOT=experiments/scaffold_to_policy/results/tau2_mock_score_smoke \
experiments/scaffold_to_policy/run_tau2_mock_score_smoke.sh
```

The script creates an isolated rootfs virtualenv, installs tau2-bench from the
pinned Sierra repo revision, clones the same pinned repo under the ignored
results tree for its `data/` directory, constructs a valid mock-domain
`create_task_1` trajectory, scores it with tau2's evaluator, ingests the result,
and writes a report input.

## Result

| Field | Value |
| --- | --- |
| Package | `tau2==1.0.1` |
| Repo revision | `668d3bcd135c02aa3438f987ef45735b7c163ee3` |
| Domain | `mock` |
| Task | `create_task_1` |
| Evaluation type | `all_ignore_basis` |
| Termination reason | `agent_stop` |
| Score | `1.0` |
| DB reward | `1.0` |
| Action reward | `1.0` |
| Communication reward | `1.0` |

Report-input checks all passed:

- `ingested_present=true`
- `all_rootfs_selected=true`
- `all_modes_labeled=true`
- `all_pins_present=true`
- `task_score_smokes_succeeded=true`

## Example

The fixture trajectory requests a mock task creation, calls the upstream
`create_task` tool, records the tool result, and includes a final assistant
confirmation:

```json
[
  {"actor": "fixture_user", "event": "request_create_task"},
  {"actor": "fixture_agent", "event": "call_create_task"},
  {"actor": "tau2_mock_environment", "event": "return_tool_result"},
  {"actor": "tau2_evaluator", "event": "compute_reward"}
]
```

The ingested metric records the upstream scorer output:

```json
{
  "name": "tau2_mock_score",
  "score": 1.0,
  "score_source": "tau2 evaluator all_ignore_basis",
  "task_metadata": {
    "domain": "mock",
    "task_id": "create_task_1",
    "reward_breakdown": {
      "DB": 1.0,
      "ACTION": 1.0,
      "COMMUNICATE": 1.0
    }
  }
}
```

## Scope

This clears the tau2 scorer-ingestion gate that was missing after package
preflight. It does not clear the full tau2 agent-execution gate because the
tested no-external-LLM `dummy_user` CLI pairings still end as infrastructure
errors. The next tau2 step is to wire either a local model provider or a
supported deterministic agent/user pairing that produces a complete tau2
simulation with non-fixture messages.
