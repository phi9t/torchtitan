# tau2 Qwen Model-Policy Probe

Run ID: `20260812Ttau2-qwen-model-policy`

This run replaces the deterministic tau2 probe agent with a bounded Qwen3-1.7B
model-policy agent while preserving the upstream tau2 task runner and saved
`results.json` scoring artifact. It is a model-policy execution artifact for
one pinned tau2 mock-domain task, not a successful tau2 benchmark result.

## Command

```bash
RUN_ID=20260812Ttau2-qwen-model-policy \
RESULTS_ROOT=experiments/scaffold_to_policy/results/tau2_qwen_model_policy \
TAU2_AGENT=torchtitan_qwen_agent \
TAU2_AGENT_LLM=fake \
TAU2_USER=torchtitan_static_user \
TAU2_USER_LLM=fake \
TAU2_MAX_STEPS=3 \
PROBE_TIMEOUT_SECONDS=300 \
SCAFFOLD_TO_POLICY_TAU2_GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_tau2_execution_probe.sh
```

The tau2 harness ran inside the bwrap rootfs in an isolated tau2 virtualenv.
The registered `torchtitan_qwen_agent` generated each action by spawning rootfs
`/usr/bin/python` with local Qwen3-1.7B, vLLM, Triton attention, and
`VLLM_USE_FLASHINFER_SAMPLER=0`.

## Pins

| Object | Value |
| --- | --- |
| tau2-bench repo | `https://github.com/sierra-research/tau2-bench.git` |
| tau2-bench revision | `668d3bcd135c02aa3438f987ef45735b7c163ee3` |
| tau2 package | `tau2==1.0.1` |
| Model | `./assets/hf/Qwen3-1.7B` |
| Task | `mock/create_task_1` |
| User | `torchtitan_static_user` |
| Agent | `torchtitan_qwen_agent` |

## Upstream Result

The raw and ingested artifacts are:

```text
experiments/scaffold_to_policy/results/tau2_qwen_model_policy/raw/tau2_execution_probe.json
experiments/scaffold_to_policy/results/tau2_qwen_model_policy/ingested/tau2_execution_probe.json
experiments/scaffold_to_policy/results/tau2_qwen_model_policy/manifests/report_input_20260812Ttau2-qwen-model-policy.json
experiments/scaffold_to_policy/results/tau2_qwen_model_policy/src/tau2-bench/data/simulations/20260812Ttau2-qwen-model-policy/results.json
```

Key parsed fields:

| Field | Value |
| --- | --- |
| `returncode` | `0` |
| `num_simulations` | `1` |
| `num_evaluated` | `1` |
| `num_infra_errors` | `0` |
| `average_reward` | `0.0` |
| `termination_reasons` | `{"max_steps": 1}` |
| `task_execution_probes_completed` | `true` |
| `task_execution_probes_succeeded` | `false` |

The upstream tau2 simulation did reach the task environment and execute the
model-chosen tool call:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "torchtitan_qwen_call_1",
      "name": "create_task",
      "arguments": {
        "title": "Important Meeting",
        "user_id": "user_1"
      },
      "requestor": "assistant"
    }
  ]
}
```

The tool returned:

```json
{
  "task_id": "task_2",
  "title": "Important Meeting",
  "description": null,
  "status": "pending"
}
```

tau2 still assigned reward `0.0` because the bounded three-step simulation
ended at `max_steps` before the agent produced the required user-facing
confirmation. This is a policy/harness behavior issue, not an import, Docker,
rootfs, vLLM, tokenizer, or tau2 scorer failure.

## Interpretation

This run clears a narrower but important gate:

- TorchTitan can register a non-oracle tau2 agent in the pinned tau2 virtualenv.
- The agent can call local Qwen3-1.7B/vLLM hermetically from inside the rootfs.
- Upstream `tau2 run` can execute the model-produced tool call and write its
  official `results.json`.
- The external-harness report input separates clean execution from task reward.

It does not show tau2 task success. The next tau2 branch should keep the same
upstream runner and task but improve turn handling so the model sees the tool
result and emits a final confirmation before `max_steps`, or increase the
bounded step budget while reporting that budget explicitly.
