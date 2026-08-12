# Hard Benchmark Continuation

Run timestamp: 2026-08-12T11:15:00Z

This report records the follow-up execution after the earlier BigCodeBench,
ARC, tau2, and Harbor checkpoint. It is conservative: blocked probes and
harness smokes are not reported as model capability results.

## Summary

Completed:

- Terminal-Bench/Harbor oracle execution is now working through the bwrap
  rootfs plus explicit host Docker passthrough and host-path repo binding.
- tau2 execution probing moved past the original dummy-user constructor
  mismatch to a clearer offline-provider blocker.
- `GPU_MEMORY_UTILIZATION` is now forwarded by the shared vLLM evaluators used
  by arithmetic, modular sequences, GSM-style, MATH/AIME-style, coding, and
  multiple-choice tasks.
- `preflight-vllm-gpu-memory` now writes structured JSON even when CUDA memory
  discovery fails under extreme GPU contention.

Still blocked:

- GPQA Diamond import is gated by Hugging Face auth in the rootfs.
- BigCodeBench-Hard `contract_chat` did not produce a model result because
  unrelated SGLang processes occupied nearly all eight B200 GPUs.
- tau2 nonfixture execution needs a real compatible provider endpoint or a
  benchmark-preserving deterministic provider/agent path.

## BigCodeBench-Hard

The completed baseline calibration remains:

- Run ID: `20260812T173000Z-bigcodebench-hard-public-vllm-expanded`
- Slice: 8 dev tasks from offset 0, 8 OOD tasks from offset 72
- Rollouts: 4 per problem
- Canonical preflight: dev 8/8, OOD 8/8
- Qwen3-1.7B result: dev/OOD pass@1 and pass@4 all `0.000`
- Failure mode: all 64 sampled candidates failed released unit tests with
  assertion failures.

The follow-up `contract_chat` condition keeps the no-tool executable-test
semantics. It changes only the prompt contract: preserve the requested function
signature and match returns, side effects, error behavior, and library calls.

New blocked probes:

- `20260812T103500Z-bigcodebench-hard-contract-chat-expanded-lowmem`
- `20260812T105000Z-bigcodebench-hard-contract-canonical-blocked`
- `20260812T110000Z-bigcodebench-hard-contract-min-lowmem`

The same selected contract slice passed canonical preflight again:

```json
{
  "dev": {"selected": true, "failure_breakdown": {"success": 8}},
  "ood_test": {"selected": true, "failure_breakdown": {"success": 8}}
}
```

The model evaluation did not complete. The machine was occupied by unrelated
root-owned SGLang scheduler processes using roughly 178-182 GiB per B200. In
the most contended state, even CUDA memory discovery failed:

```json
{
  "kind": "vllm_gpu_memory_preflight",
  "selected": false,
  "reason": "cuda memory query failed",
  "gpu_memory_utilization": 0.35,
  "devices": [
    {
      "device_index": 0,
      "error_type": "AcceleratorError",
      "selected": false
    }
  ]
}
```

This is a resource blocker, not a coding-benchmark result.

## GPQA Diamond

Probe:

- Run ID: `20260812T104500Z-gpqa-public-import-gpu-blocker`
- Dataset: `Idavidrein/gpqa`, subset `gpqa_diamond`
- Result: no benchmark execution or model score.

The script stopped at its gated-dataset blocker path:

```text
Dataset 'Idavidrein/gpqa' is a gated dataset on the Hub.
You must be authenticated to access it.
```

Next step: rerun with a valid `HF_TOKEN` available inside the bwrap rootfs,
while preserving the no-tool multiple-choice condition.

## tau2

The first tau2 execution probe reached upstream `tau2 run` but failed before
evaluation:

```text
DummyUser.__init__() got an unexpected keyword argument 'tools'
```

The runner default now uses tau2's constructor-compatible non-solo pairing:

```text
TAU2_AGENT=llm_agent
TAU2_USER=user_simulator
TAU2_AGENT_LLM=fake
TAU2_USER_LLM=fake
```

That path reaches the simulation loop, but the pinned tau2 revision routes
model calls through LiteLLM and does not provide a built-in `fake` provider:

```text
litellm.BadRequestError: LLM Provider NOT provided.
You passed model=fake
```

The raw ingested metric correctly reports `score=0.0`, `num_tasks=0`,
`num_simulations=1`, `num_infra_errors=1`, and `num_evaluated=0`. This remains
blocker evidence only.

## Harbor / Terminal-Bench

The current successful Harbor infrastructure probe is:

- Run ID: `20260812T190000Z-terminal-bench-harbor-hostpath`
- Task: `headless-terminal`
- Harness: Harbor with pinned Terminal-Bench 2.1 task repo
- Result: `n_trials=1`, `n_errors=0`, mean metric `1.0`, verifier reward `1.0`

This clears Docker/Compose/verifier mount wiring for the oracle path. It is not
a model or agent capability result. The next step is replacing the oracle with
a bounded baseline or model agent while preserving Harbor's released verifier
semantics.

## Infra Changes

- `run_bigcodebench_hard_public_vllm_smoke.sh` now passes
  `GPU_MEMORY_UTILIZATION` into `evaluate-coding-style-vllm`, not just into the
  preflight.
- `run_aime_public_vllm_smoke.sh` and `run_gpqa_public_vllm_smoke.sh` now run
  `preflight-vllm-gpu-memory` before vLLM launch and pass the same utilization
  setting into the evaluator.
- `evaluate-arithmetic-vllm`, `evaluate-modular-vllm`,
  `evaluate-gsm-style-vllm`, `evaluate-math-style-vllm`,
  `evaluate-coding-style-vllm`, and `evaluate-multiple-choice-vllm` now accept
  `--gpu-memory-utilization`.
- `preflight-vllm-gpu-memory` catches CUDA memory-query exceptions and records
  them as structured preflight failures.
- `run_tau2_execution_probe.sh` exposes `TAU2_USER_LLM` and defaults to the
  constructor-compatible tau2 non-solo agent/user pair.

## Next Steps

1. Rerun BigCodeBench-Hard `contract_chat` when the B200s are actually free.
2. Rerun GPQA with `HF_TOKEN` in the rootfs environment.
3. Add a tau2-compatible local provider or deterministic agent path that
   produces a real upstream `tau2 run` trajectory and official tau2 reward.
4. Replace Harbor oracle execution with a bounded non-oracle baseline agent.
