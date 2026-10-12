# Public Benchmark Runnable Coverage

Date: 2026-08-14
Host: 8x NVIDIA B200, bwrap rootfs, `assets/hf/Qwen3-1.7B` present.

This note records the current scaffold-to-policy benchmark coverage after the
typed lifecycle migration pass. "Runnable" means the entrypoint re-enters the
TorchTitan bwrap rootfs, runs the relevant preflight/doctor checks, and exits
with a typed run-attempt outcome: either real measurements or an explicit
blocked result with `not_run` split conditions.

## Direct Public vLLM Runners

These runners execute Qwen3-1.7B through vLLM inside the rootfs and write typed
begin/stage/finish attempt bundles.

| Benchmark runner | Current status | Evidence run |
| --- | --- | --- |
| GSM8K public smoke | runnable, typed real path from earlier migration | `run_gsm8k_public_vllm_smoke.sh` |
| HumanEval public smoke | runnable, typed real path from earlier migration | `run_humaneval_public_vllm_smoke.sh` |
| MATH public smoke | runnable, typed real path from earlier migration | `run_math_public_vllm_smoke.sh` |
| MBPP public smoke | runnable, typed real path from earlier migration | `run_mbpp_public_vllm_smoke.sh` |
| AIME public smoke | runnable, typed real and forced-blocked paths | `20260814T042000Z-aime-typed-default` |
| MMLU-Pro public smoke | runnable, typed real and forced-blocked paths | `20260814T044500Z-mmlu-pro-typed-default` |
| GPQA public smoke | runnable, typed blocked path on this host | `20260814T050500Z-gpqa-typed-default` |
| LiveCodeBench public smoke | runnable, typed real and forced-blocked paths | `20260814T052500Z-livecodebench-typed-default` |
| ARC-AGI-2 public smoke | runnable, typed real and forced-blocked paths | `20260814T_arc_typed_default` |
| BigCodeBench-Hard public smoke | runnable, typed real and forced-blocked paths | `20260814T_bigcodebench_typed_default` |

GPQA is runnable in the infrastructure sense but did not produce model
measurements on this host because the public GPQA Diamond source is gated and
no rootfs-visible token or raw cache was available. The runner now finishes that
case as `attempt_outcome=blocked` instead of failing ambiguously.

## Newly Verified Runners

### ARC-AGI-2

Entrypoint:

```bash
RUN_ID=20260814T_arc_typed_default \
experiments/scaffold_to_policy/run_arc_agi2_public_vllm_smoke.sh
```

Outcome:

```json
{
  "execution_outcome": "completed",
  "run_gate": {
    "has_real_measurement": true,
    "measurement_counts": {"real": 2},
    "num_conditions": 2,
    "promotion_counts": {"not_evaluated": 2}
  }
}
```

The smoke used two dev and two OOD tasks with two rollouts each. Dev had
pass@2 of `0.5`; OOD had pass@2 of `0.0`. The main observed failure was output
contract compliance: unparsable final grid JSON or a final grid that did not
match the target answer.

Forced blocker:

```bash
RUN_ID=20260814T_arc_blocked_typed \
GPU_MEMORY_UTILIZATION=2.0 \
experiments/scaffold_to_policy/run_arc_agi2_public_vllm_smoke.sh
```

That run stopped at the vLLM GPU-memory selection gate and wrote:

```json
{
  "execution_outcome": "blocked",
  "run_gate": {
    "has_real_measurement": false,
    "measurement_counts": {"not_run": 2},
    "num_conditions": 2,
    "promotion_counts": {"not_evaluated": 2}
  }
}
```

### BigCodeBench-Hard

Entrypoint:

```bash
RUN_ID=20260814T_bigcodebench_typed_default \
experiments/scaffold_to_policy/run_bigcodebench_hard_public_vllm_smoke.sh
```

Outcome:

```json
{
  "execution_outcome": "completed",
  "run_gate": {
    "has_real_measurement": true,
    "measurement_counts": {"real": 2},
    "num_conditions": 2,
    "promotion_counts": {"not_evaluated": 2}
  }
}
```

The smoke used two dev and two OOD tasks with two rollouts each. Both splits
had pass@2 of `0.0`; all four generated samples per split failed with
assertion failures. That is a valid hard-negative measurement, not an
infrastructure blocker.

Forced blocker:

```bash
RUN_ID=20260814T_bigcodebench_blocked_typed \
GPU_MEMORY_UTILIZATION=2.0 \
experiments/scaffold_to_policy/run_bigcodebench_hard_public_vllm_smoke.sh
```

That run passed dependency install, public dataset import, canonical preflight,
and then stopped at the vLLM GPU-memory selection gate with two `not_run`
conditions.

## External Harness Coverage

External agentic harnesses are covered as compatibility and ingestion probes,
not as model capability benchmarks.

| Harness area | Current status | Evidence |
| --- | --- | --- |
| Harbor + Terminal-Bench package preflight | package install, import/version, and CLI checks pass in isolated rootfs venv | `20260812T112500Z-external-harness-preflight-corrected` |
| Harbor + Terminal-Bench task execution | one Harbor oracle probe completed with reward `1.0` | `20260812T190000Z-terminal-bench-harbor-hostpath` |
| tau2 package preflight | Sierra tau2-bench package install and import pass in isolated rootfs venv | `20260812T112500Z-external-harness-preflight-corrected` |
| tau2 scorer ingestion | deterministic mock-domain fixture scores `1.0` through tau2 evaluator | `20260812T121500Z-tau2-mock-score-smoke` |

These do not yet count as real Terminal-Bench or tau2 model-agent evaluations.
The next gate is replacing oracle/fixture probes with a bounded model or
deterministic baseline agent while preserving the upstream harness scorers.

## Runtime Contract Now Exercised

The migrated runners all use the same contract:

- host entrypoint immediately re-enters through `scripts/rootfs/enter_rootfs.sh`;
- typed `execution.preflight` runs before benchmark work;
- `begin` records benchmark identity and scientific knobs;
- every substantive action runs as a typed stage with rootfs provenance;
- successful runs finish with per-split `completed/real/not_evaluated`;
- environment, access, GPU, and runtime blockers finish with
  `blocked/not_run/not_evaluated`;
- report inputs attach the execution preflight where the benchmark report
  builder supports completed scientific measurements.

## Verification

```bash
bash -n experiments/scaffold_to_policy/run_*.sh

python -m pytest \
  tests/unit_tests/test_scaffold_to_policy.py \
  tests/unit_tests/test_scaffold_execution_foundation.py \
  tests/unit_tests/test_execution_lifecycle_run_attempt.py \
  tests/unit_tests/test_execution_lifecycle_done_scenarios.py \
  -q
```

Result: `148 passed, 1 skipped`.

Outcome scan:

- AIME default: `completed`, two real measurements.
- AIME forced blocker: `blocked`, two `not_run` conditions.
- MMLU-Pro default: `completed`, two real measurements.
- MMLU-Pro forced blocker: `blocked`, two `not_run` conditions.
- GPQA default: `blocked`, two `not_run` conditions due to gated import.
- LiveCodeBench default: `completed`, two real measurements.
- LiveCodeBench forced blocker: `blocked`, two `not_run` conditions.
- ARC-AGI-2 default: `completed`, two real measurements.
- ARC-AGI-2 forced blocker: `blocked`, two `not_run` conditions.
- BigCodeBench-Hard default: `completed`, two real measurements.
- BigCodeBench-Hard forced blocker: `blocked`, two `not_run` conditions.

## Remaining Coverage Gap

The 47-benchmark map in `agentic_eval_benchmarks.md` is a research adoption
map, not a list of already executable local runners. Current real local
coverage is concentrated in static reasoning and executable coding smokes, plus
external-harness compatibility probes. The next high-value runnable additions
are:

1. a real tau2 model-agent smoke on one mock-domain task;
2. a Harbor/Terminal-Bench model-agent smoke that keeps Harbor's verifier;
3. SciCode M0 importer/verifier/canonical-preflight work;
4. ToolSandbox or BFCL as the first bounded tool-use runner;
5. LongBench/RULER-style local long-context runner once the report schema can
   separate no-tools long-context from retrieval systems.
