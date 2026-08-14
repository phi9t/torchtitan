# F4 blocked-family lifecycle migration: LiveCodeBench

Date: 2026-08-14
Host: 8x NVIDIA B200, bwrap rootfs, `assets/hf/Qwen3-1.7B` present.

## Scope

Migrated `experiments/scaffold_to_policy/run_livecodebench_public_vllm_smoke.sh`
from the prototype `scaffold_run_stage` manifest path onto the typed
begin/stage/finish lifecycle. The runner keeps the prior public LiveCodeBench
defaults, contest-code sandbox timeout, vLLM GPU-memory preflight, and graceful
blocker behavior.

This slice also extends `build-contest-code-report-input` to accept and attach
`--execution-preflight`, matching the migrated modular, math-style,
coding-style, and multiple-choice report builders.

Stage mapping:

- import-dev / import-ood: `acquire` / `rootfs_cpu`
- validate-splits and require-gpu-memory-selected: `verify` / `rootfs_cpu`
- preflight-gpu-memory: `preflight` / `rootfs_vllm`
- evaluate-splits: `evaluate` / `rootfs_vllm`
- blocker and final report generation: `report` / `rootfs_cpu`

Terminal paths:

- success -> `attempt_outcome=completed`, per-split
  `completed/real/not_evaluated`
- blocker -> `attempt_outcome=blocked`, per-split
  `blocked/not_run/not_evaluated`

## Blocked-path verification

Run id `20260814T052000Z-livecodebench-blocked-typed`, with
`GPU_MEMORY_UTILIZATION=2.0` to force the existing GPU-memory selection gate to
fail before model execution.

- script exit code: 0
- stages recorded: import-dev, import-ood, validate-splits,
  preflight-gpu-memory, require-gpu-memory-selected, write-blocker-report-input
- outcome: `execution_outcome=blocked`
- conditions: dev and ood_test both `measurement=not_run`,
  `promotion=not_evaluated`
- run gate: `has_real_measurement=false`

## Successful-path verification

Run id `20260814T052500Z-livecodebench-typed-default`, default knobs.

- script exit code: 0
- stages recorded: import-dev, import-ood, validate-splits,
  preflight-gpu-memory, require-gpu-memory-selected, evaluate-splits,
  build-report-input
- profile doctor: `vllm_1gpu` and `reasoning` both ready with empty blockers
- outcome: `execution_outcome=completed`
- conditions: dev and ood_test both `measurement=real`,
  `promotion=not_evaluated`
- run gate: `has_real_measurement=true`, `measurement_counts={real: 2}`

## Validation

- `bash -n experiments/scaffold_to_policy/run_livecodebench_public_vllm_smoke.sh`
- `python -m pytest tests/unit_tests/test_scaffold_to_policy.py tests/unit_tests/test_scaffold_execution_foundation.py tests/unit_tests/test_execution_lifecycle_run_attempt.py tests/unit_tests/test_execution_lifecycle_done_scenarios.py -q`

## Follow-up

Apply the same blocked-finish pattern to ARC-AGI-2 and BigCodeBench-Hard.
