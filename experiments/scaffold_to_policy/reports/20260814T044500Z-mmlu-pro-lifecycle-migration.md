# F4 blocked-family lifecycle migration: MMLU-Pro

Date: 2026-08-14
Host: 8x NVIDIA B200, bwrap rootfs, `assets/hf/Qwen3-1.7B` present.

## Scope

Migrated `experiments/scaffold_to_policy/run_mmlu_pro_public_vllm_smoke.sh`
from the prototype `scaffold_run_stage` manifest path onto the typed
begin/stage/finish lifecycle. The runner keeps the prior MMLU-Pro public
defaults, runtime metadata capture, vLLM GPU-memory preflight, and graceful
blocker behavior.

This slice also extends `build-multiple-choice-report-input` to accept and
attach `--execution-preflight`, matching the already migrated modular,
math-style, and coding-style report builders.

Stage mapping:

- capture-runtime-metadata: `preflight` / `rootfs_cpu`
- import-dev / import-ood: `acquire` / `rootfs_cpu`
- validate-splits and require-gpu-memory-selected: `verify` / `rootfs_cpu`
- preflight-gpu-memory: `preflight` / `rootfs_vllm`
- base-eval-dev / base-eval-ood_test: `evaluate` / `rootfs_vllm`
- blocker and final report generation: `report` / `rootfs_cpu`

Terminal paths:

- success -> `attempt_outcome=completed`, per-split
  `completed/real/not_evaluated`
- blocker -> `attempt_outcome=blocked`, per-split
  `blocked/not_run/not_evaluated`

## Blocked-path verification

Run id `20260814T044000Z-mmlu-pro-blocked-typed`, with
`GPU_MEMORY_UTILIZATION=2.0` to force the existing GPU-memory selection gate to
fail before model execution.

- script exit code: 0
- stages recorded: capture-runtime-metadata, import-dev, import-ood,
  validate-splits, preflight-gpu-memory, require-gpu-memory-selected,
  write-blocker-report-input
- `require-gpu-memory-selected` recorded `stage_failed` rc=1, then the runner
  emitted a blocker report and finished the attempt
- outcome: `execution_outcome=blocked`
- conditions: dev and ood_test both `measurement=not_run`,
  `promotion=not_evaluated`
- run gate: `has_real_measurement=false`

## Successful-path verification

Run id `20260814T044500Z-mmlu-pro-typed-default`, default knobs.

- script exit code: 0
- stages recorded: capture-runtime-metadata, import-dev, import-ood,
  validate-splits, preflight-gpu-memory, require-gpu-memory-selected,
  base-eval-dev, base-eval-ood_test, build-report-input
- profile doctor: `vllm_1gpu` and `reasoning` both ready with empty blockers
- outcome: `execution_outcome=completed`
- conditions: dev and ood_test both `measurement=real`,
  `promotion=not_evaluated`
- run gate: `has_real_measurement=true`, `measurement_counts={real: 2}`

## Follow-up

Apply the same blocked-finish pattern to the remaining instrumented public-vLLM
runners: livecodebench, arc_agi2, gpqa, and bigcodebench_hard.
