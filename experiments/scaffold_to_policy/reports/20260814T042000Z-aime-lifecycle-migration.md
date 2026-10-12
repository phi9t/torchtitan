# F4 blocked-family lifecycle migration: AIME

Date: 2026-08-14
Host: 8x NVIDIA B200, bwrap rootfs, `assets/hf/Qwen3-1.7B` present.

## Scope

Migrated `experiments/scaffold_to_policy/run_aime_public_vllm_smoke.sh` from the
prototype `scaffold_run_stage` manifest path onto the typed
begin/stage/finish lifecycle. The runner keeps the prior AIME public-dataset
defaults, vLLM GPU-memory preflight, and graceful blocker behavior.

Stage mapping:

- import-dev / import-ood: `acquire` / `rootfs_cpu`
- validate-splits and require-gpu-memory-selected: `verify` / `rootfs_cpu`
- preflight-gpu-memory: `preflight` / `rootfs_vllm`
- base-eval-dev / base-eval-ood_test: `evaluate` / `rootfs_vllm`
- blocker and final report generation: `report` / `rootfs_cpu`

The typed runner now finishes both terminal paths:

- success -> `attempt_outcome=completed`, per-split
  `completed/real/not_evaluated`
- blocker -> `attempt_outcome=blocked`, per-split
  `blocked/not_run/not_evaluated`

## Blocked-path verification

Run id `20260814T041500Z-aime-blocked-typed`, with
`GPU_MEMORY_UTILIZATION=2.0` to force the existing GPU-memory selection gate to
fail before model execution.

- script exit code: 0
- stages recorded: import-dev, import-ood, validate-splits,
  preflight-gpu-memory, require-gpu-memory-selected, write-blocker-report-input
- `require-gpu-memory-selected` recorded `stage_failed` rc=1, then the runner
  emitted a blocker report and finished the attempt
- outcome: `execution_outcome=blocked`
- conditions: dev and ood_test both `measurement=not_run`,
  `promotion=not_evaluated`
- run gate: `has_real_measurement=false`

## Successful-path verification

Run id `20260814T042000Z-aime-typed-default`, default knobs.

- script exit code: 0
- stages recorded: import-dev, import-ood, validate-splits,
  preflight-gpu-memory, require-gpu-memory-selected, base-eval-dev,
  base-eval-ood_test, build-report-input
- profile doctor: `vllm_1gpu` and `reasoning` both ready with empty blockers
- outcome: `execution_outcome=completed`
- conditions: dev and ood_test both `measurement=real`,
  `promotion=not_evaluated`
- run gate: `has_real_measurement=true`, `measurement_counts={real: 2}`

## Follow-up

The same blocked-finish pattern should be applied to the remaining
instrumented public-vLLM runners: mmlu_pro, livecodebench, arc_agi2, gpqa, and
bigcodebench_hard.
