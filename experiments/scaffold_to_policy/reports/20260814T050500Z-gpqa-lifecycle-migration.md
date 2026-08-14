# F4 blocked-family lifecycle migration: GPQA

Date: 2026-08-14
Host: 8x NVIDIA B200, bwrap rootfs, `assets/hf/Qwen3-1.7B` present.

## Scope

Migrated `experiments/scaffold_to_policy/run_gpqa_public_vllm_smoke.sh` from
the prototype `scaffold_run_stage` manifest path onto the typed
begin/stage/finish lifecycle. The runner keeps the prior GPQA Diamond public
defaults, offline/cache options, runtime metadata capture, vLLM GPU-memory
preflight, and graceful blocker behavior.

GPQA has an additional blocker before GPU work: dataset import can fail when
the gated Hugging Face dataset is unavailable inside the rootfs. The typed
runner preserves that behavior by finishing the attempt as blocked with
not_run conditions.

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
- import/GPU/runtime blocker -> `attempt_outcome=blocked`, per-split
  `blocked/not_run/not_evaluated`

## Import-blocked verification

Run id `20260814T050000Z-gpqa-import-blocked-typed`, with `OFFLINE=1` and no
raw cache. This forces the existing GPQA access/import blocker before model
execution.

- script exit code: 0
- stages recorded: capture-runtime-metadata, import-dev,
  write-import-blocker-report-input
- `import-dev` recorded `stage_failed` rc=1, then the runner emitted a blocker
  report and finished the attempt
- outcome: `execution_outcome=blocked`
- conditions: dev and ood_test both `measurement=not_run`,
  `promotion=not_evaluated`
- run gate: `has_real_measurement=false`

## Default-path verification on this host

Run id `20260814T050500Z-gpqa-typed-default`, default knobs.

The default path also blocked at `import-dev` in this environment, consistent
with GPQA Diamond being gated and no usable rootfs HF token/cache being present.
The typed behavior was still correct:

- script exit code: 0
- outcome: `execution_outcome=blocked`
- conditions: dev and ood_test both `measurement=not_run`
- run gate: `has_real_measurement=false`
- blocker report preserved runtime metadata and the import failure artifact

A real GPQA evaluation path can be run later by supplying rootfs-visible
`HF_TOKEN` auth or raw caches through `DEV_RAW_CACHE` and `OOD_RAW_CACHE`; no
runner changes are required for that path.

## Validation

- `bash -n experiments/scaffold_to_policy/run_gpqa_public_vllm_smoke.sh`
- `python -m pytest tests/unit_tests/test_scaffold_to_policy.py tests/unit_tests/test_scaffold_execution_foundation.py tests/unit_tests/test_execution_lifecycle_run_attempt.py tests/unit_tests/test_execution_lifecycle_done_scenarios.py -q`

## Follow-up

Apply the same blocked-finish pattern to the remaining instrumented public-vLLM
runners: livecodebench, arc_agi2, and bigcodebench_hard.
