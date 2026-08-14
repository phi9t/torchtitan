# F4 modular-sequences transfer: typed-lifecycle migration (GPU path)

Run ID: `20260814T000000Z-modular-transfer-migration`
Attempt: `attempt-01`
Device: 8x NVIDIA B200, `scripts/rootfs/enter_rootfs.sh` (bwrap rootfs), real
vLLM + torchrun LoRA SFT.

## What this proves

The synthetic reasoning transfer runner
(`run_modular_sequences_transfer_smoke.sh`) is migrated off the prototype
`record_stage`/JSONL manifest onto the typed begin/stage/finish lifecycle (F1)
plus the `vllm_1gpu` profile doctor (F2). This is the second F4 runner (after the
host-testable arithmetic-words reference) and the first migration verified on the
real GPU path: vLLM generation/evaluation, torchrun LoRA SFT, LoRA export, and
adapter evaluation. See
`experiments/scaffold_to_policy/f4_modular_sequences_transfer_migration_spec.md`.

## Acceptance criteria (all met on device)

1. `bash -n` parses; rootfs re-exec guard and all env knobs preserved with
   current defaults.
2. Scientific conditions unchanged: seeds train=4100/32, dev=4200/8,
   ood_test=4300/8; single `raw` LoRA SFT arm
   (`qwen3_1_7b_modular_sequences_lora_raw`, 4 steps, rank 16 / alpha 32);
   `build-modular-report-input --no-require-selected`.
3. Attempt bundle complete: `manifest.json`, `processes/coordinator/events.jsonl`
   (13 stages, one `stage_started` + one `stage_succeeded` each, all rc=0),
   `outcome.json`, `derived/report_input.json`.
4. `outcome.json`: `execution_outcome=completed`; all four eval conditions
   (`base_dev`, `base_ood_test`, `adapter_raw_dev`, `adapter_raw_ood_test`)
   `measurement=real`, `promotion=not_evaluated`; `run_gate.has_real_measurement
   == true`, `measurement_counts={real: 4}`.
5. `vllm_1gpu` preflight artifact written with `blocker_codes: []`; every clause
   (rootfs_active, torch, vllm, gpu_count_ge_1) passes.
6. No prototype JSONL manifest written; `record_stage` removed.

Report-input checks all pass: `preflights_present`, `summaries_present`,
`split_registry_selected`, `runtime_metadata_present`,
`artifact_provenance_labeled`, `preflight_split_counts_match`,
`summary_split_counts_match`.

## Stage order (from the coordinator event stream)

gen-train, gen-dev, gen-ood_test (generate/rootfs_cpu) -> validate-splits
(verify/rootfs_cpu) -> collect-train (generate/rootfs_vllm) -> build-dataset
(prepare/rootfs_cpu) -> base-eval-dev, base-eval-ood_test (evaluate/rootfs_vllm)
-> train-raw (train/rootfs_torchrun_sft) -> export-raw (export/rootfs_cpu) ->
eval-adapter-raw-dev, eval-adapter-raw-ood_test (evaluate/rootfs_vllm) ->
build-report-input (report/rootfs_cpu).

## Measured numbers (smoke scale: 8 problems/split, 8 rollouts, 64 total)

These are a plumbing measurement, not evidence: at 8 problems/split the deltas
carry no statistical weight. They are recorded only to confirm real base-vs-
adapter numbers flow end to end.

- dev pass@1: base 0.500 -> adapter_raw 0.500; pass@32: base 0.500 -> 0.625.
- ood_test pass@1: base 0.500 -> adapter_raw 0.625; pass@32: base 0.875 -> 0.750.
- Base dev/ood_test show `missing FINAL line` failures (22/19 rollouts); the raw
  adapter eliminates those format failures on both splits, which is the expected
  first-order effect of SFT on the scaffold format.

## Bottom line

The typed lifecycle now drives a standalone GPU runner that owns its whole
pipeline (generation + real SFT + export + adapter eval) end to end, with an
immutable, validated attempt bundle and a real-measurement run gate. F4 runner
migration advances from one host-testable reference plus Countdown to a third
runner on the real GPU path.
