# F4 public-vLLM runners: typed-lifecycle migration (reasoning + coding)

Device: 8x NVIDIA B200, `scripts/rootfs/enter_rootfs.sh` (bwrap rootfs), real
public-dataset fetch + real vLLM evaluation (plus sandboxed code execution for
HumanEval).

## What this proves

The public-dataset vLLM smoke runners are migrated off their ad hoc shell
pipelines onto the typed begin/stage/finish lifecycle (F1) with a composed
`vllm_1gpu`+`reasoning` profile preflight (F2). Two references were migrated
because they set the pattern for the whole `run_*_public_vllm_smoke.sh` family:

- `run_gsm8k_public_vllm_smoke.sh` -- public reasoning reference, run
  `20260814T014000Z-gsm8k-lifecycle-migration`.
- `run_humaneval_public_vllm_smoke.sh` -- public coding reference (real
  sandboxed code execution during evaluation), run
  `20260814T014500Z-humaneval-lifecycle-migration`.

See `experiments/scaffold_to_policy/f4_public_vllm_runner_migration_spec.md`.

## Acceptance criteria (all met on device, both runners)

1. `bash -n` parses; rootfs re-exec guard and all env knobs preserved with
   current defaults.
2. Scientific conditions unchanged: pinned dataset revision, subset, source
   split, per-split limit/offset, sampling, rollout budget, and (HumanEval)
   `--timeout-seconds`. Import provenance, split registry, per-split summaries,
   and report input land in the same paths.
3. Attempt bundle complete: `manifest.json`, `processes/coordinator/events.jsonl`
   (6 stages: import-dev, import-ood, validate-splits, base-eval-dev,
   base-eval-ood_test, build-report-input; one start + one succeed each, all
   rc=0), `outcome.json`, `derived/report_input.json`.
4. `outcome.json`: `execution_outcome=completed`; both eval conditions
   `measurement=real`; `run_gate.has_real_measurement == true`,
   `measurement_counts={real: 2}`, `promotion_counts={not_evaluated: 2}`.
5. Composed `vllm_1gpu`+`reasoning` preflight artifact written with
   `readiness=ready`, `blocker_codes: []`; both profiles present.
6. No prototype manifest written (these runners previously wrote none; the
   migration is purely additive).

All report-input checks pass for both runners (`preflights_present`,
`summaries_present`, `split_registry_selected`, `runtime_metadata_present`,
`artifact_provenance_labeled`, `preflight_split_counts_match`,
`summary_split_counts_match`).

## Stage adapters

- `import-dev` / `import-ood`: `acquire` / `rootfs_cpu` (public dataset fetch is
  CPU/network work in the rootfs).
- `validate-splits`: `verify` / `rootfs_cpu`.
- `base-eval-<split>`: `evaluate` / `rootfs_vllm`. For HumanEval the same command
  also runs in-process sandboxed code execution; the dominant runtime is rootfs
  vLLM generation, so `rootfs_vllm` is the honest single label.
- `build-report-input`: `report` / `rootfs_cpu`.

## Measured numbers (smoke scale, base-only)

Plumbing evidence, not benchmark results: small problem counts carry no
statistical weight. Recorded to confirm real numbers flow end to end.

- GSM8K (8 problems/split, 4 rollouts): dev pass@1 0.625, pass@32 0.750;
  ood_test pass@1 0.625, pass@32 0.750.
- HumanEval (4 problems/split, 4 rollouts, 5s exec timeout): dev pass@1 0.750;
  ood_test pass@1 0.250.

## Bottom line

The typed lifecycle now drives the public-vLLM reasoning and coding reference
runners end to end, each producing an immutable validated attempt bundle with a
real-measurement run gate. The remaining public-vLLM runners (math, mbpp,
mmlu_pro, aime, livecodebench, arc_agi2, gpqa, bigcodebench) share this exact
shape and follow the same wrapping. F4 runner migration now covers a
host-testable reference, Countdown, modular-sequences transfer, and both
public-vLLM references.
