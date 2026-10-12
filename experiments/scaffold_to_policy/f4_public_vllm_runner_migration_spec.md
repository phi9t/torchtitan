# F4 Migration Spec: public-vLLM runners (reasoning + coding references)

Wave F4 migrates the public-dataset vLLM smoke runners from the ad hoc shell
pipeline onto the typed begin/stage/finish lifecycle (F1) plus composed profile
preflight (F2). Two references are migrated first because they set the pattern
for the whole `run_*_public_vllm_smoke.sh` family:

- `run_gsm8k_public_vllm_smoke.sh`: public reasoning reference (no code
  execution).
- `run_humaneval_public_vllm_smoke.sh`: public coding reference (real sandboxed
  code execution during evaluation).

See ADR 0005-reorder-runner-migration-before-temporal,
`f4_reference_runner_migration_spec.md`, and
`f4_modular_sequences_transfer_migration_spec.md`.

## Why these runners next

- They are the simplest real-model path after the modular-sequences transfer
  runner: import pinned public splits, validate, evaluate the base model with
  vLLM, build the report input. There is no SFT, export, or adapter arm, so the
  migration is a clean base-only measurement.
- The public-vLLM family (gsm8k, math, mbpp, mmlu_pro, aime, humaneval,
  livecodebench, arc_agi2, gpqa, bigcodebench) shares one shape. Migrating a
  reasoning reference and a coding reference proves the pattern for both halves
  of the family; the rest follow the same wrapping.
- Unlike Countdown and modular-sequences, these runners previously wrote no
  manifest at all, so the migration is purely additive: it gives them the same
  immutable attempt bundle and validated report the migrated runners have.

## Contract preserved (must not change)

The scripts stay thin compatibility entrypoints. Unchanged:

- Script names and invocation, still self-re-exec through
  `scripts/rootfs/enter_rootfs.sh` when `TORCHTITAN_IN_ROOTFS != 1`.
- Scientific conditions and defaults: pinned `DATASET`/`DATASET_SUBSET`/
  `DATASET_REVISION`/`SOURCE_SPLIT`, `DEV_PROBLEMS`/`OOD_PROBLEMS` with their
  `DEV_OFFSET`/`OOD_OFFSET`, `NUM_ROLLOUTS`, `MAX_NEW_TOKENS`, `PROMPT_VARIANT`,
  `TEMPERATURE`, `TOP_P`, and (humaneval) `TIMEOUT_SECONDS`. The import
  provenance, split registry, per-split summaries, and report input land in the
  same paths with the same bytes for a fixed `RUN_ID`.
- Env knobs keep their current defaults and semantics.

## Preflight: composed profiles

These runners need both the public-dataset import path (`datasets`,
`transformers`) and vLLM generation (`vllm`, GPU). The preflight composes both
profiles so a missing package in either blocks the run before any work:

```
preflight --profile vllm_1gpu --profile reasoning --require-ready --output <artifact>
```

`vllm_1gpu` contributes rootfs_active + torch + vllm + gpu_count_ge_1;
`reasoning` contributes rootfs_active + torch + datasets + transformers +
gpu_count_ge_1. Composition deduplicates shared clauses and reports a single
readiness.

## Lifecycle wrapping (the migration)

1. `preflight` as above.
2. `begin ... --family <reasoning|coding> --task <task> --lane public_smoke
   --fields <json>` freezes the declaration; `--fields` records the pinned
   dataset identity (dataset, subset, revision, source split), per-split
   limit/offset, sampling, rollout budget, and (humaneval) timeout, built from
   the knobs in Python so the declaration digest changes if any knob changes.
3. One `stage` per pipeline step, each wrapping the existing CLI command:
   - `import-dev` / `import-ood`: `--kind acquire --adapter rootfs_cpu` wrapping
     `import-<task>-split`. Public dataset fetch is CPU/network work in the
     rootfs; `acquire` is the artifact-acquisition stage kind.
   - `validate-splits`: `--kind verify --adapter rootfs_cpu`.
   - `base-eval-<split>` per `dev`/`ood_test`: `--kind evaluate --adapter
     rootfs_vllm` wrapping `evaluate-<task>-vllm`. For humaneval the same
     command also runs sandboxed code execution with `--timeout-seconds`; the
     dominant runtime is vLLM generation inside the rootfs, and the sandbox is
     in-process (not an external harness), so `rootfs_vllm` is the honest single
     label. `coding_sandbox` and `external_harness` are reserved for stages
     whose executor is a separate sandbox/harness process.
   - `build-report-input`: `--kind report --adapter rootfs_cpu` wrapping
     `build-<style>-report-input --scaffold-budget <NUM_ROLLOUTS>`.
   Each stage records `--stage-extra rootfs_active=<bool>`. `set -euo pipefail`
   aborts before `finish` on a nonzero stage.
4. `finish ... --attempt-outcome completed --report-input <report_input.json>
   --evaluations <evaluations.json>`.

### evaluations.json

Both references perform real vLLM base-model evaluation, so both eval
conditions are real measurements. There is no trained arm and no
calibration/promotion gate, so promotion is `not_evaluated`:

```json
{
  "dev":      {"execution_outcome": "completed", "measurement": "real", "promotion": "not_evaluated"},
  "ood_test": {"execution_outcome": "completed", "measurement": "real", "promotion": "not_evaluated"}
}
```

The derived run gate reports `has_real_measurement=true` with
`promotion_counts={not_evaluated: 2}`. Built from the split list so adding a
split cannot silently drop a condition.

## Attempt bundle layout produced

Per `lifecycle.py`, under `<RESULTS_ROOT>/runs/<RUN_ID>/<ATTEMPT_ID>/`:
`manifest.json`, `processes/coordinator/events.jsonl`, `outcome.json`,
`derived/report_input.json`. `ATTEMPT_ID` defaults to `attempt-01`.

## Acceptance criteria

1. `bash -n` parses both scripts; rootfs re-exec guard and all env knobs
   preserved with current defaults.
2. Same import provenance, split registry, and `build-<style>-report-input`
   output for a fixed `RUN_ID` as the pre-migration script (scientific parity),
   on device.
3. Attempt bundle complete; `outcome.json` has `execution_outcome=completed`,
   both eval conditions `measurement=real`, `run_gate.has_real_measurement ==
   true`.
4. A composed `vllm_1gpu`+`reasoning` preflight artifact is written and readiness
   is `ready` on the device; the run refuses to proceed if preflight is blocked.
5. No prototype manifest is written by these runners.

## Test plan

The GPU path is verified on device (8x B200, real public dataset fetch + vLLM
evaluation, plus sandboxed code execution for humaneval). Host-side coverage is
`bash -n` syntax. Real bundle validation runs under
`scripts/rootfs/enter_rootfs.sh` and is captured in a timestamped report under
`experiments/scaffold_to_policy/reports/`.
