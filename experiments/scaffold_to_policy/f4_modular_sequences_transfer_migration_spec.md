# F4 Migration Spec: modular-sequences transfer (GPU path)

Wave F4 migrates the synthetic reasoning transfer runner
(`run_modular_sequences_transfer_smoke.sh`) from the prototype
`record_stage`/JSONL manifest path onto the typed begin/stage/finish lifecycle
(F1) plus profile preflight (F2). It is the second F4 runner migrated after the
host-testable arithmetic-words reference, and the first migration on a real GPU
path (vLLM generation + torchrun LoRA SFT + export + adapter evaluation).
See ADR 0005-reorder-runner-migration-before-temporal and
`f4_reference_runner_migration_spec.md`.

## Why this runner next

- The arithmetic-words reference proved the lifecycle end to end on the host
  with fixture rollouts. This runner proves the same lifecycle on the real GPU
  path: `vllm_1gpu` preflight, `rootfs_vllm` generation/evaluation stages, a
  `rootfs_torchrun_sft` training stage, a `rootfs_cpu` export stage, and a
  `rootfs_cpu` report stage. Countdown already exercises this pattern through
  `run_common.sh`; migrating a standalone runner confirms the pattern
  generalizes to a runner that owns its whole pipeline in one script.
- Its scientific conditions are deterministic-seed splits plus one trained arm
  (`raw`), so migration correctness is checkable by asserting the same splits,
  the same report input, and a validated attempt bundle.

## Contract preserved (must not change)

The script stays a thin compatibility entrypoint. Unchanged:

- Script name and invocation:
  `run_modular_sequences_transfer_smoke.sh [args]`, still self-re-execs through
  `scripts/rootfs/enter_rootfs.sh` when `TORCHTITAN_IN_ROOTFS != 1`.
- Scientific conditions and defaults: `generate-modular-sequences` seeds
  (train=4100/32, dev=4200/8, ood_test=4300/8), step/modulus ranges,
  `validate-modular-splits`, base `evaluate-modular-vllm` collection on train,
  `build-modular-dataset --condition raw`, the single `raw` LoRA SFT arm
  (`qwen3_1_7b_modular_sequences_lora_raw`, `TRAIN_STEPS=4`, rank 16 / alpha 32),
  `export-lora`, adapter `evaluate-modular-vllm` on `dev`/`ood_test`, and
  `build-modular-report-input --no-require-selected`.
- Env knobs: `RUN_ID`, `MODEL`, `DATA_ROOT`, `RESULTS_ROOT`, and all sampling /
  training knobs keep their current defaults and semantics. Generated data,
  adapters, and eval artifacts land in the same paths.

## Lifecycle wrapping (the migration)

The runner drives the F1 CLI (`python -m torchtitan.experiments.execution`)
instead of writing the prototype JSONL manifest via `record_stage`.

1. `preflight --profile vllm_1gpu --require-ready --output <artifact>` before any
   work. `vllm_1gpu` composes the rootfs-active clause, `torch`, `vllm`, and at
   least one visible GPU. A blocked preflight stops the run (nonzero exit) with
   the blocker codes; it never fabricates an attempt outcome.
2. `begin ... --family reasoning --task modular_sequences_transfer
   --lane gpu_smoke --fields <json>` freezes the declaration. `--fields` records
   the scientific knobs (per-split seed/num-problems, step/modulus ranges,
   sampling, rollout budgets, train steps, LoRA rank/alpha, arm=`raw`) so the
   declaration digest changes if any knob changes.
3. One `stage` per pipeline step, each wrapping the existing CLI command after
   `--`. Stage kind and adapter mirror the Countdown maps
   (`run_common.sh:countdown_stage_kind` / `countdown_stage_adapter`):
   - `gen-train` / `gen-dev` / `gen-ood_test`: `--kind generate
     --adapter rootfs_vllm`? No -- generation of the synthetic problem splits is
     CPU-only enumeration, so these are `--kind generate --adapter rootfs_cpu`.
   - `validate-splits`: `--kind verify --adapter rootfs_cpu`.
   - `collect-train`: base vLLM rollout collection on the train split,
     `--kind generate --adapter rootfs_vllm`.
   - `build-dataset`: `--kind prepare --adapter rootfs_cpu` wrapping
     `build-modular-dataset --condition raw`.
   - `base-eval`: base vLLM evaluation on `dev`/`ood_test`,
     `--kind evaluate --adapter rootfs_vllm`.
   - `train-raw`: `--kind train --adapter rootfs_torchrun_sft` wrapping the
     `torchrun ... qwen3_1_7b_modular_sequences_lora_raw` command.
   - `export-raw`: `--kind export --adapter rootfs_cpu` wrapping `export-lora`.
   - `eval-adapter-raw`: adapter vLLM evaluation on `dev`/`ood_test`,
     `--kind evaluate --adapter rootfs_vllm`.
   - `build-report-input`: `--kind report --adapter rootfs_cpu` wrapping
     `build-modular-report-input`, writing the canonical report input the runner
     feeds to `finish`.
   Each `stage` records `--stage-extra rootfs_active=<bool>` so the terminal
   event carries the same provenance the legacy JSONL row did. The runner is
   `set -euo pipefail`, so a nonzero stage aborts before `finish`.
4. `finish ... --attempt-outcome completed --report-input <report_input.json>
   --evaluations <evaluations.json>` commits the immutable outcome and the
   derived report input into the attempt bundle.

### evaluations.json

Unlike the arithmetic-words reference, this runner performs real vLLM
evaluation on Qwen3-1.7B, so every eval condition is a real measurement
(`measurement=real`). It is a smoke-sized transfer probe with a single trained
arm and no calibration/promotion gate, so promotion is `not_evaluated`:

```json
{
  "base_dev":        {"execution_outcome": "completed", "measurement": "real", "promotion": "not_evaluated"},
  "base_ood_test":   {"execution_outcome": "completed", "measurement": "real", "promotion": "not_evaluated"},
  "adapter_raw_dev": {"execution_outcome": "completed", "measurement": "real", "promotion": "not_evaluated"},
  "adapter_raw_ood_test": {"execution_outcome": "completed", "measurement": "real", "promotion": "not_evaluated"}
}
```

The derived run gate therefore reports `has_real_measurement=true` with
`promotion_counts={not_evaluated: 4}`: real numbers were measured, but this
smoke run does not itself promote an arm. The runner builds this JSON from the
condition list, not by hand, so adding a condition cannot silently drop one.

## Attempt bundle layout produced

Per `lifecycle.py`, `finish` writes under
`<RESULTS_ROOT>/runs/<RUN_ID>/<ATTEMPT_ID>/`:

- `manifest.json` (frozen declaration + digest, from `begin`)
- `processes/coordinator/events.jsonl` (stage_started + one terminal per stage)
- `outcome.json` (execution_outcome, stage invocation ids, per-condition
  evaluations, derived run_gate)
- `derived/report_input.json` (the canonical modular report input)

`ATTEMPT_ID` defaults to `attempt-01` and is overridable via env for resume
tests; a resume is a new attempt linking `--parent-attempt-id`, never a reopen.

## Acceptance criteria

1. `bash -n run_modular_sequences_transfer_smoke.sh` parses; the script keeps
   the rootfs re-exec guard and all current env knobs with current defaults.
2. The migrated script produces the same `train`/`dev`/`ood_test` splits, the
   same `raw` SFT dataset, and the same `build-modular-report-input` output for a
   fixed `RUN_ID` as the pre-migration script (scientific parity), on device.
3. The attempt bundle exists with all four files; `outcome.json` has
   `execution_outcome=completed`, all four eval conditions present with
   `measurement=real`, and `run_gate.has_real_measurement == true`.
4. A `vllm_1gpu` preflight artifact is written and readiness is `ready` on the
   device; the run refuses to proceed if preflight is blocked.
5. No prototype JSONL manifest is written by this runner (it no longer defines
   or calls `record_stage`).

## Test plan

The GPU path is verified on device (8x B200, real vLLM + torchrun SFT), not in
host unit tests, matching how the Countdown migration was verified. Host-side
CI coverage is limited to `bash -n` syntax; the real bundle validation runs
under `scripts/rootfs/enter_rootfs.sh` and is captured in a timestamped report
under `experiments/scaffold_to_policy/reports/`.
