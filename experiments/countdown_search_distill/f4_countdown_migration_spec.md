# F4 Countdown Migration Spec: shared run_common lifecycle

Wave F4 migrates the Countdown search-distill runners off the legacy per-run
JSONL manifest (`current.jsonl` / `<run-id>.jsonl`) onto the typed
begin/stage/finish lifecycle (F1) plus the profile doctor (F2). Countdown is a
separate subsystem under `experiments/countdown_search_distill/` with its own
`run_common.sh`; all ~11 runners source it and drive stages through one shared
helper, `countdown_run_stage`. Migrating that shared helper migrates every
runner at once. This follows the arithmetic-words reference runner
(`f4_reference_runner_migration_spec.md`), which proved the lifecycle on a
host-testable path first.

## Why the shared seam

- Every Countdown runner (`run_full_pilot.sh`, `run_calibration.sh`,
  `run_collect.sh`, `run_train.sh`, `run_eval.sh`, ...) already funnels its
  stages through `countdown_run_stage` in `run_common.sh`. Migrating that one
  helper gives all runners the typed attempt bundle without editing each script.
- `run_full_pilot.sh` is the orchestrator: it sets `RUN_ID`, resets the
  manifest, and calls `countdown_run_stage <name> <command>` for preflight,
  calibration_sweep, calibration, collect, validate_splits, per-arm `train_*`,
  per-split `base_eval_*`, export_adapters, eval_adapters, and
  build_report_input. These become typed stages.
- Unlike arithmetic-words, the real Countdown stages need GPU + vLLM, so the
  migration must gate on a device profile and record real measurements, not
  fixtures.

## Contract preserved (must not change)

- Runner names, flags, and invocation. `run_full_pilot.sh` still self-re-execs
  through `scripts/rootfs/enter_rootfs.sh` via `countdown_enter_rootfs_if_needed`.
- Scientific conditions and defaults: `MODE` (smoke/reduced/full), the arm sets
  per mode (`debug_smoke`; `raw hindsight curriculum`;
  `raw clean formatting hindsight curriculum`), calibration/collection seeds,
  the `preflight-reduced` gate, and all `TORCHTITAN_COUNTDOWN_*` env knobs and
  their defaults.
- Countdown's own `preflight-runtime` (model presence + free-GPU-memory check)
  and `preflight-reduced` (calibration/coverage gate) stay as-is; the typed
  preflight composes with, and does not replace, them.
- The legacy per-stage row content (stage name, run_id, mode, rootfs_active,
  timing, return_code, command, optional stage_status) is preserved as stage
  event extras so no operational detail is lost.

## Lifecycle wrapping (the migration)

`run_common.sh` gains typed-lifecycle wiring; `countdown_run_stage` drives one
typed stage per call. Concretely:

1. **Env setup (`countdown_setup_env`)** additionally exports the attempt
   locator: `TORCHTITAN_COUNTDOWN_ATTEMPT_ID` (default `attempt-01`) and the
   `RESULTS_ROOT` used as the lifecycle results root
   (`${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}`), so begin/stage/finish address one
   bundle under `<results-root>/runs/<run-id>/<attempt-id>/`.

2. **begin** is called once per run, by the orchestrator, immediately after it
   fixes `RUN_ID` (replacing the `: > manifest` reset). It freezes the Countdown
   declaration: `--family countdown --task search_distill --lane <MODE>` with
   `--fields` capturing MODE, the resolved arm set, and the seed/budget knobs so
   the declaration digest changes if any scientific knob changes. A helper
   `countdown_begin_attempt` in `run_common.sh` performs this and is idempotent
   for a fixed (run_id, attempt_id) so sub-runners invoked standalone can begin
   their own attempt.

3. **`countdown_run_stage <name> <command...>`** becomes a typed `stage`:
   - It still times the command and captures return code and optional
     `stage_status`, but now wraps the command through
     `python -m torchtitan.experiments.execution stage` with:
     - `--stage-id`/`--name` = the stage name (e.g. `collect`, `train_raw`).
     - `--kind` chosen from `models.STAGE_KINDS` by stage:
       `preflight` -> `preflight`; `calibration`/`calibration_sweep` ->
       `generate`; `collect` -> `generate`; `validate_splits` -> `verify`;
       `train_*` -> `train`; `base_eval_*`/`eval_adapters` -> `evaluate`;
       `export_adapters` -> `export`; `build_report_input` -> `report`.
     - `--adapter` from `models.EXECUTION_ADAPTERS`:
       generation/eval GPU stages -> `rootfs_vllm`; training -> `rootfs_torchrun_sft`;
       preflight/validate/report (no model forward) -> `rootfs_cpu`.
   - The legacy manifest row is retained by having the stage command carry the
     same metadata; stage events already record argv, timing, and return code
     on the coordinator stream, and `stage_status` JSON is attached as a stage
     extra when `TORCHTITAN_COUNTDOWN_STAGE_STATUS` is set.
   - The stage return code is propagated: a nonzero stage aborts the run
     (`set -euo pipefail`) before finish, exactly as today.

4. **Typed preflight** runs before real work: the orchestrator invokes
   `python -m torchtitan.experiments.execution preflight --profile vllm_1gpu
   --require-ready` (composed with Countdown's own `preflight-runtime` stage).
   `vllm_1gpu` requires rootfs active + torch + vllm + >=1 GPU, matching the
   real generation path. A blocked preflight stops the run and never fabricates
   an attempt outcome. `MODE=smoke` on a bare host is not a supported path; the
   real Countdown pilot always runs in the rootfs on GPUs.

5. **finish** is called once by the orchestrator after the terminal stage
   (build_report_input, or the last stage in smoke mode). It commits the
   immutable outcome with per-condition evaluations keyed by evaluation split
   and arm. Real generation/eval stages produce `measurement=real`; a
   `debug_smoke` arm records `measurement=smoke`; any blocked/failed condition
   records `measurement=invalid`/`not_run` with `promotion=not_evaluated`. The
   canonical report input is the existing `build-report-input` output.

### evaluations keying

`finish --evaluations` maps a condition key to `ConditionStatus`. Countdown
conditions are (arm x split) cells for adapter evaluation plus base_eval splits.
The orchestrator builds this JSON from the resolved arm/split lists (not by
hand) so adding an arm or split cannot silently drop a condition. Fixtures and
smoke arms never report `measurement=real`, keeping the derived run gate honest
(`has_real_measurement` reflects only real generation).

## Attempt bundle produced

Per `lifecycle.py`, under
`${TORCHTITAN_COUNTDOWN_RESULTS_ROOT}/runs/<RUN_ID>/<ATTEMPT_ID>/`:

- `manifest.json` (frozen Countdown declaration + digest)
- `processes/coordinator/events.jsonl` (one stage_started + one terminal per
  `countdown_run_stage` call, with legacy row content as extras)
- `outcome.json` (execution_outcome, stage invocation ids, per-condition
  evaluations, derived run_gate)
- `derived/report_input.json` (the Countdown `build-report-input` output)

The legacy `<run-id>.jsonl` manifest is no longer the source of truth; if kept
transiently for the existing `build-report-input --manifest` argument, it is
derived from the event stream, not written directly by `countdown_run_stage`.

## Acceptance criteria

1. `bash -n` parses `run_common.sh` and every sourcing runner; the rootfs
   re-exec guard and all `TORCHTITAN_COUNTDOWN_*` defaults are unchanged.
2. Host-testable: with a stubbed stage command (no GPU), `countdown_run_stage`
   drives a real typed `stage`, the coordinator event stream records the stage
   with its legacy row content as extras, and a nonzero stubbed command aborts
   the run with the propagated return code.
3. `countdown_begin_attempt` freezes one declaration; a second call for the same
   (run_id, attempt_id) is idempotent and does not corrupt the manifest.
4. On-device (required before landing): `MODE=smoke
   experiments/countdown_search_distill/run_full_pilot.sh` run through the bwrap
   rootfs with real vLLM on the B200s completes, gates on `vllm_1gpu`, produces
   the attempt bundle with all four files, records real generation stages as
   `measurement=real` (debug_smoke arm as `smoke`), and writes no legacy
   `current.jsonl` prototype manifest as the source of truth.
5. The Countdown-specific `preflight-runtime` and `preflight-reduced` gates
   still run and still stop the pilot on failure.

## Test plan

Host-testable tests (no GPU/vLLM), following
`tests/unit_tests/test_scaffold_arithmetic_words_runner.py` and
`test_scaffold_execution_foundation.py` (subprocess sourcing `run_common.sh`
with `TORCHTITAN_IN_ROOTFS=1` and a stub command):

- `test_countdown_run_stage_drives_typed_stage`: source `run_common.sh`, begin
  an attempt under a tmp results root, call `countdown_run_stage stub-stage
  true`, assert the coordinator event stream has one `stage_started` +
  `stage_succeeded` for `stub-stage` and the legacy row content is present as
  extras.
- `test_countdown_run_stage_propagates_failure`: `countdown_run_stage fail-stage
  false` returns nonzero and records `stage_failed`.
- `test_countdown_begin_attempt_idempotent`: two `countdown_begin_attempt` calls
  for the same locator leave one valid manifest.
- `test_countdown_stage_status_attached`: with
  `TORCHTITAN_COUNTDOWN_STAGE_STATUS` pointing at a JSON file, the stage extra
  carries the parsed stage_status.
- `test_countdown_runners_syntax`: `bash -n` on `run_common.sh` and all runners.

On-device verification (task 25, required before landing): the `MODE=smoke`
pilot in criterion 4, run through `scripts/rootfs/enter_rootfs.sh` on the
B200s.
