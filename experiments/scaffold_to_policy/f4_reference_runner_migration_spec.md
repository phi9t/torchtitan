# F4 Reference Runner Migration Spec: arithmetic-words

Wave F4 migrates one host-testable reference runner
(`run_arithmetic_words_smoke.sh`) from the prototype manifest path onto the
typed begin/stage/finish lifecycle (F1) plus the `host_static` profile doctor
(F2), proving the foundation end to end before any runner is wrapped in a
durable Activity. See ADR 0005-reorder-runner-migration-before-temporal.

## Why this runner first

- It is fully host-testable: fixture rollouts, no GPU and no vLLM. The
  `host_static` preflight profile is ungated, so the whole path runs on the
  host under the bwrap rootfs re-exec guard without device access.
- Its scientific conditions are two evaluation splits (`dev`, `ood_test`) built
  from deterministic seeds, so migration correctness is checkable by asserting
  the same splits, the same fixture rollouts, and the same report input as the
  pre-migration script.
- Countdown and the other runners migrate only after this reference runner
  proves compatibility (roadmap Section 18.2 and Wave F4).

## Contract preserved (must not change)

The script stays a thin compatibility entrypoint. These are unchanged:

- Script name and invocation: `run_arithmetic_words_smoke.sh [args]`, still
  self-re-execs through `scripts/rootfs/enter_rootfs.sh` when
  `TORCHTITAN_IN_ROOTFS != 1`.
- Scientific conditions and defaults: `generate-arithmetic-words` seeds
  (train=100/12, dev=200/8, ood_test=300/8), `validate-arithmetic-splits`,
  `write-arithmetic-fixture` + `evaluate-arithmetic-fixture --max-rollouts 3`
  per eval split, `build-arithmetic-report-input`.
- Env knobs: `RUN_ID`, `DATA_ROOT`, `RESULTS_ROOT` keep their current defaults
  and semantics. The generated data and eval artifacts land in the same paths.

## Lifecycle wrapping (the migration)

The runner drives the F1 CLI (`python -m torchtitan.experiments.execution`)
instead of writing the prototype manifest. Concretely:

1. `preflight --profile host_static --require-ready --output <artifact>` before
   any work. A blocked preflight stops the run (nonzero exit) with the blocker
   codes; it never fabricates an attempt outcome.
2. `begin --results-root <RESULTS_ROOT> --run-id <RUN_ID> --attempt-id <ATTEMPT_ID>
   --family reasoning --task arithmetic_words --lane host_smoke --fields <json>`
   freezes the declaration. `--fields` records the normalized declaration
   content (seeds, num-problems per split, max-rollouts) so the declaration
   digest changes if any scientific knob changes.
3. One `stage` per pipeline step, each wrapping the existing CLI command after
   `--`. Stage kind and adapter come from `models.STAGE_KINDS` /
   `EXECUTION_ADAPTERS`:
   - `stage --stage-id gen-train --name generate-train --kind generate
     --adapter host_test -- python -m ...cli generate-arithmetic-words ...`
     (and `gen-dev`, `gen-ood_test`).
   - `stage --stage-id validate-splits --name validate-splits --kind verify
     --adapter host_test -- python -m ...cli validate-arithmetic-splits ...`
   - per eval split (`dev`, `ood_test`):
     `stage --stage-id write-fixture-<split> --kind generate --adapter host_test`
     wrapping `write-arithmetic-fixture`, then
     `stage --stage-id evaluate-<split> --kind evaluate --adapter host_test`
     wrapping `evaluate-arithmetic-fixture ... --max-rollouts 3`.
   - `stage --stage-id build-report-input --name build-report-input --kind report
     --adapter host_test` wrapping `build-arithmetic-report-input`, writing the
     canonical report input to a temp path the runner then feeds to `finish`.
   Each `stage` returns the wrapped command's return code; the runner is
   `set -euo pipefail`, so a nonzero stage aborts the run before `finish`.
4. `finish --results-root <RESULTS_ROOT> --run-id <RUN_ID> --attempt-id <ATTEMPT_ID>
   --attempt-outcome completed --report-input <report_input.json>
   --evaluations <evaluations.json>` commits the immutable outcome and the
   derived report input into the attempt bundle.

### evaluations.json

`finish` maps a condition key to `ConditionStatus(execution_outcome,
measurement, promotion)`. This runner uses fixture rollouts, so each eval split
is a fixture condition, never a real measurement:

```json
{
  "dev":      {"execution_outcome": "completed", "measurement": "fixture", "promotion": "not_evaluated"},
  "ood_test": {"execution_outcome": "completed", "measurement": "fixture", "promotion": "not_evaluated"}
}
```

The derived run gate therefore reports `has_real_measurement=false`, which is
correct: a smoke runner on fixtures must never masquerade as evidence
(execution_status Section 7.1). The runner builds this JSON from the split list,
not by hand, so adding a split cannot silently drop a condition.

## Attempt bundle layout produced

Per `lifecycle.py`, `finish` writes under
`<RESULTS_ROOT>/runs/<RUN_ID>/<ATTEMPT_ID>/`:

- `manifest.json` (frozen declaration + digest, from `begin`)
- `processes/coordinator/events.jsonl` (stage_started + one terminal per stage)
- `outcome.json` (execution_outcome, stage invocation ids, per-condition
  evaluations, derived run_gate)
- `derived/report_input.json` (the canonical arithmetic report input)

`ATTEMPT_ID` defaults to `attempt-01` and is overridable via env for resume
tests; a resume is a new attempt linking `--parent-attempt-id`, never a reopen.

## Acceptance criteria

1. `bash -n run_arithmetic_words_smoke.sh` parses; the script keeps the rootfs
   re-exec guard and the three env knobs with current defaults.
2. Running the migrated script on the host produces the same `dev`/`ood_test`
   splits, fixture rollouts, and `build-arithmetic-report-input` output bytes as
   the pre-migration script for a fixed `RUN_ID` (scientific parity).
3. The attempt bundle exists with all four files above; `outcome.json` has
   `execution_outcome=completed`, both eval conditions present with
   `measurement=fixture`, and `run_gate.has_real_measurement == false`.
4. A `host_static` preflight artifact is written and readiness is `ready` on the
   host; the run refuses to proceed if preflight is blocked.
5. No prototype manifest is written by this runner (it no longer calls
   `scaffold_setup_run_manifest`).

## Test plan (host-testable, no GPU)

Extend the shell-runner test harness pattern in
`tests/unit_tests/test_scaffold_execution_foundation.py` (subprocess that
sources the script or invokes it end to end under a tmp `RESULTS_ROOT`):

- `test_arithmetic_words_runner_produces_attempt_bundle`: run the script with a
  fixed `RUN_ID`/`ATTEMPT_ID` and tmp roots; assert the four bundle files exist
  and `outcome.json` matches the fixture-condition expectations in criterion 3.
- `test_arithmetic_words_runner_preflight_artifact_ready`: assert the
  `host_static` preflight artifact is written with readiness `ready`.
- `test_arithmetic_words_runner_report_input_parity`: assert the report input in
  `derived/report_input.json` equals the output of the standalone
  `build-arithmetic-report-input` for the same inputs (scientific parity).
- `test_arithmetic_words_runner_syntax`: `bash -n` on the script.

All tests run on the host without GPU or vLLM. Real device runners
(`rootfs_cpu`, `vllm_1gpu`, `reasoning`) are out of scope for F4; they migrate
in later F4 sub-steps once this reference runner is green.
