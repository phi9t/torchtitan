# Countdown typed-lifecycle migration: on-device smoke + reduced pilot

Date: 2026-08-13 (UTC)
Run family: countdown / search_distill
Runtime: bwrap rootfs on 8x NVIDIA B200, real vLLM generation and torchrun SFT
Landed commits: `5c0a17a5e` (run_common lifecycle migration), `b1a3e7b66` (report reads typed event stream)

## Why this run happened

The F4 wave migrated `experiments/countdown_search_distill/run_common.sh` off the
legacy JSONL manifest and onto the typed execution lifecycle
(`python -m torchtitan.experiments.execution stage`). Every Countdown runner
sources `run_common.sh` and drives its stages through `countdown_run_stage`, so
migrating that one helper migrates all ~11 runners at once. The user's bar was
strict: prove the migrated path on-device with real vLLM on the B200s before
treating it as done, everything hermetic through the rootfs.

## What ran

All work ran inside `scripts/rootfs/enter_rootfs.sh` (`rootfs_active: true` on
every stage event). Host Python was used only for CPU-only unit tests.

### Smoke (`MODE=smoke run_full_pilot.sh`)

Attempt bundle `results/runs/20260813T215545Z-smoke/attempt-01/`. All seven
smoke stages recorded on the typed coordinator event stream, each `return_code:
0`:

- preflight (kind preflight, 11s)
- calibration (generate, real vLLM, 48s)
- collect (generate, real vLLM, 2m48s)
- train_debug_smoke (train, 12s)
- base_eval_dev / base_eval_iid_test / base_eval_ood_test (evaluate, real vLLM)

The legacy `current.jsonl` prototype manifest stayed 0 bytes; no runner writes a
JSONL manifest anymore. All stage records live in the typed
`processes/coordinator/events.jsonl`.

### Reduced (`MODE=reduced NGPU=8 run_full_pilot.sh`)

Attempt bundle `results/runs/20260813T223701Z-reduced/attempt-01/`. Fourteen
stages, all `return_code: 0`, including `build_report_input` (the stage that
first exposed the report-side gap; see below). The built-in `preflight-reduced`
gate passed:

- calibration selected (pass@1 0.02 in [0.0, 0.2]; pass@32 0.505 in [0.35, 0.7])
- train matched coverage 474 (>= 300)
- dev matched coverage 151 (>= 100)

All five report-input checks passed: `required_manifest_stages_succeeded`,
`split_registry_selected`, `split_registry_no_overlap`, `base_summaries_present`,
`adapter_matrix_selected`.

## The one bug this surfaced, and the fix

The last stage, `build_report_input`, failed `rc=1` on the first reduced run:
`report input validation failed: required_manifest_stages_succeeded`. Root
cause was a migration follow-up gap, not an on-device failure: the report
builder derived stage success by reading the legacy JSONL manifest, which the
migration no longer writes, so the stage list was empty.

Fix (`b1a3e7b66`): `build_countdown_report_input` now reads stage outcomes from
the typed attempt bundle's coordinator event stream, reducing each stage's
terminal `stage_succeeded` / `stage_failed` event to the same
`{stage, return_code}` row the checks already consume. It falls back to the
legacy JSONL manifest when no attempt bundle exists so a pre-migration run still
reports. Verified by re-running `build-report-input` against the real reduced
event stream (exit 0, all checks green) and by a fresh single-shot reduced pilot
that completed all 14 stages cleanly.

## Results

Base model is Qwen3-1.7B. Adapters are LoRA SFT over the three reduced-mode arms
(raw, hindsight, curriculum). pass@1 and pass@32 are bootstrap means.

Full-split metrics (all dev/iid/ood problems):

- base   dev p@1 0.023 / p@32 0.500; iid 0.017 / 0.503; ood 0.090 / 0.650
- raw    dev 0.190 / 0.900; iid 0.207 / 0.917; ood 0.240 / 0.895
- hindsight dev 0.143 / 0.857; iid 0.113 / 0.873; ood 0.210 / 0.895
- curriculum dev 0.187 / 0.910; iid 0.193 / 0.923; ood 0.285 / 0.895

Base-elicitable dev bucket (143 problems the base model solves at p@32 but not
p@1 -- the promotion gate's target bucket):

- base       p@1 0.000 / p@32 1.000
- raw        p@1 0.280 / p@32 0.993
- hindsight  p@1 0.210 / p@32 0.965
- curriculum p@1 0.266 / p@32 0.986

The promotion bar was: at least one arm improves dev pass@1 by >= 5 absolute
points on the base-elicitable dev bucket while keeping dev pass@32 within 2
absolute points of base. All three arms clear it decisively: raw +28.0, hindsight
+21.0, curriculum +26.6 points on base-elicitable dev pass@1, with pass@32 held
at 0.965-0.993 vs base 1.000. This justifies the full pilot.

## Example (dev, raw arm, problem cd-1-31-9-t41: numbers [1, 31, 9], target 41)

Base sample 0 (fails, bucket elicitable): reasons at length but emits no FINAL
line -- "...The final result must be exactly 41. Let's start by analyzing the
numbers..." (error: missing FINAL line; first correct sample not until index 11).

Raw-adapter sample 0 (succeeds, bucket easy):
"...1 + 31 = 32. Then we have 32 and 9 left. 32 + 9 = 41. That works! ... FINAL:
41" (solved_at 1, strict_solved_at 1).

Canonical solution: `31 + 1 = 32`, `32 + 9 = 41`, `FINAL: 41`. The adapter
converts a rollout the base model could only reach with heavy sampling into a
first-sample strict success -- exactly the base-elicitable-to-easy promotion the
gate rewards.

## Infra contract confirmed

- Every real stage ran under the rootfs; `rootfs_active: true` on all events.
- Typed attempt bundle produced per run: `manifest.json` (frozen once,
  idempotent begin) plus `processes/coordinator/events.jsonl`.
- Report provenance now sources stage outcomes from the typed event stream, with
  legacy JSONL fallback for pre-migration runs.
- Unit coverage: `tests/unit_tests/test_countdown_run_common_lifecycle.py` (5) and
  `tests/unit_tests/test_countdown_search_distill.py` report-input tests (4,
  including two new event-stream tests) pass in-rootfs.
