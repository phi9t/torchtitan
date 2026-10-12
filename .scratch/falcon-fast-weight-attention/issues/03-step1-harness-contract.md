# Step 1 remainder: harness contract

Type: task
Status: resolved
Blocked by:
Parent: ../spec.md

## Requirements

- Add an explicit `alignment` switch on the Falcon kernel: `delayed`
  (default, paper) vs `same_step` (ablation only).
- Keep read-after-write, `eta_0 = 0` for delayed, QK-RMSNorm default.
- Tests: delayed first write uses `k_0` with `v_1`; same-step first
  write uses `k_0` with `v_0`; label contract is shift not roll.
- Short harness note under `torchtitan/experiments/falcon/` listing
  the silent-failure modes from the spec (same-step default, circular
  labels, missing doc reset later).
- Re-run the existing Step 3 overfit gate; it must stay green on the
  delayed default.

## Exclusions

- Masked-parallel / chunk-parallel kernels
- FineWeb, addition, science-scale model
- Edits to Mini-K3 or `qwen3_5`

## Verification

- New unit tests fail before the switch exists, then pass.
- `pytest` on all `test_falcon_*.py` in rootfs is green.
- Claim label: `deterministic_pairing` + `learnability_smoke`.

## Answer

`alignment` is on `falcon_recurrent_forward` and `FalconConfig`, default
`delayed`. Same-step is ablation-only. Harness note:
`torchtitan/experiments/falcon/HARNESS.md`. Rootfs `test_falcon_*.py`:
17 passed. Report: `sdd/task-03-report.md`.
