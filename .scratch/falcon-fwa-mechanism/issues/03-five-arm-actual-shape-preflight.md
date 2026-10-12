# 03 - Certify the five-arm fixed-region actual-shape preflight

Type: task
Status: complete
Blocked by: 02
External gate: Campaign B 11 - fixed-region LM evaluator (human-enforced)
Parent: ../spec.md

## What to build

Certify that all five mechanism arms learn, remain numerically healthy, emit
valid evidence, and fit the declared screen budget at the real science shape.

## Acceptance criteria

- [x] The affected-arm fixed-sequence overfit gate is rerun after the mixer
  changes; loss collapses and the learning-rate-zero control fails to learn.
- [x] Meta-initialization, finite forward/backward, and recurrent/masked-parallel
  guards cover every affected mixer configuration.
- [x] Every arm executes a few optimizer steps at four layers, hidden size 256,
  sequence length 512, and 16,384 tokens per optimizer step using matched seeds
  and data order.
- [x] M2 and M4 satisfy their same-input/same-weight invariants; every
  non-equivalent arm pair differs by more than `1e-6` in full-precision loss or
  gradient norm on at least one post-update step, and every trajectory is
  finite.
- [x] Fixed-region evaluation and all predeclared diagnostic probe milestones
  can be produced from the actual-shape runner without a shrunk-model shortcut.
- [x] The preflight ledger projects total screen GPU-hours by summing device
  wall time and includes failures/retries, evaluation, diagnostics, and
  checkpoint writing.
- [x] Disk preflight records free space, projected peak bytes, final model-only
  checkpoint retention, and a safety reserve.
- [x] A validator emits a binary eligible/ineligible screen decision; projected
  cost above one B200-hour or incomplete evidence is ineligible.
- [x] Focused rootfs validator tests cover missing seeds/regions, shrunk versus
  actual-shape evidence, directional and practical-equivalence boundaries,
  exact/elevated compute ceilings, retry accounting, invalid attempt bundles,
  and explicit omission records before any screen spend.

## Exclusions

- No 2,000-step screen and no relaxation of the one-B200-hour ceiling.
- No inference from shrunk dry-runs or partially completed arms.

## 2026-09-08 implementation status

- Implemented the M03 `mechanism_screen_eligibility` validator and campaign
  preflight entrypoint.
- Re-ran the affected-arm fixed-sequence overfit gate after the mixer changes.
  Falcon-1A and Falcon-1 learning runs collapsed (`3.4648 -> 0.0110` and
  `3.4680 -> 0.0109` CE), while both `lr=0` controls failed to learn
  (`3.4648 -> 3.4648` and `3.4680 -> 3.4680` CE).
- Ran the focused rootfs validator suite:
  `python -m py_compile experiments/falcon/campaign_driver.py torchtitan/experiments/falcon/promotion.py`
  plus `pytest -q tests/unit_tests/test_falcon_promotion.py
  tests/unit_tests/test_falcon_mechanism.py tests/unit_tests/test_falcon_lm_eval.py
  tests/unit_tests/test_falcon_evidence.py`; final result after review fixes:
  79 passed.
- Ran the full Falcon unit suite after review fixes; result: 168 passed.
- Ran pre-commit on the M03 code/test paths; all hooks passed.
- Ran the rootfs M03 CLI smoke:
  `experiments/falcon/run.sh mechanism-preflight --master-port 29730`.
  It completed all M0-M4 arms for 3 optimizer steps at the exact science shape
  and produced fixed-region evals, probes, model-only checkpoints, and a binary
  decision artifact.
- Decision artifact:
  `experiments/falcon/results/mechanism_preflight/mechanism-screen-preflight-seed0-a85c12e807e7/mechanism_screen_eligibility.json`.
- Native evidence bundle:
  `experiments/falcon/results/mechanism_preflight/evidence/runs/falcon-mechanism-screen-preflight-M0-M4-seed0/mechanism-screen-preflight-seed0-a85c12e807e7/`.
- Frozen decision: `ineligible`.
  Numerical/probe/eval/compute checks passed, but disk accounting failed the
  reserve gate: free bytes `5648375808`, required bytes `9315971690`.
  The native attempt status is `completed`; the gate result is recorded
  separately as `promotion_decision: ineligible`.

M03 is complete and emits a binary `ineligible` decision. M04 must not launch
until disk capacity/reserve accounting is resolved or a new human budget
decision changes the constraint.
