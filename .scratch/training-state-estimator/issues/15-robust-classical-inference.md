# Issue 15: Robust Classical Inference

Status: resolved
Type: task
Blocked by: 14

## Intent

Convert analytical residuals and evidence quality into robust, advisory mode
scores, sensor-health states, observability warnings, and root-cause
candidates.

## Acceptance Criteria

- Add `torchtitan/observability/state_estimator/inference.py`.
- Define mode scores for nominal, compute degradation, network degradation,
  host/data stall, collective desynchronization, memory fault, numerical
  corruption, storage fault, control-plane fault, planned intervention, failed,
  and recovering.
- Use robust residual scoring so one outlier cannot dominate the mode score.
- Add CUSUM-style persistent degradation scores for repeated weak residuals.
- Represent sensor-health states: healthy, delayed, biased, stuck, and missing.
- Emit root-cause candidates with affected scope and evidence chains.
- Emit observability warnings when available evidence cannot distinguish
  hypotheses.
- Mark every score with calibration state.

## Non-Goals

- Online iSAM2 or dense factor-graph solver.
- Calibrated probabilities without evaluation evidence.
- Automatic actuation.

## Subagent Trace Requirement

Use fresh implementer, reviewer, and finalizer subagents. The finalizer must
review whether mode labels are overclaiming relative to available evidence and
whether observability warnings are present for ambiguous cases.

## Verification

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_inference.py tests/unit_tests/observability/state_estimator/test_estimators.py'
```

## Comments

### 2026-08-18 Issue 15 execution

- Implementer: `01a015a3-9272-7813-b15a-07c2d4493218` (`Hume`).
- Reviewer: `01a015ac-256c-7723-8555-c9755bc6e63e` (`Mill`), requested changes for label-specific robustness, broad ambiguity warnings, failed-outcome integration, and sensor-health overclaiming.
- Fix implementer: `01a015ae-752b-7182-8f9e-eb8d847c4f34` (`Noether`).
- Scoped re-reviewer: `01a015b5-6c2a-7fb3-9b29-d9ebacec422c` (`Bacon`), approved the four fixes.
- Finalizer: `01a015b6-a1ff-7253-8063-5de713dbbeb5` (`Kepler`), accepted Issue 15.
- Verification: `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_inference.py tests/unit_tests/observability/state_estimator/test_estimators.py'` -> `20 passed in 0.18s`.
- Finalizer note: thresholds and mode separation remain heuristic and uncalibrated by design until the later evaluation/calibration tickets.
