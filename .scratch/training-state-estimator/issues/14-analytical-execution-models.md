# Issue 14: Analytical Execution Models

Status: resolved
Type: task
Blocked by: 13

## Intent

Add deterministic analytical models for phase duration, max-plus critical path,
compute/memory/checkpoint residuals, and collective arrival-skew versus network
progress ambiguity.

## Acceptance Criteria

- Add `torchtitan/observability/state_estimator/analytical.py`.
- Compute phase duration summaries by process, rank, phase, and step.
- Represent max-plus predecessor relationships where evidence identifies them.
- Predict ring all-reduce duration from message bytes, group size, latency, and
  effective bandwidth.
- Split collective symptoms into arrival skew and network progress when launch
  and completion states exist.
- Add checkpoint staging/save/load duration summaries when checkpoint rows
  exist.
- Emit normalized residuals with explicit heuristic calibration and source
  evidence references.
- Avoid claiming a root cause from a residual alone.

## Non-Goals

- Full roofline FLOP accounting for every model.
- GPU kernel parser.
- Online critical-path tracking.
- Learned residual prediction.

## Subagent Trace Requirement

Use fresh implementer, reviewer, and finalizer agents. The finalizer must check
whether each analytical output is workload-conditioned when required and whether
fallback behavior is explicit when workload metadata is absent.

## Verification

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_analytical.py tests/unit_tests/observability/state_estimator/test_estimators.py'
```

## Comments

### 2026-08-18 Issue 14 execution

- Implementer: `01a0158f-b0d3-7321-831e-9c91b30151a7` (`Boole`).
- Reviewer: `01a01594-ee1b-7ee0-a8b1-0bd35bc886bb` (`Leibniz`), requested changes for unreachable checkpoint artifact summaries and missing residual sigma metadata.
- Fix implementer: `01a01597-9225-76e0-b06b-e553a4d0a257` (`Godel`).
- Scoped re-reviewer: `01a0159c-7982-7c80-8cf6-a1a48a0bba00` (`Socrates`), approved the two fixes.
- Finalizer: `01a0159f-0b77-73a2-9336-acafb419fae6` (`Plato`), accepted Issue 14.
- Verification: `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_analytical.py tests/unit_tests/observability/state_estimator/test_estimators.py'` -> `13 passed in 0.18s`.
- Finalizer note: future briefs should explicitly list which outputs are workload-conditioned predictions and which are evidence-only summaries.
