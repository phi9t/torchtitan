# Issue 10: V1 Analyzer Hardening

Status: resolved
Type: task
Blocked by: 09

## Intent

Close the gaps between the current offline analyzer and the original v1
contract before building the deeper topology and inference layers.

The current analyzer emits `phase_durations=[]`, does not wire failed-attempt
timeline center selection into `write_belief_summary`, and has limited entity
coverage for fields already present in v1 run-evidence rows.

## Acceptance Criteria

- Add real phase-duration summaries grouped by phase, step, and process/rank
  when structured events include start/end or enough paired timing fields.
- If exact durations cannot be computed, emit a quality finding explaining
  which fields are missing instead of returning an unexplained empty list.
- Wire failed-attempt timeline center selection so a failed outcome can produce
  an incident timeline without the caller passing an explicit center.
- Preserve monotonic time, event sequence, and artifact sequence in derived
  timeline rows when those fields exist.
- Expand graph entities and attributes for host, rank, role, actor, device,
  phase, and step when present in existing v1 evidence rows.
- Keep all outputs advisory and backwards-compatible with existing v1 tests.

## Non-Goals

- Multiplex physical/logical/control graph layers.
- New core evidence producers.
- External hardware/tool ingestion.
- Robust mode inference.

## Subagent Trace Requirement

Use fresh implementer, reviewer, and finalizer agents. The finalizer must check
that the implementation truly closes the v1 gaps and does not start Task 11+
scope early.

## Verification

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_graph.py tests/unit_tests/observability/state_estimator/test_timeline.py tests/unit_tests/observability/state_estimator/test_estimators.py tests/unit_tests/observability/state_estimator/test_cli.py'
```

## Comments

### 2026-08-18 Issue 10 execution

- Implementer: `01a01543-79ce-7b01-a723-e4b0ff0364e0` (`Helmholtz`).
- Reviewer: `01a0154a-25cf-7992-92b0-687dfa1f12c3` (`Beauvoir`), requested changes for skipped missing-duration warnings and incomplete device metadata.
- Fix implementer: `01a0154c-e822-7e82-be2f-a036fcda00b2` (`Epicurus`).
- Scoped re-reviewer: `01a01550-b563-71c0-8658-153bc51ae336` (`Ramanujan`), approved the two fixes.
- Finalizer: `01a01553-3377-7903-9f12-e1b45f333021` (`Herschel`), accepted Issue 10.
- Verification: `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_graph.py tests/unit_tests/observability/state_estimator/test_timeline.py tests/unit_tests/observability/state_estimator/test_estimators.py tests/unit_tests/observability/state_estimator/test_cli.py'` -> `17 passed in 0.24s`.
- Finalizer notes: no rerun required; future briefs should include explicit negative-test expectations for reviewer findings and clarify where estimator quality findings are persisted.
