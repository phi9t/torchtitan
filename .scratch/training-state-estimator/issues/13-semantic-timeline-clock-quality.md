# Issue 13: Semantic Timeline And Clock Quality

Status: resolved
Type: task
Blocked by: 12

## Intent

Build bounded semantic timelines that join observations by event time, local
sequence, step, phase, entity, and topology epoch while surfacing clock-quality
limits.

## Acceptance Criteria

- Extend `timeline.py` with semantic timeline entries and window selection.
- Support incident-centered, explicit time, explicit step, and last-N-step
  windows.
- Insert delayed observations at event time when event time exists.
- Use monotonic time and process-local sequence as local-order fallback.
- Emit warnings for missing clocks, weak cross-process ordering, and conflicting
  sequence/time order.
- Include phase, step, entity, source path, source record, and relative time in
  timeline entries.
- Provide compact incident summaries for Markdown reports.

## Non-Goals

- Global clock synchronization.
- Streaming timeline updates.
- Millisecond state nodes for every resource.

## Subagent Trace Requirement

Use the implementer/reviewer/finalizer loop. The finalizer must explicitly
check whether delayed observations, missing clocks, and local-order fallback
were tested.

## Verification

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_timeline.py'
```

## Comments

### 2026-08-18 Issue 13 execution

- Implementer: `01a0157f-b1c5-7452-879a-fa6d52b50095` (`Kierkegaard`).
- Reviewer: `01a01587-9697-74e3-90cc-aee6e314bf88` (`Aquinas`), approved.
- Finalizer: `01a01589-679b-7992-8872-f7cd2e75204e` (`Newton`), accepted Issue 13.
- Verification: `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_timeline.py'` -> `9 passed in 0.15s`.
- Finalizer checks: delayed observation ordering, missing clocks, local-order fallback, conflicting sequence/time warnings, all window modes, entry fields, compact summary, and legacy behavior were covered.
