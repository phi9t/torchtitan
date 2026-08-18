# Issue 05: Fixed-Lag Incident Timeline

Status: open
Type: task
Blocked by: 03, 04

## Intent

Reconstruct a bounded causal timeline around an incident or failed attempt.

The timeline should insert delayed observations at event time where possible
and preserve uncertainty when clocks or semantic keys are missing.

## Acceptance Criteria

- Add a timeline reconstruction pass that selects a window by incident time,
  failed process outcome, or explicit step range.
- Use wall time for cross-process joins and monotonic time plus local sequence
  for process-local order.
- Represent event time and ingestion time separately when both are available.
- Emit timeline entries with entity references, phase, step, quality, and
  source artifact reference.
- `diagnosis.md` includes a compact incident timeline section.
- Tests cover delayed observation insertion and missing-clock warnings.

## Non-Goals

- Real-time streaming.
- Global clock synchronization.
- Dense per-millisecond state nodes.
- Changing event producer fields.

## Verification

Run:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_timeline.py'
```
