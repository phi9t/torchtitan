# Issue 03: Evidence Graph Builder

Status: open
Type: task
Blocked by: 01, 02

## Intent

Implement the first offline evidence graph builder over existing run-attempt
bundles.

The builder reads raw evidence and writes deterministic derived artifacts
without mutating raw records.

## Acceptance Criteria

- Add an analyzer module under an appropriate non-core location.
- Input: one run-attempt evidence bundle.
- Output:
  - `derived/state_estimator/evidence_graph.json`;
  - `derived/state_estimator/entities.json`;
  - `derived/state_estimator/timeline.jsonl`;
  - `derived/state_estimator/quality.json`.
- The graph includes run, attempt, process, rank, host, role, actor, artifact,
  event stream, outcome, phase, step, and incident entities when available.
- The builder preserves local ordering through process-local sequences and
  monotonic time where available.
- Missing optional evidence becomes a quality finding, not a crash.
- Missing required manifest or invalid core identity fails with a clear error.

## Non-Goals

- Root-cause scoring.
- Learned models.
- Online collection.
- Changing the run-evidence schema.

## Verification

Run:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_bundle.py tests/unit_tests/observability/state_estimator/test_graph.py'
```
