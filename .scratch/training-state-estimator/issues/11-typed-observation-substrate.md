# Issue 11: Typed Observation Substrate

Status: resolved
Type: task
Blocked by: 10

## Intent

Normalize raw run-evidence records into typed state-estimator observations with
explicit identity, clock, source, quality, and raw-record references.

This is the substrate for topology, timeline, analytical modeling, inference,
learned feature extraction, and evaluation.

## Acceptance Criteria

- Add `torchtitan/observability/state_estimator/observation.py`.
- Define observation kind, clock quality, observation envelope, and normalized
  observation schemas.
- Normalize structured event rows, artifact rows, and process outcomes from
  `RunEvidenceBundle`.
- Preserve raw source path, record type, process identity, sequence fields, and
  raw record identity.
- Distinguish `event_time_ns`, `ingestion_time_ns`, `monotonic_ns`,
  `event_seq`, and `artifact_seq`.
- Emit quality findings for missing sensor identity, missing clocks, malformed
  identity fields, and non-object raw records.
- Keep analyzer imports free of torch, graph libraries, neural libraries, and
  GPU libraries.

## Non-Goals

- Physical topology inference.
- Fault-mode scoring.
- Learned model execution.
- Core `Trainer` instrumentation.

## Subagent Trace Requirement

Execute with a fresh implementer subagent, fresh reviewer subagent, and fresh
finalizer subagent. The finalizer must decide whether the task needs a rerun
and propose any improvements to this issue, the plan, or task-specific skills.

## Verification

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_observation.py tests/unit_tests/observability/state_estimator/test_bundle.py'
```

## Comments

### 2026-08-18 Issue 11 execution

- Implementer: `01a01556-b13d-71a3-9bba-9455cb8096c3` (`Confucius`).
- Reviewer: `01a0155c-4d64-7711-8cba-6f609e1e1b27` (`Halley`), requested changes for nondeterministic malformed-field ordering and duplicate malformed `event_time_ns` findings.
- Fix implementer: `01a0155f-a635-7673-80e6-71b85f12b161` (`Ptolemy`).
- Scoped re-reviewer: `01a01562-c7d9-7e11-b669-3fe0871331df` (`Pauli`), approved the two fixes.
- Finalizer: `01a01565-0341-7422-a93e-95a2658f48a3` (`Cicero`), accepted Issue 11.
- Verification: `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_observation.py tests/unit_tests/observability/state_estimator/test_bundle.py'` -> `9 passed in 0.16s`.
- Finalizer note for Issue 12+: `NormalizedObservation.id` can collide for malformed or missing `process_id` when rows share sequence fields. `envelope.raw_record_id` preserves provenance, but graph consumers should specify uniqueness or include raw record identity in graph node IDs.
