# Issue 18: Learned Factor Ingestion And Feature Store

Status: resolved
Type: task
Blocked by: 13, 15

## Intent

Create the optional learned-factor ingestion path without adding ML dependencies
or letting learned output control safety decisions.

## Acceptance Criteria

- Add `torchtitan/observability/state_estimator/features.py`.
- Extend `learned.py` with model identity, factor bundle, calibration metadata,
  and source segment references.
- Extract scalar, event, trace-summary, communication, hardware, framework, and
  transaction feature segments from normalized observations.
- Load learned factor JSON bundles only when explicitly requested.
- Reject learned factor bundles that omit model identity, advisory flag,
  calibration state, training-data manifest, or source segment identity.
- Keep default analyzer imports free of torch, transformers, graph libraries,
  neural CDE/ODE packages, and GPU libraries.

## Non-Goals

- Training a learned model.
- Running learned inference by default.
- Calibrating learned scores.
- Using learned factors for transaction or recovery decisions.

## Subagent Trace Requirement

Use fresh implementer, reviewer, and finalizer agents. The finalizer must check
dependency boundaries and whether learned metadata is sufficient for later
calibration review.

## Verification

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_learned.py'
```

## Execution Notes

- Implementer: `01a015d4-a151-7cf2-b16a-572b1877db85` (`Bernoulli`).
- Reviewer: `01a015db-6577-7381-9454-bee18074ef79` (`Aristotle`), requested
  changes because valid advisory bundles could not cite transaction source
  segments.
- Fix implementer: `01a015de-299c-77c0-9c14-fceffdf933d7` (`Locke`).
- Scoped re-reviewer: `01a015e2-cf6e-70a2-b404-f85e69be804a` (`Gibbs`),
  approved the fix and verified `22 passed in 0.19s`.
- Finalizer: `01a015e5-7a53-7283-bfdd-1abfb5fa4c75` (`Peirce`), accepted with
  no rerun required.
- Verification: `22 passed in 0.17s` for the required rootfs command.
- Finalizer note: transaction as provenance is allowed; transaction as
  authority or control output remains forbidden.
