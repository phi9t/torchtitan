# Issue 19: Evaluation Corpus And Calibration

Status: resolved
Type: task
Blocked by: 15, 17, 18

## Intent

Build the evaluation manifest, metrics, and report format needed to distinguish
detection, localization, observability, probe, and calibration claims.

## Acceptance Criteria

- Extend `evaluation.py` with evaluation manifests, case classes, metric
  summaries, calibration metrics, and report writing.
- Add `.scratch/training-state-estimator/evaluation_manifest_schema.md`.
- Support synthetic, CPU/fake-backend, real small-run, and injected-fault case
  classes.
- Report detection latency, false alerts, expected mode found, expected probe
  found, top-k root cause, failure-domain scope score, observability-warning
  correctness, Brier score, expected calibration error, artifact bytes, and
  optional overhead metrics.
- Keep detection, localization, probe/action, and calibration claims separate.
- Refuse to mark a score calibrated unless the evaluation manifest has labels
  for that score family.

## Non-Goals

- Large benchmark corpus.
- GPU fault injection implementation.
- Training learned models.
- System utility claims without real training labels.

## Subagent Trace Requirement

Use fresh implementer, reviewer, and finalizer agents. The finalizer must check
that calibration claims are not inferred from synthetic-only cases unless the
metric explicitly says synthetic.

## Verification

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_evaluation.py'
```

## Execution Notes

- Implementer: `01a015e9-e52d-71f3-9dee-23fc2c9f48ed` (`Popper`).
- Reviewer: `01a015f0-8e29-7573-a742-4ec6496cea70` (`Anscombe`), requested
  changes because synthetic-only manifests could produce generic calibrated
  score-family output.
- Fix implementer: `01a015f2-2d09-7ab1-bd6d-eec247d4b37b` (`Hubble`).
- Scoped re-reviewer: `01a015f6-86d6-7ce1-9402-0a6505a2ed26` (`Descartes`),
  approved the fix and verified `9 passed in 0.15s`.
- Finalizer: `01a015f9-6a2a-7ec1-9948-f9847e4ec5ad` (`Planck`), accepted with
  no rerun required.
- Verification: `9 passed in 0.15s` for the required rootfs command.
- Finalizer note: `write_evaluation_report()` remains a compatibility path; use
  `evaluate_manifest()` for manifest-aware calibration semantics.
