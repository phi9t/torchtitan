# Training State Estimator Evaluation Manifest Schema

Status: draft
Date: 2026-08-18

The evaluation manifest describes labeled state-estimator cases. It is a small,
repo-local corpus format for reporting detection, localization, probe/action,
observability, resource, overhead, and calibration metrics separately.

## Top-Level Fields

- `schema_version`: integer schema version. Use the state-estimator
  `SCHEMA_VERSION`.
- `id`: stable manifest identifier.
- `description`: optional human-readable description.
- `score_label_families`: object whose keys are score-family names and whose
  boolean values declare whether labels are present for calibration. A score
  family must not be reported as calibrated when this value is absent or false.
- `cases`: array of evaluation cases.

## Case Fields

Required fields:

- `id`: stable case identifier, unique within the manifest.
- `case_class`: one of `synthetic`, `cpu_fake_backend`, `real_small_run`, or
  `injected_fault`.
- `attempt_path`: path to the run-attempt evidence bundle or fixture.
- `nominal`: boolean. True means any non-empty fault hypothesis is a false
  alert.
- `expected_mode`: expected diagnostic mode, or null for nominal cases.
- `expected_probe`: expected advisory probe kind, or null when no probe is
  expected.

Optional labels:

- `onset_time_ns`: event time for the injected or observed fault onset.
- `affected_scope`: labeled failure domain, such as `processes`, `hosts`,
  `devices`, `steps`, `phases`, or mesh-axis fields.
- `expected_root_cause`: expected root-cause candidate label. Defaults to
  `expected_mode` when omitted.
- `expected_warning_kinds`: observability warning kinds expected for the case.
- `expected_artifact_bytes`: expected derived-artifact byte count when known.
- `overhead_metrics`: numeric overhead labels, such as
  `runtime_overhead_pct` or `tier0_overhead_pct`.
- `real_training_labels`: true only when the case has real training labels
  that can support system-utility claims.
- `injected_fault`: injected fault class for `injected_fault` cases.

## Metric Sections

Evaluation reports keep claims separated:

- Detection: case count, detected count, false alerts, expected mode found, and
  detection latency.
- Localization: top-1 and top-k root-cause hits plus failure-domain scope
  score.
- Probe/action: expected probe found.
- Observability: expected warning correctness and warning counts.
- Resource and overhead: artifact bytes plus optional overhead metric
  summaries.
- Calibration: Brier score and expected calibration error only for score
  families with manifest labels. Synthetic-only labeled score families must use
  an explicit synthetic state such as `synthetic_calibrated`, not generic
  `calibrated`.

Calibration is not implied by high scores, synthetic labels, or detection
accuracy. A report must refuse calibrated status for any score family missing
labels in `score_label_families`. A report must also preserve calibration label
case-class provenance so synthetic-only metrics cannot be mistaken for real-run
calibration evidence.
