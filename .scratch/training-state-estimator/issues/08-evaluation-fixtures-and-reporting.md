# Issue 08: Evaluation Fixtures and Reporting

Status: open
Type: task
Blocked by: 04, 05, 06

## Intent

Add the first evaluation harness for the offline estimator using synthetic and
small captured evidence bundles.

The goal is to measure detector behavior and localization quality separately
from training performance claims.

## Acceptance Criteria

- Define an evaluation manifest format for labeled evidence bundles.
- Include labels for:
  - injected or synthetic cause;
  - onset time or step;
  - affected scope;
  - expected diagnostic class;
  - expected safe probe class.
- Report:
  - detection latency;
  - correct failure-domain rate;
  - root-cause top-1/top-k where labels exist;
  - false alert count on nominal fixtures;
  - missing-evidence observability warnings.
- Emit a compact Markdown report and JSON summary.
- Tests use small fixtures only and run their Python process inside the
  TorchTitan bwrap rootfs.

## Non-Goals

- Large benchmark corpus.
- GPU fault injection.
- Learned-model calibration.
- System utility claims such as tokens per GPU-second.

## Verification

Run:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_evaluation.py'
```
