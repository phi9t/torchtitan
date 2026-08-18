# Issue 06: Probe Recommendations

Status: open
Type: task
Blocked by: 04, 05

## Intent

Add recommendation-only active diagnostic planning for ambiguous estimator
outputs.

The system should say what evidence would best separate competing hypotheses,
not run the probe or alter the training job.

## Acceptance Criteria

- Define a probe recommendation schema with:
  - candidate probe;
  - competing hypotheses;
  - expected information target;
  - cost class;
  - risk class;
  - required authority;
  - live-safe versus stopped-job-only flag;
  - source evidence that motivated the recommendation.
- Implement deterministic mappings for at least:
  - network fault versus late rank arrival;
  - host/data stall versus compute stall;
  - sensor fault versus resource fault;
  - numerical corruption versus performance-only degradation;
  - checkpoint lineage uncertainty.
- Emit recommendations in `belief_summary.json` and `diagnosis.md`.
- Tests cover ambiguous and already-observable cases. If the existing evidence
  is enough to distinguish the cause, the analyzer should not recommend an
  unnecessary probe.

## Non-Goals

- Running probes.
- Selecting GPU allocations.
- Starting DCGM EUD, nccl-tests, Nsight, py-spy, replay, or canary workloads.
- Automatic retry, eviction, fencing, or rollback.

## Verification

Run:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_probes.py tests/unit_tests/observability/state_estimator/test_estimators.py'
```
