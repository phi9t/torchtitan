# Issue 17: Active Diagnostic Planner

Status: resolved
Type: task
Blocked by: 15, 16

## Intent

Expand probe recommendations from one hard-coded ambiguity to a scored,
authority-aware active diagnostic planner.

## Acceptance Criteria

- Extend `ProbeRecommendation` with target entities, expected information gain
  class, stopped-job requirement, preconditions, expected artifacts, and source
  evidence.
- Recommend across these probe families:
  - local compute canary;
  - isolated collective;
  - same-microbatch replay on another GPU;
  - same-data replay under different placement;
  - independent sensor query;
  - stack snapshot;
  - profiler window;
  - Flight Recorder dump preservation;
  - checkpoint lineage validation;
  - BF16/FP32 replay for numerical instability;
  - stopped-job `nccl-tests`;
  - stopped-job SuperBench or EUD; and
  - prior-code-version replay.
- Include cost, risk, required authority, live-safe flag, and expected artifact
  outputs for every recommendation.
- Rank recommendations deterministically by ambiguity reduction class, risk,
  cost, and source evidence.
- Planner never runs probes.

## Non-Goals

- GPU allocation selection.
- Starting diagnostics.
- Rerouting, fencing, rollback, eviction, or retry.

## Subagent Trace Requirement

Use fresh implementer, reviewer, and finalizer agents. The finalizer must check
that no probe recommendation can be mistaken for automatic action and that
stopped-job probes require explicit authority.

## Verification

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_probes.py'
```

## Execution Notes

- Implementer: `01a015c8-2554-7402-83ca-9d001edfe887` (`James`).
- Reviewer: `01a015cf-98e6-7550-830f-8f24e047fab9` (`Archimedes`), approved with no blocking findings.
- Finalizer: `01a015d1-53eb-76e2-a35f-cdc64c3e0076` (`Carson`), accepted with no rerun required.
- Verification: `7 passed in 0.13s` for the required rootfs command.
- Finalizer note: duplicate recommendation merging is acceptable for current
  same-authority families, but future mixed-authority duplicates should merge to
  the stricter authority and highest risk.
