# Observability profiles and RSI report

Type: task
Status: complete
Blocked by: -

## Requirement

Implement ByteRobust/Mycroft/Argus/Eroica-style observability as schema-backed
advisory profiles and matrix-level RSI evidence.

## Scope

Allowed:

- extend matrix arm summaries with requested and effective observability
  profiles;
- classify observability features as `observed`, `missing`, `unavailable`,
  `stale`, or `advisory`;
- add semantic timeline extraction hooks for Mycroft-style completion/incident
  summaries using existing run/preflight/telemetry artifacts;
- add Argus-style diagnostic recommendations only as recommendations, never as
  automatic probe launches;
- add Eroica-style cross-arm report fields for training metrics, blockers,
  observability warnings, and recommendation quality inputs;
- preserve existing parse/summarize artifacts.

Excluded:

- no automatic retry, restart, rollback, quarantine, eviction, or recovery;
- no active diagnostics launched by the planner;
- no learned safety decisions;
- no TorchTitan core Trainer hooks.

## Verification Evidence

Run inside rootfs:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_run_experiment_matrix.py tests/unit_tests/test_modded_nanogpt_b200_summarize.py'
```
