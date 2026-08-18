# Issue 09: First End-to-End Offline Analyzer

Status: open
Type: task
Blocked by: 03, 04, 05, 06, 08

## Intent

Package the first usable command-line analyzer that runs the graph builder,
deterministic estimators, incident timeline, probe recommendations, and
evaluation-friendly output in one offline command.

This is the first integration ticket after the individual pieces exist.

## Acceptance Criteria

- Add a CLI entrypoint for offline analysis of a run-attempt evidence bundle.
- The command writes all derived artifacts under
  `derived/state_estimator/` by default.
- The command is idempotent or has an explicit overwrite policy.
- The command exits nonzero for malformed required evidence and writes a clear
  failure report when possible.
- The command does not require GPU, distributed launch, or optional ML
  dependencies.
- All Python execution for the analyzer, tests, compile checks, and CLI
  invocation goes through `scripts/rootfs/enter_rootfs.sh` before running
  Python.
- If verification uncovers missing Python packages, add them inside rootfs with
  `uv`. If it uncovers missing non-Python runtimes, compilers, CLIs, or tools,
  add or pin them with `mise` unless a rootfs provisioning script already owns
  them.
- Execute the task through the subagent orchestration loop: fresh implementer,
  fresh reviewer, fresh finalizer reviewing both traces, then orchestrator
  integration or task rerun.
- The command can analyze at least one real small run-evidence bundle produced
  by core training or a representative fixture.

## Non-Goals

- Online dashboard.
- Streaming ingestion.
- Control-plane integration.
- Automatic actuation.

## Verification

Run:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_cli.py'
```

If a small real bundle is available, also run the CLI against it and record the
output path in this issue's comments.

## Completion Evidence

- `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator'`
  - Result: `24 passed in 0.32s`.
- `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/test_run_evidence.py tests/unit_tests/observability/test_structured_logging.py'`
  - Result: `141 passed in 20.19s`.
- `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m py_compile torchtitan/observability/state_estimator/*.py'`
  - Result: exit code `0`.
- `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_cli.py::test_cli_runs_against_fixture_bundle'`
  - Result: `1 passed in 0.21s`.
- `git diff --check -- torchtitan/observability/state_estimator tests/unit_tests/observability/state_estimator .scratch/training-state-estimator docs/research/2026-08-18-robotics-state-estimation-training-jobs.md`
  - Result: exit code `0`.

The analyzer remains offline and advisory. It does not alter raw evidence,
training control flow, optimizer commit behavior, checkpoint validity,
membership, or recovery policy.
