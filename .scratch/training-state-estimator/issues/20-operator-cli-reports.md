# Issue 20: Operator CLI And Reports

Status: resolved
Type: task
Blocked by: 19

## Intent

Make the offline analyzer usable as an operator tool over one attempt, multiple
attempts, and labeled evaluation manifests.

## Acceptance Criteria

- Add `torchtitan/observability/state_estimator/reports.py`.
- Extend CLI with subcommands:
  - `analyze-attempt`;
  - `analyze-run`;
  - `compare-attempts`; and
  - `evaluate`.
- Support writing derived artifacts to a separate output directory without
  modifying raw evidence.
- Emit JSON stdout with all derived artifact paths.
- Produce Markdown reports with sections for evidence quality, topology,
  timeline, analytical residuals, mode scores, root-cause candidates,
  transaction risk, probes, learned factors, and evaluation metrics.
- Exit nonzero with actionable JSON for malformed required evidence.
- Preserve the existing simple CLI behavior through a compatibility path or
  documented migration.

## Non-Goals

- Web dashboard.
- Remote exporter.
- Streaming monitor.
- Running active probes.

## Subagent Trace Requirement

Use fresh implementer, reviewer, and finalizer agents. The finalizer must check
CLI backward compatibility, raw-evidence immutability, and report usefulness.

## Verification

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_reports.py tests/unit_tests/observability/state_estimator/test_cli.py'
```

## Execution Notes

- Implementer: `01a015fc-a067-7e11-b168-3c446d6cfc8c` (`Singer`).
- Reviewer: `01a01605-bdab-79d0-bb33-2bea9c2801d5` (`Franklin`), requested
  changes because `compare-attempts` collapsed attempts with the same
  `attempt_id` across different runs.
- Fix implementer: `01a0160a-a74d-7553-b7d4-34de67055618` (`Boyle`).
- Scoped re-reviewer: `01a01611-525c-74e0-8b96-45020c9b2abd` (`Lorentz`),
  approved the fix and verified `12 passed in 0.93s`.
- Finalizer: `01a01613-deda-72a3-aa0f-b48dd327b23a` (`Nash`), accepted with no
  rerun required.
- Verification: `12 passed in 0.90s` for the required rootfs command.
- Finalizer note: compare output keys now handle same-attempt IDs across runs;
  future work should define or sanitize artifact-safe identity keys if raw
  evidence IDs become arbitrary path-like strings.
