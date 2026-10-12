# Issue 02: Evidence Graph Fixtures

Status: open
Type: task
Blocked by: 01

## Intent

Create synthetic run-evidence fixtures for the offline state estimator. Tests
are host-side in scope but every Python command must run inside the TorchTitan
bwrap rootfs.

The fixture builder should make it cheap to test graph construction and
diagnosis without launching distributed training, requiring GPUs, or relying on
large checked-in artifacts.

## Acceptance Criteria

- Add a test fixture helper that creates minimal run-attempt bundles matching
  `docs/run_evidence.md` schema version 1.
- Cover at least:
  - one single-process successful attempt;
  - one two-rank attempt with phase events;
  - one attempt with malformed optional evidence;
  - one attempt with a failed process outcome.
- Fixture paths are temporary test directories, not committed generated
  bundles.
- Tests assert fixture shape using real file reads, not mocks of the future
  analyzer.

## Non-Goals

- Implementing the production graph builder.
- Adding GPU requirements.
- Adding fixture data for every artifact kind.

## Verification

Run the new focused fixture tests, for example:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_fixtures.py'
```
