# Issue 16: Transaction And Checkpoint Advisory State

Status: resolved
Type: task
Blocked by: 15

## Intent

Represent checkpoint lineage, data position, process-group epoch, replica epoch,
and transaction-risk evidence without changing commit or checkpoint behavior.

## Acceptance Criteria

- Add `torchtitan/observability/state_estimator/transactions.py`.
- Represent committed step, speculative step, process-group epoch, replica
  epoch, checkpoint parent, checkpoint shard inventory, RNG/dataloader/data
  cursor evidence, save/load/staging status, and restore validation status when
  evidence exists.
- Emit transaction-risk advisory fields that distinguish:
  - enough evidence to assess risk;
  - missing transaction evidence;
  - anomaly before a likely commit boundary;
  - anomaly after a likely commit boundary;
  - checkpoint lineage incomplete; and
  - checkpoint restore validation absent.
- Keep all transaction and checkpoint conclusions advisory.
- Add diagnosis/report text that explains what evidence is missing before any
  transaction-safety claim could be made.

## Non-Goals

- Calling TorchFT `should_commit()`.
- Changing optimizer, checkpoint, dataloader, process-group, or replica state.
- Validating DCP internals directly.

## Subagent Trace Requirement

Use fresh implementer, reviewer, and finalizer agents. The finalizer must check
that no wording or code path treats advisory transaction risk as a control
decision.

## Verification

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_transactions.py tests/unit_tests/observability/state_estimator/test_estimators.py'
```

## Execution Notes

- Implementer: `01a015bc-4626-7dd3-9c1e-2eaba9957d55` (`Ampere`).
- Reviewer: `01a015c2-2453-7272-92d9-9336ea524d52` (`Arendt`), approved with no blocking findings.
- Finalizer: `01a015c4-96bf-72a2-8f04-518ebd8fcbae` (`Parfit`), accepted with no rerun required.
- Verification: `18 passed in 0.18s` for the required rootfs command.
- Finalizer note: anomaly placement is step-based relative to `committed_step`;
  it remains a heuristic timing advisory, not a transaction proof.
