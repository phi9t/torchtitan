# Issue 21: Evidence Producer Roadmap And Core Hooks

Status: resolved
Type: design
Blocked by: 13

## Intent

Specify which additional core evidence producers or structured-event fields are
needed to support the remaining estimator work, without making code changes to
core training in this ticket.

## Acceptance Criteria

- Update `docs/run_evidence.md` or a linked tracker doc with a producer roadmap.
- Specify minimum fields for topology, microbatch, collective, checkpoint, data
  cursor, process-group epoch, replica epoch, transaction, and incident
  evidence.
- For every proposed producer field, list:
  - owning component;
  - evidence tier;
  - overhead risk;
  - source event or artifact kind;
  - validation command;
  - fallback behavior when absent; and
  - child implementation ticket needed, if any.
- Keep this ticket documentation-only unless the user explicitly approves a
  child implementation ticket.
- Preserve the existing run-evidence v1 contract.

## Non-Goals

- Editing `Trainer` or run-evidence recorder code directly.
- Adding a central collector.
- Adding optional diagnostic dependencies to core.
- Changing training control flow.

## Subagent Trace Requirement

Use fresh implementer, reviewer, and finalizer agents. The finalizer must check
that proposed producer fields are minimal, tiered, and not disguised code
changes.

## Verification

```bash
git diff --check -- docs/run_evidence.md .scratch/training-state-estimator
```

## Execution Notes

- Implementer: `01a01618-8361-7cf3-bb3b-78869ce1e8a5` (`Huygens`).
- Reviewer: `01a0161e-b734-7b23-b82f-8f911dc54d9b` (`Hegel`), requested
  changes because some validation commands used `COMM_MODE=local_tensor` even
  though that path returns before `Trainer` construction.
- Fix implementer: `01a01621-7c4c-7803-8202-038aa50417aa` (`Chandrasekhar`).
- Scoped re-reviewer: `01a01624-28a6-7d00-9ab1-d250a98e5985` (`Ohm`),
  approved the fix and verified `git diff --check` exited 0.
- Finalizer: `01a01626-630b-7090-b0b4-72db40ea6ef6` (`Faraday`), accepted with
  no rerun required.
- Verification: `git diff --check -- docs/run_evidence.md
  .scratch/training-state-estimator` exited 0 with no output.
- Finalizer note: child implementation tickets should pin exact rootfs-wrapped
  module/config commands before execution.
