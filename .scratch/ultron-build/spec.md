# Ultron TorchTitan Seam Analysis

Status: ready-for-agent
Parent spec: /data02/home/philip.yang/workspace/ultron/.scratch/ultron-build/spec.md
Parent ticket: /data02/home/philip.yang/workspace/ultron/.scratch/ultron-build/issues/07-torchtitan-run-evidence-spike.md

## Intent

Analyze where TorchTitan can emit Ultron run evidence and update-commit ledger
records without changing training semantics or blocking the training hot path.
TorchTitan is the first framework integration locus because it owns trainer
steps, distributed setup, checkpoint/training state, metrics, and run evidence.

This child tracker serves the Ultron parent build only. It does not replace
TorchTitan's repo-local observability contract or authorize implementation.

## Authority Boundary

This tracker does not authorize code edits, commits, pushes, pull requests,
large GPU jobs, live fault injection, optimizer changes, checkpoint behavior
changes, or distributed process-group behavior changes. Any TorchTitan code edit
or Git action requires explicit human authorization for that repo/action.

If implementation is later authorized, it must run in an isolated TorchTitan
worktree and preserve the dependency direction `experiments -> core`.

## Repo-Local Guidance

- Read `CONSTITUTION.md`, `AGENTS.md`, and `docs/agents/issue-tracker.md` before
  acting.
- Classify any future code work against TorchTitan's surfaces: core training,
  online RL, or repo-local research programs.
- For this effort, the expected surface is core training observability and run
  evidence, not online RL.
- Tier 0 observability must stay within TorchTitan's 1% median steady-state
  throughput regression budget if later implementation claims production
  hot-path safety.

## Required Evidence For This Child Tracker

- Current TorchTitan file:line evidence for possible run identity,
  topology-epoch, trainer-step, process-group-generation, active-replica-set,
  PREPARED/COMMITTED, checkpoint-generation, and run-evidence seams.
- A hot-path classification for each proposed emission point:
  `static-seam-only`, `fixture-proven`, or `performance-proven`.
- Explicit statement of whether the proposed seam performs synchronous IO,
  dynamic allocation, blocking communication, or cross-process coordination on
  the training hot path.
- If not measured, state that hot-path overhead is not performance-proven.

## Verification Gates

- Focused seam-analysis ticket is resolved with source citations.
- No production code is modified unless separately authorized.
- Any future implementation must include visible red-green behavior tests and
  ticket-level Standards/Spec review.
- Branch-level independent verification follows the Ultron parent verifier
  decision: prefer `roborev review --branch`, otherwise separate-context review.

## Child Tickets

- `issues/01-run-evidence-seam-analysis.md`
