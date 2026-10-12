# Inventory TorchTitan release work

Type: task
Status: ready-for-agent
Blocked by:
Parent: ../spec.md

## Requirements

- Fingerprint every branch, worktree, dirty tree, submodule, and publication
  object without recording raw sensitive values.
- Establish the trusted public-upstream boundary and assign every local item an
  integrate, superseded, retain-local, or reject disposition.
- Isolate rootfs release work from unrelated training experiments.

## Verification

- Coverage and recovery pointers are independently reviewed; no existing work
  is moved, rewritten, or deleted.

## Comments
