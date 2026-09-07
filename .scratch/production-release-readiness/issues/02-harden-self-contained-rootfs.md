# Harden the self-contained TorchTitan rootfs

Type: task
Status: needs-info
Blocked by: 01
Parent: ../spec.md

## Requirements

- Remain `needs-info` until Ultron's parent ticket records the resolved Vaso
  `02-freeze-rootfs-spec-v1` digest and changes this ticket's status.

- Pin every base image and tool input; bind identity to the complete recipe and
  filesystem content.
- Enforce the Vaso v1 lifecycle with repository-owned code and repo-local state.
- Preserve TorchTitan's native rootfs entrypoints and execution boundary.

## Verification

- Visible red-green slices cover materialization, selection, bwrap planning,
  execution, and fail-closed identity checks.

## Comments
