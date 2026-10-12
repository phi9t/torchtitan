# Prove TorchTitan rootfs conformance and native tests

Type: task
Status: ready-for-agent
Blocked by: 02
Parent: ../spec.md

## Requirements

- Run Vaso v1 behavior-equivalent conformance, negative-path, clean-bootstrap,
  and repository-native focused and broader tests inside bwrap.
- Prove native commands cannot silently bypass the rootfs boundary.

## Verification

- Fresh machine-readable results bind to the exact commit and rootfs digest.

## Comments
