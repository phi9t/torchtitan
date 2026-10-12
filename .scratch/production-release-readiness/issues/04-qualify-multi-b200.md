# Qualify TorchTitan on B200

Type: task
Status: ready-for-agent
Blocked by: 03, 07
Parent: ../spec.md

## Requirements

- Run the exact human-approved multi-B200 debug-training and distributed
  contract from ticket 07 in bwrap without changing its inputs or thresholds.
- Record correctness, completion, device/driver compatibility, and sanitized
  run-attempt evidence for the exact candidate.

## Verification

- The signed B200 receipt verifies independently against the federation trust
  root and schema, and failure injection fails closed.

## Comments
