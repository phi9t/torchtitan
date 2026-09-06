# Tiny Trainer overfit gate

Type: task
Status: resolved
Blocked by: 01
Parent: ../spec.md

## Requirements

- Tiny Falcon LM (2 layers) + repeating 8-sequence bank.
- `run_overfit` trains synchronously: each step sees the full bank.
- Pass if final CE < 0.5 * initial CE and final CE < 1.0.
- `lr = 0` control must not pass.
- Register `MODULE=falcon CONFIG=falcon_tiny_overfit`.

## Verification

- `tests/unit_tests/test_falcon_overfit.py` passes in rootfs.
- Config manager resolves the new module/config pair.

## Answer

Tiny LM + repeating bank + `falcon_tiny_overfit` are in place. Overfit gate:
CE 3.47 -> 0.011 in 80 steps for Falcon-1 and Falcon-1A. `lr=0` control fails
the gate. Launch with `experiments/falcon/run.sh overfit`.
