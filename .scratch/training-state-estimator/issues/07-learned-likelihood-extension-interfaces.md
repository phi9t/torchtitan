# Issue 07: Learned Likelihood Extension Interfaces

Status: open
Type: task
Blocked by: 03, 04

## Intent

Define extension interfaces for future learned observation factors and residual
models without adding ML dependencies to TorchTitan core or the first offline
analyzer.

This ticket should make future learned components pluggable while keeping the
classical analyzer useful by itself.

## Acceptance Criteria

- Define data classes or schemas for:
  - trace segment input;
  - learned likelihood factor output;
  - residual duration prediction;
  - adaptive sensor-quality output;
  - fault-mode prior output.
- Extension loading is optional and disabled by default.
- Core analyzer imports do not require torch, transformers, graph libraries, or
  neural ODE/CDE packages.
- Learned output is explicitly marked advisory and uncalibrated unless an
  evaluation artifact proves calibration.
- Tests verify the analyzer runs without optional learned extensions installed.

## Non-Goals

- Training a model.
- Adding a neural architecture.
- Adding online inference.
- Making learned output control transaction safety or recovery.

## Verification

Run:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_learned.py'
```
