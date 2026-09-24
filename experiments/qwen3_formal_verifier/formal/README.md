# Qwen3 Step Formal Trace

This directory contains generated formal artifacts for a TorchTitan-style Qwen3 dense DPxTP training step.

- `Qwen3StepTrace.tla` captures the finite DPxTP mesh, process groups, work ids, work roles, and group membership used by the TLA+ transition model.
- `Qwen3StepTrace.lean` captures the same finite trace facts in a Lean-friendly shape for trace-level invariant proofs.

The source of truth is the Python trace IR emitted by `build_torchtitan_qwen3_step_trace`; regenerate these files through `write_qwen3_step_formal_artifacts` after schema or role changes.
