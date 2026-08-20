# TorchTitan Foundation Design

Date: 2026-08-20
Status: draft design

## Purpose

Fortify the core TorchTitan training foundation so it can support foundation
model certification, observability, and downstream research programs without
turning the core trainer into an experiment harness.

The core surface remains `torchtitan/train.py` -> `ConfigManager` ->
`Trainer`. Core emits PyTorch-native training evidence and enforces training
invariants. Repo-local research programs and optional diagnostic tools consume
that evidence through adapters.

## Current State

The approved research-vehicle design sets the sequence:

1. observability foundation;
2. Qwen3 foundation training certification;
3. multimodal training;
4. post-training lineage.

The key architectural choice is evidence-contract first. `Trainer` emits a
small stream of semantic evidence. Launchers, offline analyzers, and optional
tools own orchestration, export, diagnosis, and richer capture.

Core and experiment surfaces are already separated:

- core training: `torchtitan/train.py`, `torchtitan/trainer.py`,
  `torchtitan/components/`, `torchtitan/distributed/`,
  `torchtitan/models/`;
- shared experiment execution: `torchtitan/experiments/execution/`;
- online RL: `torchtitan/experiments/rl/`;
- repo-local research programs: top-level `experiments/` plus reusable code in
  `torchtitan/experiments/`.

## Production Target

Core should expose three deep interfaces.

### Training Configuration

Interface:

- module and config are resolved only through `ConfigManager`;
- fields live on the config section that owns the behavior;
- CLI overrides apply after registry defaults;
- invalid combinations fail before training starts.

Implementation can remain distributed across model registries and component
configs, but callers should not need a second configuration path. Any future
foundation or multimodal certification work must prove it uses the same config
path as normal training.

### Trainer Evidence

Interface:

- every run has stable run and attempt identity;
- process/rank/mesh/device/step/phase fields are attached consistently;
- Tier 0 evidence is always on and low overhead;
- derived summaries can be rebuilt from raw evidence.

Implementation details such as TensorBoard, JSONL handlers, Flight Recorder,
DCGM ingestion, profiler windows, and state-estimator outputs stay behind
artifact and capture interfaces.

### Promotion Gate

Interface:

- a promotion request names model, data, global batch, sequence length,
  precision, mesh, checkpoint policy, and claim type;
- proof artifacts are attached by category;
- failure to supply a required proof yields a hold, not an implicit pass.

Promotion is not a property of a successful launch. It is a conclusion over
configuration, deterministic numerical behavior where promised, checkpoint
resume, observability, convergence, and matched performance evidence.

## Data and Control Flow

```text
ConfigManager
  -> Trainer construction
  -> Trainer emits core evidence
  -> ArtifactRegistry indexes evidence
  -> offline analyzers and dashboards derive views
  -> promotion gate evaluates claim-specific evidence
```

The hot path must not synchronously depend on a central collector. The trainer
should be able to continue producing Tier 0 evidence when optional exporters are
disabled or unavailable.

## Required Invariants

- Core training remains PyTorch-native.
- Core does not import NanoGPT, vLLM, external benchmark harnesses, or learned
  state-estimator dependencies.
- Optional tools enter through adapters that target the evidence contract.
- `training.local_batch_size` remains per data-parallel rank per gradient
  accumulation step.
- `training.steps` remains optimizer steps.
- Sequence, tensor-parallel, context-parallel, and pipeline-parallel invariants
  are validated against the mesh rather than inferred from a 1D topology.
- Checkpoint changes account for model, optimizer, scheduler, RNG, dataloader,
  and train-step state.
- Batch-invariant RL numerics do not become a core `Trainer` option.

## Hardening Tasks

1. Define the minimal core evidence interface and make it explicit in code and
   docs.
2. Add schema compatibility tests for run, attempt, process, rank, step, phase,
   and checkpoint lineage fields.
3. Add config-audit tests that exercise every affected model registry when a
   shared config or component changes.
4. Add a promotion-gate helper that reports missing evidence by category instead
   of relying on prose in reports.
5. Measure Tier 0 overhead on a representative training path and store the
   result as an artifact, not only a chat claim.

## Verification Ladder

- Unit: config validation, schema serialization, artifact registration, and
  promotion-gate missing-evidence diagnostics.
- Integration: fake-backend and local-tensor launches through `run_train.sh`.
- GPU: 2/4/8 GPU Qwen3 runs with fixed global batch and deterministic settings
  where numerical identity is promised.
- Performance: at least 10 measured training steps after initialization and
  warmup, with matched model, data, precision, topology, and feature settings.
- Fault: v1 drills for hangs, rank death, stragglers, nonfinite values,
  checkpoint interruption/corruption, inconsistent per-rank config, and hardware
  evidence workflows.

## Open Design Decisions

- Whether schema files for core evidence live beside `torchtitan/observability`
  code or under a shared `docs/schema/` tree.
- Whether promotion gates should be a standalone CLI, a library function, or
  both.
- How much of the existing experiment lifecycle layer should be promoted from
  repo-local experiment support into a stable public TorchTitan extension point.
