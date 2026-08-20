# Post-Training Programs Design

Date: 2026-08-20
Status: draft design

## Purpose

Fortify the repo-local post-training programs so they can produce trustworthy
evidence for policy improvement without mixing offline generation, SFT,
evaluation, and online RL into one ambiguous workflow.

This direction covers Countdown search distillation, scaffold-to-policy,
agentic/coding benchmark harnesses, and the bridge into online RL.

## Current State

### Countdown

`experiments/countdown_search_distill/` and
`torchtitan/experiments/countdown_search_distill/` implement an offline
search-distillation pilot. The program generates exact-DP-solvable Countdown
problems, samples with a scaffold, filters or selects examples, trains Qwen3
LoRA adapters with TorchTitan configs, exports adapters, and evaluates with the
same pass@k path.

The important contract is exact arithmetic:

- each number is used at most once;
- division is exact;
- intermediates are positive integers;
- output format includes `FINAL: <target>`.

The program distinguishes easy, elicitable, and unreached problems. Historical
reports and ADRs record the reduced-pilot gate, split-registry discipline, and
replication/sweep evidence.

### Scaffold-To-Policy

`experiments/scaffold_to_policy/` and
`torchtitan/experiments/scaffold_to_policy/` implement a broader research
workspace for converting verified scaffold behavior into policy behavior. It
includes synthetic tasks, public reasoning benchmarks, coding benchmarks,
external-harness probes, report artifacts, runtime doctor checks, and Qwen/vLLM
execution paths.

The program is already organized around claim separation:

- scaffold reachability;
- SFT compression;
- deterministic RL improvement;
- controlled async RL throughput and quality;
- downstream agentic benchmark transfer.

### Online RL

`torchtitan/experiments/rl/` is a separate Monarch-controlled system. Its
pipeline is:

```text
dataset -> prompt-group work buffer -> rollouts -> training samples
        -> packed microbatches -> policy update -> weight sync -> generators
```

It is not a mode of the core `Trainer`. Its hardest invariants are prompt-group
capacity, policy age, weight sync ordering, prefix-cache reset, trainer/generator
model alignment, and deterministic parity when exact logprob or loss claims are
made.

## Production Target

Post-training programs should share one evidence lifecycle while retaining
domain-specific verifiers and rewards:

```text
base checkpoint
  -> data or prompt declaration
  -> scaffold generation
  -> verification or reward
  -> selected dataset
  -> SFT or distillation
  -> exported policy
  -> deterministic RL or async RL
  -> evaluation
  -> claim report
```

Every stage should produce artifacts that can be joined by run, attempt, stage,
checkpoint, dataset, sample, verifier, reward, policy version, and evaluation
identity.

## Module Interfaces

### TaskContract

Owns:

- task identity;
- split policy;
- verifier identity and version;
- prompt or chat-template contract;
- accepted output format;
- scoring semantics;
- exclusion policy.

Hardening requirements:

- task success and format compliance are recorded separately;
- verifier failures are distinct from model failures;
- OOD claims require a predeclared distribution-shift statement;
- benchmark locked sets cannot influence prompts, filtering, or stop decisions.

### SampleLineage

Owns:

- base checkpoint;
- prompt source;
- scaffold type and budget;
- generation parameters;
- verification or reward result;
- selection rule;
- adapter or policy checkpoint;
- evaluation target.

Hardening requirements:

- every selected training row links to the generated candidate and verifier or
  reward evidence that justified it;
- reused artifacts record freshness and digest;
- manual curation is either disallowed for a claim or recorded explicitly;
- base, raw scaffold, selected scaffold, SFT, and RL policies are kept separate.

### PolicyComparison

Owns:

- compared policies;
- fixed task suite or split draw;
- paired result cells;
- uncertainty and replication scope;
- promotion evidence and recommendation.

Hardening requirements:

- zero is a valid measurement, not a blocked or failed run;
- smoke and fixture measurements cannot support quality claims;
- a later claim cannot repair a failed earlier claim;
- comparisons preserve base-vs-candidate causal anchors;
- comparison output is advisory; human review or an explicitly designed control
  plane owns any final policy promotion action.

### RLBridge

Owns:

- exported policy format;
- TorchTitan load path;
- vLLM load path;
- trainer/generator model alignment;
- deterministic parity mode;
- async policy-age and throughput metrics.

Hardening requirements:

- deterministic RL claims require strict on-policy FIFO, matched seeds, matched
  TP, batch-invariant mode, disabled sequence parallelism, compatible
  prefix-cache reset, and full-precision loss comparison artifacts;
- async RL claims report policy age, queue occupancy, rollout wait, generation
  latency, trainer throughput, and weight-sync wait separately;
- RL pipeline parallelism remains unsupported unless a separate design changes
  that contract.

## Claim Boundaries

Use these labels consistently:

- **plumbing**: a smoke or fixture path proves the runner works.
- **base calibration**: a frozen base model is measured on a pinned task/split.
- **scaffold reachability**: best-of-N or tool-assisted generation can reach a
  verified solution.
- **SFT compression**: supervised training transfers reachable behavior into
  first-sample policy behavior.
- **deterministic RL improvement**: strict on-policy RL improves over a declared
  base or SFT policy without off-policy ambiguity.
- **controlled async RL**: one async factor changes with policy-age and quality
  evidence.
- **downstream transfer**: one selected policy improves an official external
  harness or benchmark under a predeclared claim scope.

## Error Handling

- Missing model assets, GPU memory, optional packages, or external credentials
  block the stage; they are not failed quality measurements.
- Invalid verifier setup invalidates the measurement cell, not the whole
  research idea.
- Benchmark harness crashes are recorded as execution failures with stage
  evidence.
- If policy age exceeds the declared bound, the async RL cell is invalid for the
  declared claim.
- If the locked set informs prompts, cleaning, or stop decisions, the locked
  claim is invalid and must be rerun on a fresh holdout.

## Verification Ladder

- Unit: exact verifiers, importers, report artifacts, lineage records, and
  status classification.
- Runtime doctor: rootfs, Python packages, vLLM, model assets, GPU memory, and
  external harness availability.
- Smoke: small generated or imported task sets with complete artifacts.
- Calibration: frozen base on pinned splits or benchmark subsets.
- SFT: selected-policy training with checkpoint and export proof.
- Replication: multiple split draws and seeds for claims beyond one fixed run.
- Deterministic RL: strict parity and policy freshness proof.
- Async RL: one-factor throughput/quality experiments after deterministic proof.
- Downstream: official harness comparison after the selected policy exists.

## Hardening Tasks

1. Move shared run declaration, stage, artifact, and status recording to the
   experiment execution substrate.
2. Keep Countdown arithmetic and split logic local to Countdown.
3. Keep scaffold-to-policy task adapters, benchmark harness adapters, and report
   templates local to scaffold-to-policy.
4. Add a shared `TaskContract` shape used by Countdown and scaffold-to-policy.
5. Add sample-lineage tests that prove every selected row links back to source
   generation and verification or reward evidence.
6. Add policy-comparison fixtures for zero-score, blocked, failed, fixture,
   smoke, and real measurement cells.
7. Add RL bridge tests for deterministic parity prerequisites and policy-age
   invalidation.

## Open Design Decisions

- Whether `TaskContract` belongs under `torchtitan/experiments/execution/` or a
  separate `torchtitan/experiments/evaluation/` package.
- Whether external harness execution should use the training rootfs, a separate
  candidate sandbox, or both with explicit trust boundaries.
- Which minimum replication matrix is required before claiming recipe-general
  improvement rather than improvement for a fixed policy realization.
