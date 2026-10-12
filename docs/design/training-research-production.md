# Training Research Production Design

Date: 2026-08-20
Status: draft design

## Purpose

This document is the master design map for hardening this TorchTitan checkout
before starting additional research programs. It categorizes the work already
landed in the repository and points to detailed direction designs that should
guide the next implementation tickets.

The immediate goal is not to add another benchmark, model, or training run. The
goal is to turn the current working foundation into production-grade modules:
small, stable interfaces with strong validation, replayable evidence, clear
claim boundaries, and tests that exercise the same seams future agents will use.

## Direction Docs

- [TorchTitan foundation](torchtitan-foundation.md): core training,
  observability contract, config boundaries, and promotion gates.
- [Experiment execution](experiment-execution.md): shared
  rootfs, lifecycle, attempt-store, preflight, and artifact interfaces for
  repo-local research programs.
- [NanoGPT B200 speedrun](nanogpt-b200-speedrun.md): the
  modded-nanogpt B200 harness, lane model, claim classification, runtime
  verification, and diagnostic probe ladder.
- [Qwen certification](qwen-certification.md): Qwen3 foundation
  training, FineWeb HSDP plus TP lessons, model promotion, dataset and
  checkpoint proof, and relation to post-training programs.
- [State estimator](state-estimator.md): offline training-state
  inference over run-attempt evidence, normalized observations, learned
  advisory factors, reports, and probe recommendations.
- [Post-training programs](post-training-programs.md): Countdown,
  scaffold-to-policy, online RL, benchmark/evaluation programs, and lineage from
  base checkpoint to policy and evaluation.

## Current Work Categories

### Core Training Foundation

The core foundation is defined by the existing TorchTitan training entrypoint:
`torchtitan/train.py` builds configuration through `ConfigManager`, then
constructs `torchtitan/trainer.py::Trainer`. The approved research-vehicle
design keeps this core PyTorch-native and makes it emit evidence through small
interfaces rather than importing experiment-specific dependencies.

Current foundation work includes the run-evidence machinery under
`torchtitan/observability/`, research notes under `docs/research/`, the earlier
approved research-vehicle design, and tests around run evidence and distributed
configuration.

### Shared Experiment Execution

The execution layer under `torchtitan/experiments/execution/` is the emerging
shared substrate for repo-local experiments. It defines immutable run
declarations, attempts, stage specs, stage events, artifact references, and
condition status. It also includes rootfs identity and preflight helpers.

This layer is the natural place to absorb duplicated runner behavior from
NanoGPT, Countdown, scaffold-to-policy, Qwen experiments, and future programs.
It must remain experiment-owned: useful to repo-local research programs, but not
a required dependency of the core trainer.

### NanoGPT B200 Speedrun

The B200 speedrun harness lives under `experiments/modded_nanogpt_b200/` with
its canonical tracker/spec under `.scratch/modded-nanogpt-b200/`. It is a
repo-local harness around a pinned external upstream, not a TorchTitan training
mode.

The current implementation covers source preparation, data preparation,
preflight, rootfs runtime verification, launch readiness, active-job checks,
log parsing, summary generation, matrix execution, optimized-kernel
certification, and diagnostic performance probes. It is implemented through a
non-launch foundation and two-GPU Lane B prerequisite path. It does not yet
carry a successful B200 reproduction claim.

### Qwen Work

The Qwen work spans foundation training and post-training usage:

- `experiments/qwen3_fineweb_hsdp_tp/` holds the local FineWeb HSDP plus TP
  experiment and distributed introspection assets.
- Qwen3 0.6B and 1.7B are the first certification targets for the research
  vehicle.
- Qwen3 8B and Qwen3 30B-A3B MoE are later promotion targets.
- Countdown and scaffold-to-policy use Qwen3-1.7B for SFT, vLLM evaluation, and
  policy experiments.
- Online RL uses Qwen-family recipes through a separate Monarch-controlled
  trainer/generator system.

This needs one certification vocabulary so "smoke", "numerical identity",
"checkpoint compatibility", "convergence", "matched performance", and
"quality improvement" cannot be conflated.

### State Estimation

`torchtitan/observability/state_estimator/` is the offline analysis layer for
training-job evidence. It consumes run-attempt bundles, normalizes evidence into
observations, builds graph and timeline views, estimates likely fault states,
loads optional learned advisory factors, recommends probes, and writes reports.

The critical production boundary is that the estimator advises. It does not
decide retry, rollback, checkpoint validity, quarantine, or recovery actions.

### Post-Training Programs

Countdown, scaffold-to-policy, and online RL are related but distinct:

- Countdown is offline search distillation for exact arithmetic tasks.
- Scaffold-to-policy is a broader benchmark and training-research scaffold for
  reasoning, coding, and agentic tasks.
- Online RL is a Monarch-controlled async trainer/generator system, not an
  alternate core `Trainer` loop.

They should share lifecycle, rootfs, artifact, and evidence primitives while
keeping verifiers, datasets, rewards, and domain-specific claim gates local.

## Production-Grade Criteria

A direction is production-grade when all of the following are true.

- The owner surface is explicit: core, online RL, repo-local experiment, or
  shared experiment execution.
- The public interface is small enough that callers do not need to understand
  the implementation to use it correctly.
- Invariants are encoded in types, schemas, config validation, or tests rather
  than comments alone.
- Generated artifacts are immutable or append-only where claim evidence depends
  on them.
- Every claim-bearing artifact has a schema version, run identity, attempt
  identity, source/data/runtime provenance, and conservative claim
  classification.
- Optional dependencies stay behind experiment-owned or adapter-owned seams.
- Rootfs-required work fails closed when invoked from the host.
- Tests exercise behavior through the same interface used by real runners.
- Documentation states what the system can prove and what it cannot prove.

## Shared Vocabulary

- **Run declaration**: immutable scientific intent for one logical run or
  campaign cell. Reusing a run ID with different normalized content is invalid.
- **Attempt**: one operational execution of a declaration. A resume after a
  terminal result creates a new attempt with parent linkage.
- **Stage**: one declared unit of work inside an attempt, such as preflight,
  prepare, train, evaluate, ingest, or report.
- **Artifact reference**: metadata that records what bytes were produced,
  reused, resumed, imported, or external, and what freshness is proven.
- **Claim classification**: the conservative category describing what a result
  may support, independent of whether the run was successful.
- **Evidence bundle**: the immutable or append-only record from which summaries,
  reports, dashboards, and state-estimator output can be regenerated.
- **Rootfs boundary**: the execution seam that separates host orchestration from
  Python, CUDA, NCCL, data, parsing, summarization, and training work.
- **Certification ladder**: ordered proof levels from static checks and smoke
  runs through numerical, checkpoint, convergence, performance, and replicated
  evaluation evidence.

## Shared Architecture Target

The desired architecture is:

```text
repo-local program wrappers
  -> experiment execution interface
      -> rootfs/runtime adapters
      -> attempt store and artifact registry
      -> preflight and claim validators
      -> optional state-estimator analysis
  -> core Trainer evidence hooks
```

The dependency direction remains:

```text
repo-local research programs -> experiment integrations -> PyTorch-native core
```

Core must not import NanoGPT, vLLM, benchmark harnesses, or optional state
estimator extensions. Repo-local programs may depend on shared experiment
execution modules and may consume core evidence.

## Hardening Backlog

### P0: Freeze the Shared Interfaces

- Define the stable `RunDeclaration`, `Attempt`, `StageSpec`, `StageEvent`,
  `ArtifactRef`, and `AttemptOutcome` interface in
  `torchtitan/experiments/execution/`.
- Define one rootfs execution interface for shell wrappers and Python callers.
- Define one artifact schema versioning policy for attempt, preflight, runtime,
  summary, and state-estimator outputs.
- Add compatibility tests that fail if an interface changes without a schema or
  migration decision.

### P1: Collapse Duplicate Runner Logic

- Move reusable rootfs re-entry, active-job checks, command capture, terminal
  outcome classification, and atomic report writing into shared modules.
- Keep NanoGPT-specific lane rules and patch provenance in
  `experiments/modded_nanogpt_b200/`.
- Keep Countdown/scaffold verifiers and task formats in their own packages.

### P2: Make Claim Boundaries Machine-Checkable

- Represent smoke, diagnostic, prerequisite, baseline, compatibility, and
  reproduction states as validated enums.
- Require claim-bearing artifacts to include enough evidence to re-run the
  classification offline.
- Make summary generation reject inconsistent source, runtime, data, or launch
  facts rather than downcasting silently.

### P3: Harden Evidence Consumption

- State-estimator ingestion should read the same attempt bundles that launchers
  write.
- Reports must include source paths and evidence states for every conclusion.
- Learned factors remain advisory and must not emit control actions.

### P4: Promote by Ladder, Not by Anecdote

- Qwen foundation promotion requires configuration, numerical, checkpoint,
  convergence, performance, and observability evidence.
- NanoGPT reproduction requires final validation under the lane-specific source,
  data, runtime, and launch rules.
- Post-training quality claims require frozen split identities, verifier
  versions, base-vs-policy comparisons, and replication where the claim scope
  demands it.

## What Is Out Of Scope For This Design Bundle

- Running new full GPU jobs.
- Pushing to remotes.
- Moving experiment logic into core.
- Declaring NanoGPT reproduction success.
- Declaring Qwen promotion success.
- Automatic recovery, quarantine, or retry control based on state-estimator
  output.

## Review Questions Before Implementation

- Which shared interface should be treated as the first stable contract:
  experiment lifecycle, rootfs runtime, artifact schemas, or state-estimator
  ingestion?
- Should `torchtitan/experiments/execution/` own JSON Schema files, Python
  dataclasses, or both?
- Should NanoGPT keep local schemas temporarily, or migrate immediately to the
  shared execution schema with adapter fields?
- What is the minimum verification ladder required before any future experiment
  is allowed to claim more than "plumbing works"?
