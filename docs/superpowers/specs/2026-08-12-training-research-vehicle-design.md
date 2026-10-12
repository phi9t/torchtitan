# TorchTitan Training Research Vehicle Design

Date: 2026-08-12
Status: Approved design

## Purpose

Develop this TorchTitan checkout into a research vehicle for:

1. production-grade distributed-training observability;
2. foundation-model training with Qwen3 dense and MoE baselines;
3. native and pluggable multimodal training; and
4. SFT, rejection sampling, and online RL post-training.

The program must retain TorchTitan's existing surface boundaries. Core training,
online RL, and repo-local research programs remain distinct systems with shared
contracts where appropriate. Optional experiment and diagnostic dependencies
must not become requirements of the PyTorch-native core.

This design is informed by the bounded primary-source review in
[`docs/research/2026-08-12-training-observability-paper-closure.md`](../../research/2026-08-12-training-observability-paper-closure.md).

## Approved decisions

- Build the research foundation first, multimodal capabilities second, and the
  full post-training pipeline third.
- Use a 2-GPU, 4-GPU, and 8-GPU validation ladder.
- Validate with Qwen3 0.6B and 1.7B before promoting to Qwen3 8B and Qwen3
  30B-A3B MoE.
- Make repo-local artifacts the canonical observability record. TensorBoard
  remains the compact scalar dashboard; remote services are optional exporters.
- Integrate observability into the core `Trainer` first. Online RL and offline
  research programs adopt the evidence schema later.
- Use an evidence-contract-first architecture rather than making individual
  tools or the `Trainer` itself the observability control plane.
- Keep low-overhead telemetry always on. Capture heavier evidence when an issue
  is suspected, before aborting where safe.
- Limit the always-on layer to at most 1% steady-state throughput regression.
- Continue after captured performance warnings. Abort on fatal correctness or
  unrecoverable distributed faults.
- Diagnose, but do not automatically recover from or quarantine, faults in v1.
- Include broad v1 failure drills: hangs, rank death, compute and data
  stragglers, nonfinite values, checkpoint corruption/interruption, per-rank
  configuration inconsistency, and hardware evidence/EUD workflows.

## Program boundaries and sequence

### Phase 1: observability foundation

Instrument the core `Trainer` and establish the evidence contract, capture
tiers, local analysis, overhead policy, and fault-injection suite. The design
must anticipate multiple roles and meshes, but v1 delivery is not blocked on RL
or offline-program adoption.

### Phase 2: foundation-model training

Certify matched runs on Qwen3 0.6B and 1.7B across 2, 4, and 8 GPUs. Promote the
same operating and evidence contracts to Qwen3 8B and then Qwen3 30B-A3B MoE.
Promotion requires numerical, checkpoint, convergence, performance, and
observability evidence rather than successful launch alone.

### Phase 3: multimodal training

Pursue native vision-language models and pluggable vision encoders as distinct
paths. Use Qwen3-VL as an initial compatibility target, then build on the
repo's Qwen3.5 native multimodal seams. Add a Qwen3.6 integration only from a
verified public architecture and checkpoint contract. In parallel, support
SigLIP2 fine-tuning and a trained vision-token projector for compatible
language-model families.

### Phase 4: post-training

Connect SFT, offline candidate generation and rejection sampling, retraining or
distillation, online RL, and evaluation through explicit checkpoint, dataset,
sample, verifier, reward, and policy lineage.

The dependency direction remains:

```text
research programs -> experiment integrations -> PyTorch-native core
```

Online RL remains a Monarch-controlled trainer/generator system, not a mode of
the core `Trainer`. Repo-local generation and evaluation programs remain
offline experiments, not online RL controller stages.

## Architecture choice

Three architectures were considered.

### Evidence-contract first -- selected

Define versioned run identity, correlation keys, artifact registration,
incident state, and capture policy. Existing and new diagnostic tools remain
native artifact producers. This creates a durable join across training,
distributed, hardware, checkpoint, and correctness evidence and can extend to
RL without coupling core training to a remote service.

### Tool-adapter first -- rejected as the organizing principle

Launcher wrappers around each preferred tool would produce practical evidence
quickly, but identity, trigger, retention, and incident semantics would diverge.
Adapters remain part of the selected design, but must target the shared evidence
contract.

### Trainer-native controller -- rejected for v1

Putting diagnosis and operational actions inside `Trainer` would simplify some
automatic triggers but would couple core training to optional tools and make
multi-actor reuse harder. The trainer emits semantic evidence and signals;
launcher-owned or offline components orchestrate optional external diagnostics.

## Run-attempt evidence model

Each execution produces an immutable run-attempt bundle. A representative
layout is:

```text
run_id/
  attempt_id/
    manifest.json
    processes/<process_id>/events.jsonl
    artifacts/<producer>/...
    indexes/artifacts.<process_id>.jsonl
    checkpoints/<checkpoint_id>.json
    incidents/<incident_id>.json
    outcome.json
    derived/...
```

`run_id` identifies a logical experiment. `attempt_id` identifies a launch,
retry, restart, or recovery attempt. Raw records and artifact indexes are
append-only. Derived timelines, summaries, and dashboard inputs may be
regenerated from raw evidence.

Every record uses a versioned correlation envelope sufficient to join:

- run, attempt, role, actor instance, host, PID, and process identity;
- global and local rank, mesh ranks and axes, process group, and device UUID;
- wall clock, monotonic clock, and process-local event sequence;
- optimizer step, microbatch, phase, and optional module/compiled-graph identity;
- checkpoint and parent checkpoint, data position, and membership epoch when
  those concepts apply.

CUDA device index alone is not a cross-host device identity. Communicator hashes
alone are not a cross-attempt process-group identity. Wall time supports
cross-process joins; monotonic time and event sequence preserve local ordering.

### Attribution layers

The envelope and artifact index must preserve joins across these layers:

- Kernel evidence records the profiler or launch correlation, CUDA stream and
  device, enclosing operator/module/phase, and deliberately enabled shape,
  dtype, or compiled-code identity.
- Torch module evidence uses stable `record_function` or NVTX ranges around
  model blocks, attention, MoE routing, loss, optimizer, checkpoint staging,
  and collectives. Eager module FQNs and compiled graph identities remain
  distinguishable.
- Distributed evidence records process-group identity, mesh axes, collective
  sequence, operation, tensor metadata, participation, and completion state.
- Job evidence joins progress and phase timing to host, device, power, thermal,
  clock, error, and utilization series without treating correlation as root
  cause.
- Checkpoint and correctness evidence records semantic state inventory, lineage,
  data position, validation, actual work per step, and nonfinite sentinels.

Deep tools may omit fields that would violate their overhead tier. The contract
must distinguish an uncollected field from an unknown value.

### Component seams

- `EvidenceContext` owns process-local identity and attaches correlation fields.
- `ArtifactRegistry` indexes logs, traces, dumps, snapshots, checkpoints, and
  diagnostic reports in their native formats.
- `CapturePolicy` owns capture tiers, triggers, hysteresis, rank selection,
  retention, artifact budgets, and incompatible-tool rules.
- `IncidentRecorder` records detection, evidence capture, continuation or abort,
  diagnosis, validation, useful work, lost work, and terminal disposition.
- Offline analyzers join phase skew, collective state, hardware telemetry,
  checkpoint lineage, incident timelines, and effective training time ratio.
- Launcher-owned tool adapters invoke or ingest optional DCGM, py-spy, NCCL,
  EUD, SuperBench, and Nsight evidence.

No synchronous central collector is required on the training hot path. Core
depends only on small, PyTorch-native evidence interfaces.

## Capture and escalation policy

The high-level capture state is:

```text
NORMAL -> SUSPECTED -> CAPTURING -> CONTINUE
                              \-> ABORT_AND_PRESERVE
```

Triggers must be persistent, peer-relative where appropriate, rate-limited, and
hysteretic. Compilation, warm-up, checkpointing, or transient scheduling noise
must not cause a capture storm.

### Tier 0: always on

The always-on evidence set includes:

- run manifest and structured per-rank events;
- progress, phase timing, loss, grad norm, valid tokens, MFU, memory, and data
  loading timing;
- checkpoint lineage and terminal outcome;
- bounded PyTorch Flight Recorder history;
- low-rate DCGM health and job telemetry; and
- stdout, stderr, and process-exit evidence.

Tier 0 must meet the measured 1% steady-state throughput budget. GPU-memory
delta, host CPU use, artifact bytes per GPU-hour, and detection latency are
reported separately and have explicit release thresholds.

### Tier 1: scheduled samples

Use short PyTorch Profiler windows on selected ranks, bounded CUDA allocation
history and memory snapshots, and richer per-rank phase distributions. Matched
comparative runs use the same steps and rank-selection rules.

### Tier 2: anomaly-triggered capture

- Hang or rank death: dump Flight Recorder, query NCCL RAS, collect repeated
  py-spy process-tree snapshots, and preserve surviving-rank evidence tails.
- Persistent compute or data straggler: preserve peer-relative timing and DCGM
  windows, capture process stacks, and schedule a bounded profiler window if
  the job is still progressing.
- Nonfinite or correctness fault: preserve the evidence window and checkpoint
  lineage, then abort.
- Checkpoint corruption: retain manifest, validation, storage, and restore
  evidence and do not silently select a different checkpoint.
- Hardware event: retain passive DCGM evidence and stop the workload before
  active diagnostics.

Performance anomalies continue after bounded capture unless they cross an
explicit fatal threshold. V1 does not automatically retry, recover, or
quarantine hardware.

### Tier 3: stopped-job diagnosis

Use targeted DCGM EUD, `nccl-tests`, SuperBench, deterministic workload replay,
Nsight Systems, and filtered Nsight Compute only after stopping or in a separate
diagnostic allocation. Each result is a new execution linked to the originating
incident, not a direct recording of the failed process.

Tool versions, target device identity, start/end time, command and configuration,
and before/after device settings are required evidence. Capture policy must
record tool conflicts and side effects, including DCGM profiling conflicts and
the possibility that active EUD changes device settings.

## Validation and promotion contract

### Baseline matrix

Qwen3 0.6B and 1.7B are certified at 2, 4, and 8 GPUs. Comparisons fix the model,
checkpoint, dataset, global batch, tokens per optimizer step, precision,
parallelism semantics, and seed. Mesh layouts may differ only when the test is
explicitly studying them.

For Tier 0 overhead certification:

- compare disabled and enabled paired runs;
- exclude initialization, compilation, and warm-up;
- measure at least ten steady-state optimizer steps;
- repeat each condition at least three times;
- require no more than 1% median throughput regression; and
- report memory, CPU, artifact volume, and detection latency separately.

Non-computation instrumentation must preserve bitwise loss and grad-norm
identity under TorchTitan's deterministic comparison setup. Full-precision
evidence, not rounded stdout, establishes identity.

### Broad fault-injection suite

The initial suite covers:

- a rank hanging before a collective;
- abrupt rank death;
- persistent trainer/compute straggling;
- persistent dataloader straggling;
- injected NaN or Inf correctness failure;
- corrupted checkpoint manifest or shard;
- interrupted save or restore plus restore-equivalence validation;
- inconsistent per-rank configuration; and
- DCGM hardware-event ingestion followed by a stopped-job EUD drill.

A drill passes when the evidence bundle identifies the correct failure class and
injected process/rank or bounded distributed fault domain, retains the required
pre-failure window, records the chosen action, and links every artifact. An EUD
result is follow-up hardware evidence and is not assumed to reproduce every
intermittent or silent failure.

### Model promotion

After the small-model matrix passes:

- Qwen3 8B becomes the representative dense convergence and performance
  baseline.
- Qwen3 30B-A3B becomes the MoE baseline, adding expert-load imbalance,
  routing, expert communication, and large checkpoint-state gates.

A larger-model success cannot compensate for a failed small-model observability,
checkpoint, or correctness gate.

## Foundation-training workflow

Foundation research continues through `run_train.sh -> torchrun -> Trainer`.
Research runners may select registry configurations, define matrices, and gather
evidence; they must not duplicate the core training loop or create a second
configuration path.

Every promoted configuration records:

- model and tokenizer provenance;
- normalized configuration, source revision, and dirty-worktree state;
- dataset identity, preprocessing, sequence length, and token counts;
- local batch, batch-mesh degree, accumulation, and effective global batch;
- precision, optimizer, scheduler, clipping, compilation, activation
  checkpointing, and parallelism;
- checkpoint ancestry and the complete train-state inventory; and
- numerical, convergence, throughput, memory, and observability evidence.

The model ladder has distinct purposes:

- Qwen3 0.6B: rapid configuration, distributed-logic, checkpoint, and
  fault-injection validation.
- Qwen3 1.7B: stronger numerical and scaling verification across 2, 4, and 8
  GPUs.
- Qwen3 8B: representative dense convergence and performance research.
- Qwen3 30B-A3B: MoE routing, load-balance, expert-parallel communication, and
  checkpoint-scale research.

Promotion proceeds through configuration validation, fake/local communication
checks where applicable, deterministic short runs, checkpoint-resume
equivalence, representative convergence evidence such as C4, and matched
performance runs. Smoke runs prove plumbing only.

## Multimodal architecture

### Native VLM path

- Add Qwen3-VL as the compatibility baseline.
- Build on the existing Qwen3.5 native vision-language implementation.
- Add Qwen3.6 only against a verified public processor, architecture, model,
  and checkpoint contract.
- Preserve model-family processor behavior, image-token layout, positional
  encoding, attention masking, state-dict naming, and checkpoint semantics.

### Pluggable vision path

```text
images -> SigLIP2 encoder -> vision-token projector -> language-model embeddings
```

The projector contract owns dimensional projection, normalization, token
pooling and layout, image boundaries, and insertion metadata. Thin family
adapters translate the contract into Qwen3, DeepSeek-v4, or GLM-5.2 embedding,
mask, and position conventions. Each named external family is a research target,
not an assertion of compatibility; integration begins only after its public
processor, architecture, and checkpoint contract has been verified.

Training is staged:

1. fine-tune and validate SigLIP2 independently;
2. freeze both backbones and train the projector;
3. optionally unfreeze selected vision or language-model layers for joint
   tuning; and
4. export encoder, projector, and language-model states with separate
   namespaces and provenance.

SigLIP2 and external model-family packages remain optional experiment
dependencies. Promote an interface to shared core only after at least two model
families use the same semantics. Matching hidden dimensions is not sufficient
evidence of plugin compatibility; require processor equivalence, forward and
state-dict validation, checkpoint round trips, and task evaluation.

Multimodal evidence extends the run bundle with preprocessing time, image-token
counts, dynamic-resolution distributions, projector and vision-module ranges,
vision memory, modality-specific losses, and data-quality failures.

## Post-training pipeline

The intended lineage is:

```text
base checkpoint
  -> SFT
  -> candidate generation
  -> verification/reward
  -> rejection-selected dataset
  -> SFT/distillation
  -> online RL
  -> evaluation and promotion
```

Every generated sample records prompt identity, source split, producing
checkpoint, sampling parameters, candidate identity, verifier or reward
version, component scores, selection decision, and downstream dataset
membership. Training, validation, and test identities remain globally disjoint.

Execution boundaries remain explicit:

- SFT and LoRA use experiment-owned configurations over reusable training
  components.
- Rejection sampling is an offline generation, verification, selection, and
  retraining program.
- Online RL remains the Monarch controller with separate TorchTitan trainer and
  vLLM generator meshes.

RL adds rollout, prompt-group, generator-replica, trainer-policy, packed-batch,
optimizer-step, and weight-sync identities to the common envelope. Reports
separate generation latency, queueing, batch construction, trainer forward and
backward, weight synchronization, and end-to-end step time.

Correctness and production evidence remain separate:

- strict on-policy, deterministic, batch-invariant runs establish trainer and
  generator log-probability and loss parity;
- production-style asynchronous runs evaluate throughput, reward, policy age,
  backpressure, and stability;
- rejection sampling reports task success separately from format compliance;
  and
- promotion requires seed or split-draw replication and comparison with the
  originating base or SFT checkpoint.

Checkpoint and dataset lineage must allow a final policy to be traced through
RL weights, selected examples, candidate generation, SFT data, and the original
foundation checkpoint.

## `AGENTS.md` operating contract

`AGENTS.md` will preserve its concrete surface, configuration, testing,
numerical, and engineering guidance while adding a concise research-program
layer. It will encode:

- the four-phase sequence and surface boundaries;
- the 2/4/8-GPU and Qwen3 promotion ladders;
- evidence-contract-first observability;
- the Tier 0 overhead and capture-first escalation rules;
- broad fault-injection requirements;
- native VLM versus SigLIP2/projector boundaries;
- post-training checkpoint, sample, verifier, reward, and policy lineage; and
- the difference between plumbing, numerical equivalence, convergence,
  performance, reliability, and quality-improvement evidence.

Detailed schemas, tool invocations, milestone matrices, and research findings
remain in linked documents rather than being duplicated in agent instructions.
The bounded paper closure is the rationale and tool inventory for the
observability choices.

## Completion and non-goals

This design does not authorize one monolithic implementation. Each phase must
be split into reviewable milestones with focused tests and explicit promotion
evidence.

V1 observability does not include fleet scheduling, automatic host quarantine,
or an FT-HSDP recovery implementation. It creates the evidence needed to study
such systems correctly. It also does not make Prometheus, OpenTelemetry, W&B,
DCGM, py-spy, Nsight, vLLM, SigLIP2, or external model packages core
dependencies.

The first implementation plan is limited to updating `AGENTS.md` so future work
follows this approved operating contract. Implementations of the observability,
foundation, multimodal, and post-training milestones require their own plans
and proof gates.
