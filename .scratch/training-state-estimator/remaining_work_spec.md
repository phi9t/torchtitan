# Training State Estimator Remaining Work Spec

Status: draft
Date: 2026-08-18

## Purpose

Specify the remaining TorchTitan state-estimation work after the first offline
analyzer slice. The target is the robotics-inspired estimator described in
`docs/research/2026-08-18-robotics-state-estimation-training-jobs.md`, staged
so each step is testable, reviewable, and safe for the training control path.

The current v1 analyzer is an offline, evidence-bundle-driven tool. It reads
existing run-attempt evidence, writes derived state-estimator artifacts, emits
heuristic liveness and peer-skew findings, recommends stopped-job probes for
ambiguous skew, defines learned-extension schemas, and provides a basic
evaluation harness. It is intentionally not yet a topology-aware hybrid
factor-graph estimator.

## Execution Rules

All remaining work follows the repository execution guidance:

- Python, PyTorch, modeling, analyzer, tests, static Python checks, data
  preparation, validation, parsing, summarization, and CLI entrypoints run
  inside `scripts/rootfs/enter_rootfs.sh`.
- The host shell may only orchestrate rootfs entry, build or verify the rootfs,
  and inspect files with non-Python tools.
- Missing Python packages are added inside rootfs with `uv`.
- Missing non-Python runtimes, compilers, CLIs, or tools are added or pinned
  with `mise` unless a rootfs provisioning script already owns them.
- Each implementation ticket uses a fresh implementer subagent, a fresh review
  subagent, and a fresh finalizer subagent. The main agent coordinates,
  records trace links, applies accepted plan/skill improvements, and reruns a
  task only after the finalizer says a rerun is needed.

## Safety Boundary

The estimator remains advisory until a later human-approved automation program
changes that boundary. Remaining work must not:

- change optimizer commit behavior;
- change checkpoint validity decisions;
- change process-group or replica membership;
- auto-evict, auto-fence, auto-reroute, auto-restart, or auto-rollback;
- treat learned output as a transaction-safety decision;
- present heuristic scores as calibrated probabilities; or
- make throughput, detection, localization, or quality claims without recorded
  evaluation evidence.

Hard invariants live outside learned systems. Learned factors may add
likelihoods, residuals, priors, or uncertainty estimates, but they may not
override transaction, checkpoint, membership, or numerical-corruption policy.

## Current V1 Baseline

Implemented package:

```text
torchtitan/observability/state_estimator/
  schema.py
  bundle.py
  fixtures.py
  graph.py
  data_collection.py
  estimators.py
  timeline.py
  probes.py
  learned.py
  evaluation.py
  cli.py
```

Current derived artifacts:

```text
derived/state_estimator/evidence_graph.json
derived/state_estimator/entities.json
derived/state_estimator/timeline.jsonl
derived/state_estimator/quality.json
derived/state_estimator/belief_summary.json
derived/state_estimator/diagnosis.md
derived/state_estimator/evaluation_summary.json
derived/state_estimator/evaluation_report.md
```

Current capabilities:

- required bundle reader for `manifest.json`, artifact indexes, structured
  logs, and process outcomes;
- deterministic graph of runs, attempts, processes, artifacts, outcomes, and
  structured observations;
- data-collection quality report for required raw signals and a few optional
  signals;
- process liveness from outcome records;
- peer-relative phase skew grouped by phase and step;
- fixed incident window around failed attempts when a center time is supplied;
- advisory probe recommendation for ambiguous late-arrival versus network
  hypotheses;
- learned-output schemas with advisory flags and no ML dependency;
- fixture-based evaluation for mode and probe expectations; and
- rootfs-verified unit and adjacent observability tests recorded in issue 09.

## Gap Summary

The research note requires a hierarchical hybrid graphical state estimator. The
v1 analyzer covers only the first offline skeleton. The remaining gaps are:

- **Evidence contract depth:** no explicit topology epoch, process-group epoch,
  membership epoch, device UUID, physical topology, logical mesh graph,
  control graph, checkpoint semantic manifest, data cursor, incident record,
  sensor identity, or ingestion clock.
- **Observation model:** no typed scalar/event/categorical/trace summaries, no
  sensor-health model, no clock-quality model, no sampling policy, and no
  uncertainty representation.
- **Execution model:** no phase duration summaries, no max-plus critical path,
  no analytical compute roofline model, no collective duration model, no
  arrival-skew versus network-progress split, and no workload conditioning.
- **Inference:** no latent state representation, no robust likelihoods, no
  change-point detector, no IMM/HSMM mode inference, no sparse root-cause
  inversion, and no covariance or observability matrix output.
- **Topology:** no multiplex physical/logical/control graph; no device, host,
  NIC, storage, checkpoint, process-group, or replica relationships beyond the
  process-artifact-outcome graph.
- **Transactions:** no committed/speculative step state, replica epoch,
  process-group epoch, checkpoint freshness, data cursor, or transaction-risk
  report.
- **Active diagnosis:** recommendations are one hard-coded mapping, not a
  scored probe planner with information target, cost, risk, authority, and
  expected ambiguity reduction across probe families.
- **Learned interfaces:** schemas exist, but there is no segment extraction,
  learned-factor ingestion contract, calibration metadata, feature store,
  residual model adapter, or offline scoring harness.
- **Evaluation:** no labeled corpus manifest, no injected-fault fixture suite,
  no detection latency, no failure-domain localization, no top-k root cause,
  no calibration metrics, no system utility metrics, and no reproducible
  benchmark report format.
- **Operator usability:** the CLI has no explicit output directory, no
  evaluation manifest option, no real-bundle audit mode, no versioned report
  summary, and limited failure diagnostics.

## Target Architecture

The remaining program implements the research note in staged layers:

0. **V1 hardening:** close the current analyzer gaps before adding new layers:
   populate phase-duration summaries where possible, wire failed-attempt
   timeline center selection, preserve local ordering fields, and expose
   v1 host/rank/role/actor/device/phase/step attributes already present in
   evidence.
1. **Evidence substrate v2:** richer run-attempt bundle reads and derived
   schema for identity, topology, time, sensor quality, transaction state, and
   artifact lineage.
2. **Multiplex evidence graph:** physical, logical, and control graph layers
   with topology epochs and stable entity references.
3. **Semantic timeline and telemetry store:** bounded fixed-lag windows over
   phase, step, collective, checkpoint, hardware, and incident observations.
4. **Analytical execution models:** workload-conditioned phase, compute,
   memory, checkpoint, and collective duration predictors.
5. **Robust classical inference:** process/device/link/rank/job latent state,
   robust residuals, change detection, discrete mode scoring, observability
   warnings, and sparse root-cause ranking.
6. **Active diagnostic planner:** scored, authority-aware recommendations for
   canary, replay, independent sensor, stopped-job benchmark, profiler, and
   lineage-validation probes.
7. **Learned extension integration:** trace segments, learned likelihood
   factors, residual predictions, covariance or sensor-quality predictions,
   and fault-mode priors as optional advisory inputs.
8. **Evaluation and calibration:** synthetic, replayed, real small-run, and
   injected-fault cases with detection, localization, calibration, and utility
   metrics.
9. **CLI and reports:** reproducible offline analysis and evaluation commands
   that generate stable JSON plus concise Markdown reports from raw evidence.

## Data Collection And Evidence Contract

### Required identity envelope

Every observation inserted into the estimator should carry available fields
from this envelope:

```text
run_id
attempt_id
schema_version
record_type
role
actor_id
process_id
host_name
pid
global_rank
local_rank
world_size
device_index
device_uuid
mesh_axis_<axis>_rank
mesh_axis_<axis>_size
process_group_id
process_group_epoch
replica_id
replica_epoch
topology_epoch
step
microbatch
phase
event_time_ns
ingestion_time_ns
monotonic_ns
event_seq
artifact_seq
sensor_id
sampling_policy
quality
```

Missing fields are not invented. They become `unknown` or `uncollected`
quality entries depending on whether the producer existed.

### Raw evidence families

The next-stage bundle reader must recognize and classify:

- structured events and scalar metrics;
- artifact lifecycle rows;
- process outcomes;
- typed incident records;
- checkpoint save/load lifecycle rows;
- checkpoint semantic manifests;
- distributed/process-group membership snapshots;
- Flight Recorder declarations and discovered dumps;
- profiler traces and trace summaries;
- CUDA memory snapshots and memory summaries;
- DCGM or hardware telemetry summaries;
- NCCL RAS or collective-health reports;
- py-spy or stack signature summaries;
- stopped-job diagnostic outputs from EUD, `nccl-tests`, SuperBench, and
  Nsight; and
- evaluation labels and intervention records.

V1 may ingest summaries first and leave native deep artifacts indexed.

### Derived observation families

The analyzer should normalize raw inputs into typed observations:

- scalar observation;
- event observation;
- categorical state observation;
- trace summary observation;
- artifact transition observation;
- checkpoint lineage observation;
- transaction observation;
- topology observation;
- incident observation;
- evaluation label observation; and
- learned factor observation.

Each normalized observation should preserve raw source path and raw record
identity.

## Graph And Topology Model

The evidence graph must become a multiplex graph:

- **Physical graph:** host, GPU, HBM, NVLink, PCIe root, NIC, storage, power or
  cooling domain, and optional rack/fabric entities.
- **Logical graph:** DP/FSDP/HSDP, TP, PP, CP, EP process groups, mesh axes,
  collective operations, microbatch dependencies, sequence/context assignment,
  and expert ownership where available.
- **Control graph:** scheduler or launcher identity, heartbeats, supervision
  links, checkpoint ownership, incident ownership, replica membership, and
  recovery coordinator fields when available.

The graph must support topology epochs. If topology is incomplete, derived
quality must say which relationships are missing and which diagnoses are
therefore unobservable.

## Timeline And Clock Model

The timeline layer must distinguish:

- event time;
- ingestion time;
- monotonic process-local time;
- artifact sequence;
- event sequence;
- semantic phase order; and
- local versus cross-process ordering confidence.

It must support:

- incident-centered windows;
- explicit step or phase windows;
- last-N-step windows;
- delayed observations inserted by event time;
- missing-clock warnings;
- local sequence fallback;
- wall-clock skew warnings; and
- compact Markdown summaries of the most relevant events.

## Analytical Models

The first analytical model library should remain dependency-light and
deterministic. It should include:

- phase-duration summaries by process, rank, phase, and step;
- step critical-path model using max-plus predecessor relationships;
- simple compute roofline predictor from estimated FLOPs, bytes, and resource
  capacity when workload metadata exists;
- memory-pressure predictor from allocator and snapshot summaries;
- ring all-reduce duration predictor using message bytes, group size, latency,
  and effective bandwidth;
- collective decomposition into arrival skew and network progress time when
  collective launch/completion states exist;
- checkpoint staging/save/load duration model; and
- residual normalization:

```text
normalized_residual =
  (observed_duration_ns - expected_duration_ns) / expected_sigma_ns
```

Analytical outputs are explanatory features for inference, not claims of root
cause by themselves.

## Classical Inference

The first classical inference milestone should implement an offline robust
estimator, not a full online iSAM2 system. It should produce:

- process/rank liveness and progress state;
- per-entity sensor-health state;
- phase and step residuals with Student-t-like robust scoring;
- CUSUM-style persistent degradation scores;
- discrete mode scores for nominal, compute degradation, network degradation,
  host/data stall, collective desynchronization, memory fault, numerical
  corruption, storage fault, control-plane fault, planned intervention, failed,
  and recovering;
- root-cause candidates with affected scope and evidence chain;
- observability warnings for indistinguishable hypotheses;
- transaction-risk advisory fields; and
- uncertainty/calibration state on every score.

Mode scores remain heuristic until evaluated against labeled cases. The output
field should say `calibration="heuristic"` or `calibration="calibrated"` per
score.

## Transaction And Checkpoint State

The estimator must represent:

- committed step;
- speculative step;
- process-group epoch;
- replica epoch;
- latest recoverable checkpoint;
- checkpoint parent;
- checkpoint shard inventory;
- RNG/dataloader/data cursor evidence;
- save/load/staging status;
- restore validation status; and
- whether evidence is sufficient to reason about transaction risk.

V1 transaction-risk output is advisory. It can say that evidence is missing or
that an anomaly occurred before a commit boundary, but it cannot decide that an
optimizer update or checkpoint is valid.

## Active Diagnostic Planner

Probe recommendations should be scored objects:

```text
kind
target_entities
competing_hypotheses
information_target
expected_information_gain_class
cost_class
risk_class
required_authority
live_safe
requires_stopped_job
preconditions
expected_artifacts
source_evidence
```

Initial probe families:

- local compute canary;
- isolated collective;
- same-microbatch replay on another GPU;
- same-data replay under different placement;
- independent sensor query;
- stack snapshot;
- profiler window;
- Flight Recorder dump preservation;
- checkpoint lineage validation;
- BF16/FP32 replay for numerical instability;
- stopped-job `nccl-tests`;
- stopped-job SuperBench or EUD; and
- prior-code-version replay.

The planner recommends only. It must not run probes.

## Learned Extension Path

Learned work is staged behind optional interfaces:

1. Segment extraction from normalized observations into trace segments.
2. Offline feature store for scalar, event, trace, communication, hardware,
   and framework-semantics modalities.
3. Learned likelihood factor ingestion with source model identity,
   calibration, training-data manifest, and advisory flag.
4. Residual prediction ingestion for workload-conditioned expected residuals.
5. Sensor-quality and fault-mode-prior ingestion.
6. Evaluation and calibration gates before any score is presented as calibrated.

No learned package is imported by the core analyzer unless explicitly requested.

## Evaluation And Calibration

Evaluation must cover four sources:

- synthetic fixtures;
- rootfs CPU or fake-backend runs;
- real small GPU run bundles;
- controlled fault-injection or stopped-job diagnostic bundles.

Metrics:

- detection probability by severity;
- time to detection;
- false alerts per case and eventually per GPU-hour;
- root-cause top-1 and top-k;
- failure-domain precision/recall or scope intersection-over-union;
- observability-warning correctness;
- expected probe found;
- Brier score and expected calibration error for calibrated outputs;
- credible-interval coverage where continuous labels exist;
- artifact bytes per GPU-hour;
- runtime overhead for always-on producers;
- validated committed tokens per allocated GPU-second when real training labels
  exist; and
- operator-minutes and rerun cost when intervention records exist.

Calibration is separate from detection. A high anomaly score is not a
high-confidence root cause.

## Reporting

The CLI should support:

- analyze one attempt;
- analyze a run with multiple attempts;
- compare attempts;
- evaluate against a labeled manifest;
- write to a separate output directory;
- emit JSON paths in stdout;
- produce Markdown diagnosis and evaluation reports;
- fail with actionable errors for malformed required evidence; and
- record which optional evidence was absent and what diagnoses this prevents.

## Milestones

### Milestone A: Evidence Substrate V2

Complete when v1 hardening is done and the analyzer can ingest richer raw
records and emit normalized typed observations with identity, clock, topology,
and quality metadata.

### Milestone B: Multiplex Graph And Timeline

Complete when derived artifacts include physical/logical/control graph layers,
topology epochs, semantic windows, clock-quality warnings, and process-group
relationships.

### Milestone C: Analytical And Robust Classical Estimator

Complete when the analyzer produces phase, compute, collective, checkpoint,
and persistent-degradation residuals; mode scores; root-cause candidates; and
observability warnings.

### Milestone D: Transaction And Checkpoint Reasoning

Complete when checkpoint lineage, data cursor, process-group epoch, replica
epoch, and transaction-risk advisory fields are present and tested.

### Milestone E: Active Diagnosis Planner

Complete when probe recommendations cover the initial probe families with
explicit authority, risk, cost, preconditions, and expected artifacts.

### Milestone F: Learned Extension And Feature Store

Complete when learned factors can be loaded as advisory inputs with model
identity and calibration metadata, without importing ML dependencies by
default.

### Milestone G: Evaluation Corpus And Calibration

Complete when synthetic, CPU/fake-backend, real small-run, and injected-fault
manifests can be evaluated with detection, localization, probe, observability,
and calibration metrics.

### Milestone H: Operator CLI And Reports

Complete when the CLI can run the full offline pipeline over one attempt or a
run of attempts and produce stable JSON plus Markdown reports.

## Open Design Questions

- Which structured event fields should core training emit first for phase,
  microbatch, collective, and checkpoint semantics without exceeding the Tier-0
  overhead budget?
- Which physical topology source is authoritative on a single host: DCGM,
  NCCL topology dump, `nvidia-smi topo`, launcher metadata, or an external
  fixture?
- How should process-group identity be represented before PyTorch exposes a
  stable cross-rank group ID for every collective?
- What is the minimum checkpoint semantic manifest that can prove lineage
  without duplicating DCP internals?
- Which first controlled fault suite is cheap enough to run regularly inside
  this checkout?
- Which outputs can be usefully calibrated from synthetic fixtures, and which
  require real injected faults?

## Non-Goals For Remaining Planning

- Automatic recovery or quarantine.
- Fleet scheduler integration.
- Online streaming dashboard.
- Mandatory DCGM, NCCL RAS, Nsight, py-spy, Mycroft, ARGUS, Aegis, or neural
  dependency in the core analyzer.
- Dense covariance over every resource in memory.
- Claiming calibrated probabilities before labeled evaluation supports them.
