# Training State Estimator Roadmap

Status: ready-for-planning

## Intent

Design a staged TorchTitan program that turns repo-local run evidence into an
offline belief-state estimator for distributed training jobs.

The long-term vision is a hierarchical hybrid graphical state estimator for
large training runs: analytical execution models, factor-graph smoothing,
discrete fault-mode inference, learned observation and residual models, active
diagnostic probing, and hard safety invariants.

The first implementation phase is intentionally narrower:

- read immutable run-attempt evidence bundles after or alongside a run;
- build a time-indexed evidence graph of ranks, devices, process groups,
  phases, artifacts, incidents, and checkpoint state;
- produce deterministic, explainable diagnostic summaries for a bounded set of
  incidents;
- recommend probes only, with no automatic actuation; and
- keep learned models and control decisions behind explicit future extension
  points.

## Primary Sources

- Research note:
  `docs/research/2026-08-18-robotics-state-estimation-training-jobs.md`
- Training research vehicle design:
  `docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md`
- Observability paper closure:
  `docs/research/2026-08-12-training-observability-paper-closure.md`
- Implemented run evidence contract:
  `docs/run_evidence.md`
- Reliability boundary:
  `docs/adr/0001-robust-training-reliability-boundary.md`

## Classification

Route: gated path.

Reason:

- new observability subsystem;
- new durable schemas and analyzer outputs;
- cross-cutting interaction with run evidence, distributed semantics,
  checkpoint lineage, and incidents;
- future interfaces for active diagnostics and learned likelihoods; and
- safety-sensitive boundary around optimizer commit, checkpoint validity, and
  recovery actions.

Initial implementation surface: offline analyzer and research-program tooling.
Task execution follows a subagent-orchestrated workflow: a fresh implementer
subagent performs each task from precise context, a fresh reviewer subagent
checks spec and repository conformance, and a fresh finalizer subagent audits
both traces for challenges, issues, bottlenecks, plan or skill improvements,
and whether the task must be rerun. The main agent coordinates, records, and
integrates; it does not default to local task implementation.

Core `Trainer` changes are allowed only when a ticket explicitly requires a new
evidence field or marker and preserves the existing `run_evidence` contract.
No ticket in the initial series may change optimizer commit semantics,
checkpoint restore semantics, process-group membership semantics, or automatic
recovery policy.

## Authority Boundary

This spec authorizes design and staged implementation planning. It does not
authorize:

- committing, pushing, or opening a pull request;
- changing training control flow;
- auto-restarting, evicting, fencing, or rerouting live jobs;
- treating learned output as a safety decision;
- adding heavyweight ML dependencies to TorchTitan core;
- requiring optional diagnostic tools on the hot path; or
- making performance or correctness claims without run-bundle evidence.

The first implementation should be offline and read-only over existing evidence
bundles. All Python execution for this work must enter the TorchTitan bwrap
rootfs first through `scripts/rootfs/enter_rootfs.sh`; the host shell may only
orchestrate the wrapper, build or verify the rootfs itself, and inspect files
with non-Python tools. Missing Python packages are added inside rootfs with
`uv`; missing non-Python runtimes, compilers, CLIs, or tools are added or pinned
with `mise` unless a more specific rootfs provisioning script owns them.

## Domain Vocabulary

Belief state:
: A structured posterior over continuous health state, discrete mode, and root
  cause for a run, attempt, time window, and entity scope.

Evidence graph:
: A derived, time-indexed graph built from a run-attempt bundle. Nodes represent
  runs, attempts, ranks, hosts, devices, process groups, phases, checkpoints,
  artifacts, incidents, and observations. Edges represent physical, logical,
  control, lineage, and temporal relationships.

Topology epoch:
: A stable interval in which physical, logical, and control relationships are
  valid for interpretation. Process-group or replica membership changes start a
  new epoch.

Observation:
: A scalar, event, categorical state, trace summary, artifact transition, or
  incident record with event time, ingestion time when available, entity
  identity, sensor identity, quality, and sampling policy.

Sensor health:
: State describing the observation source itself: healthy, delayed, biased,
  stuck, or missing. Sensor-health uncertainty prevents resource diagnoses from
  trusting one faulty metric source.

Fault mode:
: A discrete operating or failure family such as nominal, compute degradation,
  network degradation, host/data stall, collective desynchronization, memory
  fault, numerical corruption, storage fault, control-plane fault, planned
  intervention, degraded, failed, or recovering.

Transaction risk:
: The estimated risk that an anomaly invalidates committed training state,
  speculative optimizer state, checkpoint state, membership epoch, or data
  cursor. V1 may report this risk only as an analysis field; it may not change
  commit or checkpoint behavior.

Probe recommendation:
: A recommendation for an operator or future controller to collect additional
  evidence, such as stack capture, local compute canary, isolated collective,
  replay, stopped-job diagnostic, or independent sensor query. V1 does not run
  probes automatically.

## Architecture

### Layer 0: existing evidence producers

The estimator consumes existing run-attempt bundles, structured logs, artifact
indexes, process outcomes, checkpoint artifacts, incident records, and optional
tool outputs.

It must preserve the existing evidence contract:

- immutable manifest and process outcomes;
- append-only artifact indexes;
- process-local event and artifact sequence numbers;
- wall clock for cross-process joins;
- monotonic clock for local ordering; and
- explicit distinction between uncollected, unknown, missing, and failed
  evidence.

### Layer 1: evidence graph builder

The builder reads one run attempt and emits a normalized graph snapshot:

```text
derived/state_estimator/evidence_graph.json
derived/state_estimator/timeline.jsonl
derived/state_estimator/entities.json
derived/state_estimator/quality.json
```

The graph is deterministic and rebuildable from raw evidence.

V1 graph relationships:

- run to attempt;
- attempt to process;
- process to rank, host, role, actor, and device when present;
- process to structured event stream;
- process to artifact index;
- process to outcome;
- artifacts to producer, kind, relation, state, and path;
- events to phase, step, process, and local sequence;
- incident records to evidence windows and terminal outcomes when present.

Physical topology can be incomplete in V1. Missing topology must be represented
as unknown, not guessed.

### Layer 2: deterministic classical estimators

The first estimators are explainable and offline:

- process liveness and outcome classifier;
- phase-duration and step-duration summarizer;
- peer-relative phase skew detector;
- collective arrival-skew analyzer when distributed event fields exist;
- host/data stall likelihood from data-loading and CPU/process evidence;
- compute stall likelihood from kernel/phase duration and GPU telemetry
  summaries when available;
- network stall likelihood from collective state and network evidence when
  available;
- checkpoint/transaction evidence completeness report; and
- evidence-quality and observability warning generator.

Output:

```text
derived/state_estimator/belief_summary.json
derived/state_estimator/diagnosis.md
```

The output may contain heuristic scores or likelihood-like values, but must not
present them as calibrated probabilities until calibration tests exist.

### Layer 3: fixed-lag incident timeline

The timeline smoother reconstructs a causal window around incidents:

- event time versus ingestion time;
- local ordering from monotonic clock and sequence;
- cross-process joins by wall clock and semantic keys;
- phase, step, and process-group context;
- late observations inserted at their event time; and
- explicit uncertainty when clocks or fields are missing.

V1 fixed lag can be a bounded offline window, such as last N steps or last M
minutes around an incident.

### Layer 4: probe recommendation

The recommendation engine maps ambiguity to next evidence actions:

- network versus late arrival -> capture collective state, run local compute
  canary, or run isolated collective after stopping;
- data stall versus compute stall -> capture stacks and dataloader evidence;
- sensor fault versus resource fault -> query independent sensor;
- numerical fault versus hardware SDC -> preserve checkpoint lineage and
  recommend deterministic replay under a stopped diagnostic workflow;
- checkpoint uncertainty -> validate shard inventory and parent lineage.

Recommendations include cost, risk, required authority, and whether the probe
is safe while the job is live.

### Layer 5: learned extension points

Learned systems are future work. V1 should define interfaces, not models:

- trace segment to likelihood factor;
- residual duration model;
- adaptive covariance or sensor-quality model;
- fault-mode prior;
- action-effect model.

These extension points must live outside the core hot path and must not require
TorchTitan core imports to load optional ML dependencies.

## Data Flow

```text
run_evidence bundle
  -> evidence graph builder
  -> entity/timeline/quality derived artifacts
  -> deterministic estimators
  -> belief summary and diagnosis
  -> probe recommendations
```

The analyzer writes derived artifacts under the same run-attempt bundle or a
separate ignored analysis output directory. Raw evidence remains untouched.

## Error Handling

- Malformed raw records are preserved as quality findings.
- Missing optional evidence weakens observability, not the whole analyzer.
- Missing required manifest or process identity fails the graph build for that
  attempt.
- Contradictory immutable evidence produces an explicit conflict finding.
- Analyzer failures must not modify raw artifacts.

## Testing Strategy

Use a small synthetic evidence-bundle fixture builder for unit tests. The
initial analyzer does not require a GPU or live distributed process, but every
Python test, compile check, and CLI invocation must run inside the TorchTitan
bwrap rootfs.

Test categories:

- graph construction from minimal single-process evidence;
- multi-rank evidence joins by run, attempt, process, rank, phase, and step;
- malformed or missing evidence quality findings;
- deterministic peer-skew classification fixtures;
- incident timeline smoothing with delayed observations;
- probe recommendation mapping for ambiguous hypotheses;
- schema round-trip tests for derived artifacts.

Later GPU or distributed integration tests may replay real run bundles, but the
first ticket series must remain host-scope and rootfs-executed.

## Initial Ticket Series

1. Persist vocabulary and schema alignment.
2. Build the offline evidence graph fixture and schema.
3. Implement the evidence graph builder.
4. Add deterministic phase and peer-skew estimators.
5. Add fixed-lag incident timeline reconstruction.
6. Add probe recommendation output.
7. Add learned-likelihood extension interfaces.
8. Add evaluation fixtures and reporting.

Each issue under `issues/` is a separately implementable ticket with its own
acceptance criteria and verification.

## Implemented V1 Slice

The first offline analyzer slice lives under
`torchtitan/observability/state_estimator/` and is run with:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m torchtitan.observability.state_estimator.cli --attempt-path <path>'
```

It reads a run-attempt evidence bundle and writes derived artifacts under
`derived/state_estimator/`:

- `evidence_graph.json`
- `entities.json`
- `timeline.jsonl`
- `quality.json`
- `belief_summary.json`
- `diagnosis.md`

The current implementation includes deterministic graph construction,
data-collection quality planning, process liveness, peer-skew heuristics,
fixed incident windows, recommendation-only diagnostic probes, learned-extension
schemas, and evaluation helpers. Learned outputs and probe recommendations are
advisory only.

## Non-Goals for the Initial Series

- Online control loop.
- Automatic rank eviction, replica fencing, rerouting, restart, or recovery.
- Checkpoint commit decision changes.
- Training hot-path central collector.
- Production learned models.
- Fleet-level quarantine policy.
- Global dense covariance over resources.
- Mandatory dependency on DCGM, NCCL RAS, Mycroft, ARGUS, Aegis, or py-spy.

## Completion Criteria for the Roadmap

The roadmap is ready to enter implementation when:

- the research note is persisted;
- this spec exists;
- all initial tickets have clear scope, non-goals, and verification commands;
- the first ticket is host-testable and does not require GPU allocation; and
- safety boundaries are explicit in every ticket that touches transactions,
  checkpoints, probes, or learned output.
