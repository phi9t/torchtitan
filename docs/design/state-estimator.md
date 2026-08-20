# State Estimator Design

Date: 2026-08-20
Status: draft design

## Purpose

Fortify `torchtitan/observability/state_estimator/` as the offline analysis
module for large training-job evidence. It should help operators understand
training state and choose follow-up probes while preserving a strict advisory
boundary.

The estimator consumes evidence. It does not control training, retry, rollback,
checkpoint validity, quarantine, or recovery.

## Current State

The package includes:

- `schema.py`: shared dataclasses, evidence states, severities, calibration
  states, paths, canonical JSON, and atomic writes;
- `bundle.py`: run-attempt bundle loading;
- `observation.py`: normalized observations and envelopes;
- `data_collection.py`: evidence collection helpers;
- `topology.py`: process, rank, mesh, and device topology snapshots;
- `timeline.py`: event timeline construction;
- `graph.py`: evidence graph construction;
- `analytical.py`: analytical summaries;
- `inference.py`: fault-mode scoring;
- `learned.py`: optional learned advisory factors and extension loading;
- `probes.py`: recommended follow-up probes;
- `reports.py`: diagnosis and evaluation report generation;
- `cli.py`: command-line entrypoints.

The tests cover schemas, bundle ingestion, observations, topology, timelines,
graphs, analytical summaries, inference, learned factors, reports, probes, and
CLI behavior.

The research note `docs/research/2026-08-18-robotics-state-estimation-training-jobs.md`
provides the conceptual model: training jobs are partially observed distributed
systems, and the estimator should represent uncertainty instead of presenting
correlation as root cause.

## Production Target

The state estimator should expose this deep interface:

```text
load attempt evidence
  -> normalize observations
  -> derive topology, timeline, graph, and quality findings
  -> estimate advisory belief over latent states
  -> recommend probes
  -> write reproducible reports
```

The public caller should provide an attempt path and optional analysis config.
The caller should receive derived artifacts and a terminal report. The caller
should not need to know how logs, metrics, hardware samples, checkpoint records,
and learned-factor bundles are joined internally.

## Module Interfaces

### EvidenceBundleReader

Owns:

- run-attempt bundle layout;
- source path discovery;
- raw record loading;
- missing, malformed, contradictory, uncollected, and unknown evidence states.

Hardening requirements:

- accept the shared experiment execution attempt layout;
- preserve source paths and raw record identities;
- keep malformed evidence as evidence, not as silent absence;
- avoid importing heavy optional dependencies.

### ObservationNormalizer

Owns:

- correlation envelope validation;
- normalized observation identity;
- event and ingestion clocks;
- process, rank, role, actor, device, step, and phase keys.

Hardening requirements:

- distinguish event time, monotonic time, ingestion time, and sequence order;
- validate that CUDA device index is not used as a cross-host identity;
- preserve unknown fields in payloads for forward compatibility.

### DerivedViews

Owns:

- topology snapshot;
- incident timeline;
- evidence graph;
- quality findings;
- analytical summary.

Hardening requirements:

- every derived assertion carries an evidence state and source path where
  possible;
- derived files are reproducible from raw evidence;
- contradictions are represented explicitly rather than resolved implicitly.

### BeliefEstimator

Owns:

- heuristic and learned likelihood factors;
- fault-mode scores;
- calibration state;
- advisory metadata.

Hardening requirements:

- learned extensions load only when explicitly requested;
- default imports do not load ML or GPU dependencies;
- factor bundles are schema-validated;
- all output is advisory and excludes control-decision terms.

### ProbeRecommender

Owns:

- next diagnostic probes;
- evidence gaps;
- expected cost or disruption class;
- prerequisites for each probe.

Hardening requirements:

- recommendations must cite the missing or contradictory evidence that motivates
  them;
- probes are ranked but not automatically executed;
- expensive or destructive probes require launcher or human authorization.

## Advisory Boundary

Allowed estimator outputs:

- likely fault states;
- confidence or calibration state;
- evidence gaps;
- diagnostic probe recommendations;
- report summaries;
- contradictions or quality findings.

Disallowed estimator outputs:

- retry this job;
- roll back to this checkpoint;
- mark this checkpoint valid;
- quarantine this node;
- continue training;
- promote this model.

Those are control-plane or human decisions. The estimator may provide evidence
for them, but must not encode them as actions.

## Data Flow

```text
attempt bundle
  -> EvidenceBundleReader
  -> ObservationNormalizer
  -> DerivedViews
  -> BeliefEstimator
  -> ProbeRecommender
  -> reports and derived artifacts
```

The derived artifact root should remain:

```text
<attempt>/derived/state_estimator/
```

Each output should include schema version, source references, and evidence-state
metadata.

## Verification Ladder

- Unit: schema round trips, malformed evidence, missing evidence, contradictory
  evidence, learned-factor validation, and advisory-boundary rejects.
- Import isolation: default imports do not load `torch`, `triton`, `cuda`,
  `networkx`, or learned-model dependencies.
- Fixture bundles: generated attempt fixtures cover complete, partial, blocked,
  failed, and malformed runs.
- Golden reports: markdown and JSON report outputs are stable for representative
  fixtures.
- Integration: ingest real NanoGPT, Qwen, Countdown, and scaffold attempt
  bundles through the shared execution layout.
- Performance: large event streams are bounded in memory and runtime, with
  streaming or indexing where needed.

## Hardening Tasks

1. Define the attempt-bundle reader against the shared execution schema rather
   than per-program assumptions.
2. Add a small analysis config that selects outputs without changing ingestion
   semantics.
3. Add fixture bundles for blocked, failed, diagnostic, and successful attempts.
4. Freeze learned-factor bundle schemas and default import-isolation tests.
5. Add evidence-state propagation tests from raw record to final report.
6. Add probe recommendation tests that prove recommendations cite evidence gaps.
7. Add large-stream tests before using the estimator on long multi-rank jobs.

## Open Design Decisions

- Whether the evidence graph should remain a JSON-only derived artifact or gain
  an optional graph-library adapter.
- Whether learned factor bundles should be signed or content-addressed before
  they are trusted in production reports.
- How to represent multi-attempt comparisons without letting the estimator make
  retry or promotion decisions.
