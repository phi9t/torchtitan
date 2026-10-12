# Training State Estimator Remaining Work Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. The main agent is an orchestrator. Every task uses a fresh implementation subagent, a fresh review subagent, and a fresh finalizer subagent that audits both traces before the orchestrator integrates or reruns the task.

**Goal:** Evolve the v1 offline analyzer into the staged, robotics-inspired training state estimator described in `remaining_work_spec.md`.

**Architecture:** Keep the analyzer offline and advisory while deepening it in layers: evidence substrate, multiplex graph, semantic timeline, analytical models, robust classical inference, transaction/checkpoint reasoning, active diagnostic planning, learned advisory factors, evaluation/calibration, and operator CLI/reporting. Core training emits or indexes evidence only through explicit tickets; estimator outputs do not alter training control.

**Tech Stack:** Python standard library first; optional dependencies only when a ticket proves they are needed. Python execution, tests, static checks, analyzer commands, and PyTorch/modeling work run inside `scripts/rootfs/enter_rootfs.sh`. Use `uv` for missing Python packages inside rootfs and `mise` for missing runtimes/tools unless a rootfs provisioning script owns them.

**Spec:** `.scratch/training-state-estimator/remaining_work_spec.md`

## Global Constraints

- The estimator remains offline and advisory.
- Raw run-attempt evidence is read-only; derived artifacts are rebuildable.
- No task may change optimizer commit, checkpoint validity, process-group
  membership, recovery, reroute, restart, rollback, eviction, or quarantine
  behavior.
- Learned outputs are advisory and uncalibrated unless evaluation artifacts
  prove calibration.
- Every Python command is rootfs-wrapped:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && <python command>'
```

- The only code that may run outside rootfs is code that builds, verifies, or
  launches the bwrap rootfs.
- Every task brief must include: exact files, allowed edits, prohibited
  actions, rootfs commands, expected artifacts, and finalizer questions.
- Every task completes only after implementer, reviewer, and finalizer traces
  agree that the task is acceptable or the orchestrator records a rerun.
- Keep optional tool integrations as indexed native artifacts or summaries; do
  not make them required imports of TorchTitan core.
- Keep large generated bundles, traces, checkpoints, and evaluation corpora out
  of git.

## Orchestrator Protocol

For each task:

1. Create a task brief from the matching issue file, this plan, and the spec.
2. Dispatch a fresh implementer subagent with precise context and no unrelated
   history.
3. Require the implementer to run red-green tests through rootfs, list changed
   files, and provide command evidence.
4. Dispatch a fresh reviewer subagent to check spec conformance, repo guidance,
   safety boundaries, rootfs/dependency compliance, tests, and unintended
   scope.
5. Dispatch a fresh finalizer subagent to read the implementer and reviewer
   traces. It must report:
   - challenges encountered;
   - issues and bottlenecks;
   - missed dependencies or tools;
   - ambiguous instructions;
   - reusable improvements to this plan or task/workstream skills; and
   - whether the implementation must be rerun.
6. If rerun is required, write the improved brief and dispatch a new
   implementer subagent. Do not reuse the failed implementer.
7. If accepted, record trace links, verification evidence, and finalizer
   findings in the issue file.

## File Structure

Expected new or deepened modules:

```text
torchtitan/observability/state_estimator/
  observation.py
  topology.py
  graph.py
  timeline.py
  analytical.py
  inference.py
  transactions.py
  probes.py
  learned.py
  evaluation.py
  reports.py
  cli.py
```

Expected tests:

```text
tests/unit_tests/observability/state_estimator/
  test_observation.py
  test_topology.py
  test_graph.py
  test_timeline.py
  test_analytical.py
  test_inference.py
  test_transactions.py
  test_probes.py
  test_learned.py
  test_evaluation.py
  test_reports.py
  test_cli.py
```

The exact split may change if a finalizer identifies a better seam. Preserve
the same interfaces and issue acceptance criteria if file names change.

## Workstream Order

Blocker graph:

```text
10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18 -> 19 -> 20
                              \             /
                               -> 21 -------
```

- 10: v1 analyzer hardening.
- 11: typed observation substrate.
- 12: topology and multiplex graph.
- 13: semantic timeline and clock quality.
- 14: analytical execution models.
- 15: robust inference and mode scoring.
- 16: transaction and checkpoint advisory state.
- 17: active diagnostic planner.
- 18: learned-factor ingestion and feature store.
- 19: evaluation corpus and calibration reports.
- 20: operator CLI and reports.
- 21: evidence producer roadmap and core-training hooks.

Task 21 may start after Task 13 for design/spec work, but any code changes to
core training wait for explicit ticket approval and must preserve the existing
run-evidence contract.

## Task 10: V1 Analyzer Hardening

**Issue:** `issues/10-v1-analyzer-hardening.md`

**Files:**
- Modify: `torchtitan/observability/state_estimator/graph.py`
- Modify: `torchtitan/observability/state_estimator/estimators.py`
- Modify: `torchtitan/observability/state_estimator/timeline.py`
- Test: `tests/unit_tests/observability/state_estimator/test_graph.py`
- Test: `tests/unit_tests/observability/state_estimator/test_timeline.py`
- Test: `tests/unit_tests/observability/state_estimator/test_estimators.py`

**Interfaces:**
- Produces populated `phase_durations` where evidence supports it.
- Wires failed-attempt center selection into `write_belief_summary`.
- Adds v1 evidence fields to graph entities and timeline rows.

**Acceptance:**
- Adds phase-duration summaries or explicit missing-evidence quality findings.
- Failed outcomes can produce incident timelines without explicit center input.
- Monotonic time, event sequence, and artifact sequence are preserved when
  present.
- Graph entities include host, rank, role, actor, device, phase, and step when
  present in existing evidence.

**Verification:**

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_graph.py tests/unit_tests/observability/state_estimator/test_timeline.py tests/unit_tests/observability/state_estimator/test_estimators.py tests/unit_tests/observability/state_estimator/test_cli.py'
```

## Task 11: Typed Observation Substrate

**Issue:** `issues/11-typed-observation-substrate.md`

**Files:**
- Create: `torchtitan/observability/state_estimator/observation.py`
- Modify: `torchtitan/observability/state_estimator/schema.py`
- Modify: `torchtitan/observability/state_estimator/bundle.py`
- Test: `tests/unit_tests/observability/state_estimator/test_observation.py`

**Interfaces:**
- Produces `ObservationKind`, `ClockQuality`, `ObservationEnvelope`,
  `NormalizedObservation`, `normalize_bundle_observations(bundle)`.
- Consumes existing `RunEvidenceBundle`.

**Acceptance:**
- Normalizes structured events, artifact rows, and outcomes into typed
  observations.
- Preserves raw source path and raw record identity.
- Distinguishes event time, ingestion time, monotonic time, event sequence, and
  artifact sequence.
- Emits quality findings for missing sensor IDs, missing clocks, and malformed
  identity fields.

**Verification:**

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_observation.py tests/unit_tests/observability/state_estimator/test_bundle.py'
```

## Task 12: Topology And Multiplex Graph

**Issue:** `issues/12-topology-and-multiplex-graph.md`

**Files:**
- Create: `torchtitan/observability/state_estimator/topology.py`
- Modify: `torchtitan/observability/state_estimator/graph.py`
- Test: `tests/unit_tests/observability/state_estimator/test_topology.py`
- Test: `tests/unit_tests/observability/state_estimator/test_graph.py`

**Interfaces:**
- Produces `TopologyEpoch`, `GraphLayer`, `build_topology_snapshot`,
  `build_multiplex_evidence_graph`.
- Consumes normalized observations from Task 11.

**Acceptance:**
- Represents physical, logical, and control layers separately.
- Adds host, process, rank, device, mesh-axis, process-group, artifact,
  checkpoint, and incident entities when present.
- Emits `unknown` or `uncollected` quality findings for absent topology rather
  than guessing.
- Keeps deterministic entity and edge ordering.

**Verification:**

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_topology.py tests/unit_tests/observability/state_estimator/test_graph.py'
```

## Task 13: Semantic Timeline And Clock Quality

**Issue:** `issues/13-semantic-timeline-clock-quality.md`

**Files:**
- Modify: `torchtitan/observability/state_estimator/timeline.py`
- Test: `tests/unit_tests/observability/state_estimator/test_timeline.py`

**Interfaces:**
- Produces `TimelineWindow`, `TimelineEntry`, `ClockQualityFinding`,
  `build_semantic_timeline`.
- Consumes multiplex graph observations from Task 12.

**Acceptance:**
- Supports incident-centered, explicit time, explicit step, and last-N-step
  windows.
- Inserts delayed observations at event time.
- Uses monotonic time and process-local sequence for local ordering fallback.
- Emits clock-quality warnings when cross-process ordering is weak.
- Provides compact incident summaries for reports.

**Verification:**

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_timeline.py'
```

## Task 14: Analytical Execution Models

**Issue:** `issues/14-analytical-execution-models.md`

**Files:**
- Create: `torchtitan/observability/state_estimator/analytical.py`
- Modify: `torchtitan/observability/state_estimator/estimators.py`
- Test: `tests/unit_tests/observability/state_estimator/test_analytical.py`
- Test: `tests/unit_tests/observability/state_estimator/test_estimators.py`

**Interfaces:**
- Produces `DurationResidual`, `PhaseSummary`, `CollectivePrediction`,
  `predict_ring_all_reduce_ns`, `summarize_phase_durations`,
  `estimate_collective_components`.
- Consumes semantic timeline from Task 13.

**Acceptance:**
- Computes phase duration summaries by process/rank/phase/step.
- Models max-plus critical path for phase dependencies where predecessors are
  known.
- Predicts ring all-reduce duration from message bytes, group size, latency,
  and bandwidth.
- Splits collective symptoms into arrival skew and network progress when data
  exists.
- Emits normalized residuals with explicit `heuristic` calibration.

**Verification:**

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_analytical.py tests/unit_tests/observability/state_estimator/test_estimators.py'
```

## Task 15: Robust Classical Inference

**Issue:** `issues/15-robust-classical-inference.md`

**Files:**
- Create: `torchtitan/observability/state_estimator/inference.py`
- Modify: `torchtitan/observability/state_estimator/estimators.py`
- Test: `tests/unit_tests/observability/state_estimator/test_inference.py`
- Test: `tests/unit_tests/observability/state_estimator/test_estimators.py`

**Interfaces:**
- Produces `ModeScore`, `RootCauseCandidate`, `SensorHealthState`,
  `ObservabilityWarning`, `score_fault_modes`, `rank_root_causes`.
- Consumes analytical residuals from Task 14.

**Acceptance:**
- Scores nominal, compute degradation, network degradation, host/data stall,
  collective desynchronization, memory fault, numerical corruption, storage
  fault, control-plane fault, planned intervention, failed, and recovering.
- Uses robust residual scoring that limits one outlier's influence.
- Adds CUSUM-style persistent degradation scores for repeated weak residuals.
- Produces root-cause candidates with affected scope and evidence chains.
- Emits observability warnings when hypotheses are indistinguishable.

**Verification:**

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_inference.py tests/unit_tests/observability/state_estimator/test_estimators.py'
```

## Task 16: Transaction And Checkpoint Advisory State

**Issue:** `issues/16-transaction-checkpoint-state.md`

**Files:**
- Create: `torchtitan/observability/state_estimator/transactions.py`
- Modify: `torchtitan/observability/state_estimator/graph.py`
- Modify: `torchtitan/observability/state_estimator/estimators.py`
- Test: `tests/unit_tests/observability/state_estimator/test_transactions.py`

**Interfaces:**
- Produces `CheckpointLineageState`, `TransactionEvidenceState`,
  `TransactionRiskAdvisory`, `summarize_checkpoint_lineage`,
  `estimate_transaction_risk`.

**Acceptance:**
- Represents committed step, speculative step, process-group epoch, replica
  epoch, checkpoint parent, shard inventory, data cursor, save/load status, and
  validation status when evidence exists.
- Emits missing-evidence warnings when transaction risk cannot be assessed.
- Does not decide commit validity or checkpoint validity.

**Verification:**

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_transactions.py tests/unit_tests/observability/state_estimator/test_estimators.py'
```

## Task 17: Active Diagnostic Planner

**Issue:** `issues/17-active-diagnostic-planner.md`

**Files:**
- Modify: `torchtitan/observability/state_estimator/probes.py`
- Test: `tests/unit_tests/observability/state_estimator/test_probes.py`

**Interfaces:**
- Extends `ProbeRecommendation`.
- Produces `rank_probe_recommendations`.

**Acceptance:**
- Covers local compute canary, isolated collective, same-microbatch replay,
  same-data replay, independent sensor query, stack snapshot, profiler window,
  Flight Recorder preservation, checkpoint lineage validation, BF16/FP32
  replay, stopped-job `nccl-tests`, stopped-job SuperBench/EUD, and prior-code
  replay.
- Each recommendation includes target entities, competing hypotheses,
  information target, cost, risk, authority, stopped-job requirement,
  preconditions, expected artifacts, and source evidence.
- Planner recommends only and never runs probes.

**Verification:**

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_probes.py'
```

## Task 18: Learned Factor Ingestion And Feature Store

**Issue:** `issues/18-learned-factor-ingestion.md`

**Files:**
- Modify: `torchtitan/observability/state_estimator/learned.py`
- Create: `torchtitan/observability/state_estimator/features.py`
- Test: `tests/unit_tests/observability/state_estimator/test_learned.py`

**Interfaces:**
- Produces `FeatureSegment`, `LearnedModelIdentity`, `LearnedFactorBundle`,
  `extract_feature_segments`, `load_learned_factor_bundle`.

**Acceptance:**
- Extracts feature segments from normalized observations without ML imports.
- Loads learned factor JSON bundles with model identity, training-data manifest,
  calibration state, advisory flag, and source segments.
- Rejects learned factors that omit advisory/calibration metadata.
- Does not import torch, transformers, graph libraries, neural CDE/ODE, or GPU
  libraries by default.

**Verification:**

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_learned.py'
```

## Task 19: Evaluation Corpus And Calibration

**Issue:** `issues/19-evaluation-corpus-calibration.md`

**Files:**
- Modify: `torchtitan/observability/state_estimator/evaluation.py`
- Create: `tests/unit_tests/observability/state_estimator/test_evaluation.py`
- Create: `.scratch/training-state-estimator/evaluation_manifest_schema.md`

**Interfaces:**
- Produces `EvaluationManifest`, `EvaluationMetricSummary`,
  `compute_calibration_metrics`, `write_calibration_report`.

**Acceptance:**
- Supports synthetic, CPU/fake-backend, real small-run, and injected-fault case
  classes.
- Reports detection latency, false alerts, expected mode found, expected probe
  found, top-k root cause, failure-domain scope score, observability-warning
  correctness, Brier score, expected calibration error, artifact bytes, and
  optional overhead metrics.
- Separates detection, localization, action/probe, and calibration claims.

**Verification:**

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_evaluation.py'
```

## Task 20: Operator CLI And Reports

**Issue:** `issues/20-operator-cli-reports.md`

**Files:**
- Create: `torchtitan/observability/state_estimator/reports.py`
- Modify: `torchtitan/observability/state_estimator/cli.py`
- Test: `tests/unit_tests/observability/state_estimator/test_reports.py`
- Test: `tests/unit_tests/observability/state_estimator/test_cli.py`

**Interfaces:**
- Produces CLI subcommands `analyze-attempt`, `analyze-run`,
  `compare-attempts`, and `evaluate`.

**Acceptance:**
- CLI supports a separate output directory.
- CLI emits JSON stdout with all derived artifact paths.
- Reports include sections for evidence quality, topology, timeline,
  analytical residuals, mode scores, root-cause candidates, transaction risk,
  probes, learned factors, and evaluation metrics.
- Malformed required evidence exits nonzero with actionable JSON error.

**Verification:**

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_reports.py tests/unit_tests/observability/state_estimator/test_cli.py'
```

## Task 21: Evidence Producer Roadmap And Core Hooks

**Issue:** `issues/21-evidence-producer-roadmap-core-hooks.md`

**Files:**
- Modify: `docs/run_evidence.md`
- Modify: `.scratch/training-state-estimator/remaining_work_spec.md`
- Optional code changes only through child tickets.

**Interfaces:**
- Produces a roadmap of core producer fields and separate child tickets.

**Acceptance:**
- Specifies the minimum core structured-event fields needed for topology,
  microbatch, collective, checkpoint, data cursor, and transaction evidence.
- Separates documentation-only roadmap from any core code implementation.
- For every proposed core hook, lists overhead risk, owner component,
  validation command, and fallback if absent.
- Creates child tickets before any core `Trainer` or run-evidence code change.

**Verification:**

```bash
git diff --check -- docs/run_evidence.md .scratch/training-state-estimator
```

## Final Verification

After all tasks complete:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator'
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/test_run_evidence.py tests/unit_tests/observability/test_structured_logging.py'
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m py_compile torchtitan/observability/state_estimator/*.py'
git diff --check -- torchtitan/observability/state_estimator tests/unit_tests/observability/state_estimator .scratch/training-state-estimator docs/run_evidence.md
```

## Plan Self-Review

- Spec coverage: tasks map to evidence substrate, topology graph, timeline,
  analytical models, robust inference, transactions, probes, learned factors,
  evaluation, CLI, and producer roadmap.
- Safety boundary: no task authorizes automatic actuation or transaction
  decisions.
- Rootfs boundary: all Python verification commands are rootfs-wrapped.
- Subagent boundary: every task requires implementer, reviewer, and finalizer
  traces.
- Dependency boundary: optional tools remain optional artifacts or summaries.
