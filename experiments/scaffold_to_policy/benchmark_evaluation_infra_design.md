# Benchmark Evaluation Infrastructure Design

Status: draft design for the scaffold-to-policy benchmark evaluation program.

This document designs the shared infrastructure needed to run benchmark
evaluations and audit their results. It consolidates requirements already
spread across `runtime_preflight_roadmap.md`, `reasoning_research_plan.md`,
`coding_research_plan.md`, `agentic_eval_benchmarks.md`, and the full-matrix
report. It does not replace benchmark-specific research plans or upstream
benchmark definitions.

## 1. Objective

Build a reusable evaluation layer that can cover reasoning, coding, tool-use,
terminal, and agentic benchmarks while making every result auditable.

The infrastructure must answer five questions for every reported score:

1. What exact benchmark object was evaluated?
2. What exact model, adapter, harness, tools, rootfs, and runtime condition ran?
3. Did the canonical benchmark contract pass before model evaluation?
4. Did execution produce a real measurement, a fixture/plumbing measurement, an
   invalid measurement, or a typed blocker?
5. Can an independent reviewer recompute the report from immutable artifacts?

The primary design goal is not more scripts. It is a narrow shared interface
that forces every benchmark to provide enough evidence for audit, while leaving
task semantics inside benchmark adapters and upstream harnesses.

## 2. Scope

In scope:

- benchmark declarations, condition identity, split registries, and evaluator
  identity;
- rootfs doctor and profile selection for every real run;
- canonical-solution and benchmark-semantic preflights;
- task import, split validation, contamination checks, and answer-position
  audits;
- model/harness execution through typed lifecycle attempts;
- artifact hashing, freshness, lineage, and replay checks;
- score classification as execution outcome, measurement state, and promotion
  state;
- representative examples and failure taxonomies;
- report-input validation and immutable Markdown report generation;
- audit commands that fail closed when required evidence is missing.

Out of scope:

- replacing upstream benchmark metrics with generic LLM judging;
- changing benchmark semantics to make a model look better;
- committing large generated artifacts, model weights, or rollouts;
- moving experiment dependencies into core TorchTitan;
- providing a general cluster scheduler;
- claiming coverage for benchmarks that do not have local entrypoints.

## 3. Current Baseline

The repo already has useful foundations:

- `scripts/rootfs/enter_rootfs.sh` enforces the bwrap rootfs for real model and
  benchmark work.
- `experiments/scaffold_to_policy/run_common.sh` provides rootfs re-entry,
  runtime metadata capture, and stage wrappers.
- `run_runtime_doctor.sh` verifies rootfs, packages, model assets, CUDA, vLLM,
  and selected harness requirements.
- Typed lifecycle attempts now exist for several public vLLM smokes.
- Legacy report inputs exist for older smokes and harness probes.
- The full matrix report separates typed results, report-only smokes, typed
  blockers, and compatibility probes.

Known gaps:

- Some runners still produce only report inputs, not typed attempt bundles.
- Benchmark declarations are implicit in script arguments and output paths.
- Evaluation IDs are not uniformly derived from the full condition.
- Artifact freshness is partly inferred from path names rather than producer
  receipts.
- Canonical preflight coverage is uneven across benchmark families.
- Report builders do not all fail closed on missing lifecycle or audit data.
- External harness probes are not yet model-agent evaluations.
- GPQA needs a permanent answer-position and permutation audit gate before
  scores can be trusted.

## 4. Design Principle

Use one deep shared module for evaluation evidence, with small benchmark
adapters at the seam.

Benchmark adapters own:

- acquiring or locating tasks;
- preserving upstream task semantics;
- rendering prompts or harness inputs;
- invoking official verifiers or exact local verifiers;
- explaining benchmark-specific limitations.

Shared evaluation infrastructure owns:

- immutable declarations;
- lifecycle attempts;
- preflight profiles;
- artifact receipts;
- audit rules;
- result classification;
- report-input validation;
- promotion/futility decisions.

This avoids a shallow design where every runner hand-rolls identity, path
layout, freshness checks, blocker handling, and report generation.

## 5. Module Layout

Shared Python should live under experiment-only packages:

```text
torchtitan/experiments/evaluation/
  declarations.py
  identities.py
  registry.py
  artifacts.py
  audits.py
  report_inputs.py
  promotion.py
  adapters/
    reasoning.py
    coding.py
    external_harness.py
```

Lifecycle and rootfs execution should continue to live under the execution
foundation described by `runtime_preflight_roadmap.md`:

```text
torchtitan/experiments/execution/
  lifecycle/
  preflight/
  rootfs/
  reporting/
```

Top-level shell entrypoints remain under:

```text
experiments/scaffold_to_policy/run_*.sh
```

The shell entrypoints should become thin declarations plus adapter invocation.
They should not own audit logic.

## 6. Core Interfaces

### 6.1 BenchmarkDeclaration

One immutable benchmark object.

Required fields:

- benchmark name and upstream release;
- dataset revision or source revision;
- task manifest hash;
- official evaluator or local verifier name and version;
- allowed tools and action space;
- split names and split registry hash;
- metric semantics;
- known limitations.

Changing any field changes the declaration digest.

### 6.2 EvaluationCondition

One runnable condition over a declaration.

Required fields:

- model, tokenizer, base checkpoint, adapter, or merged policy hash;
- vLLM or harness runtime profile;
- rootfs digest and dependency lock;
- prompt renderer, system prompt hash, scaffold, and output contract;
- sampling parameters and rollout budget;
- tool schemas, action budgets, context policy, and retry policy;
- hardware constraints and timeout limits;
- source revision and dirty-worktree summary.

The normalized condition receives an `evaluation_id`. A report table row should
always be traceable to this ID.

### 6.3 BenchmarkAdapter

Small benchmark-specific interface:

```text
acquire(declaration) -> TaskManifest
preflight(declaration, condition) -> PreflightReport
run(condition, attempt) -> RawOutputs
score(raw_outputs) -> EvaluationRows
examples(evaluation_rows) -> RepresentativeExamples
```

The shared infrastructure calls this interface. The adapter may delegate to an
upstream harness such as tau2, Harbor/Terminal-Bench, or future benchmark
packages. When an upstream harness owns the metric, the adapter records harness
inputs and outputs; it does not reinterpret the score.

### 6.4 AuditResult

Audits are machine-readable checks over declarations, conditions, artifacts,
and report inputs.

Every audit emits:

- `audit_id`;
- `status`: `pass`, `warn`, or `fail`;
- `scope`: declaration, split, runtime, execution, metric, report, or promotion;
- evidence artifact paths and hashes;
- human-readable finding;
- blocking decision.

Any blocking failure prevents a promoted result. Some failures also prevent
model execution, for example canonical-solution failure or missing rootfs
requirements.

### 6.5 ReportInput

The canonical machine-readable summary. It must keep these axes separate:

```text
execution_outcome: completed | blocked | failed | interrupted
measurement: real | fixture | smoke | invalid | not_run
promotion: promote | hold | reject | not_evaluated
```

A model that runs and scores zero is:

```text
execution_outcome=completed
measurement=real
promotion=reject or hold
```

A missing dependency or gated dataset is:

```text
execution_outcome=blocked
measurement=not_run
promotion=not_evaluated
```

An answer-position bug like the historical GPQA importer is:

```text
execution_outcome=completed
measurement=invalid
promotion=reject
```

## 7. Artifact Layout

Use the run-attempt bundle from the lifecycle roadmap:

```text
<results-root>/runs/<run-id>/<attempt-id>/
  manifest.json
  processes/<process-id>/events.jsonl
  artifacts/<producer>/...
  indexes/artifacts.<process-id>.jsonl
  incidents/<incident-id>.json
  outcome.json
  derived/report_input.json
  derived/audit_report.json
```

Benchmark-specific task artifacts live under data roots:

```text
experiments/scaffold_to_policy/data/<benchmark>/
  raw/
  imported/
  split_registry.json
  declarations/<declaration-id>.json
```

Runtime artifacts stay ignored by git. Reports and small declarations may be
committed.

## 8. Required Audit Gates

### 8.1 Runtime Gate

Required before real generation, training, or benchmark-harness execution:

- entered via `scripts/rootfs/enter_rootfs.sh`;
- rootfs path and digest recorded;
- required Python packages import inside rootfs;
- GPU count and selected devices recorded;
- model assets exist and hash or manifest is captured;
- vLLM configuration and memory headroom recorded when vLLM is used;
- external binaries such as Docker, Harbor, or tau2 CLIs are recorded when used.

### 8.2 Declaration Gate

Required before any model run:

- benchmark release is pinned;
- task manifest hash is frozen;
- split registry exists and has stable problem IDs;
- exact duplicate IDs and content hashes are absent across forbidden splits;
- benchmark-specific contamination policy ran;
- metric semantics and verifier version are recorded.

### 8.3 Canonical Preflight Gate

Required before model scoring:

- canonical answers pass the verifier;
- known invalid fixtures fail the verifier when available;
- prompt/rendered input fits the declared context budget;
- output parser has adversarial fixtures for common malformed outputs;
- for executable coding, candidate execution isolation gate is satisfied;
- for multiple choice, answer positions are balanced or intentionally declared,
  and verifier semantics are invariant under choice relabeling.

### 8.4 Execution Gate

Required after model or harness execution:

- every stage has lifecycle events and a terminal status;
- raw outputs are present and hashed;
- per-problem evaluation rows are present;
- blocked, failed, interrupted, and timed-out conditions are distinct;
- zero-score real measurements are not converted into infrastructure failures;
- retries and resumes are linked to prior attempts.

### 8.5 Report Gate

Required before a Markdown report is considered audit-ready:

- report input validates against schema;
- all referenced artifact paths exist and hashes match;
- execution, measurement, and promotion states are separated;
- pass@k values do not exceed the actual rollout budget;
- fixture, smoke, report-only, and real measurements are labeled;
- representative examples are attached for success, dominant failure, and
  borderline cases when available;
- every caveat that blocks promotion is machine-readable, not just prose.

### 8.6 Promotion Gate

Required before claiming improvement:

- baseline and candidate use matched benchmark declarations;
- cells differ only in declared treatment variables;
- problem identities are disjoint from training targets;
- enough seeds/draws exist for the intended claim level;
- uncertainty or per-cell effects are reported;
- hard negatives and futility decisions are preserved;
- no opened or invalid set is promoted as locked confirmation.

## 9. Benchmark Family Requirements

### 9.1 Static Reasoning

Examples: arithmetic words, modular sequences, GSM8K, MATH, AIME, GPQA,
MMLU-Pro, ARC-AGI-2.

Additional audits:

- exact-answer or task-specific verifier fixtures;
- strict-format success separated from task success;
- prompt contract and parser version recorded;
- multiple-choice permutation audit where applicable;
- opened/public/post-hoc status recorded for every split.

### 9.2 Executable Coding

Examples: HumanEval, MBPP, LiveCodeBench, BigCodeBench-Hard, SciCode.

Additional audits:

- canonical solution passes;
- generated code execution isolation gate;
- timeout, memory, output, and package policy;
- test artifact hashes;
- public-test versus hidden/private-test status;
- extraction/rescore version and linkage to original generations.

### 9.3 External Harnesses

Examples: tau2, Terminal-Bench/Harbor, future ToolSandbox or BFCL.

Additional audits:

- upstream harness revision and package lock;
- container image digest or rootfs profile;
- tool schema hash;
- system prompt and context policy hash;
- action and wall-clock budgets;
- final-state or official score artifact;
- trajectory capture and replayability;
- distinction between oracle/scripted/noop/model-agent runs.

### 9.4 Long Context And Memory

Examples: RULER, LongBench, BEAM, LongMemEval.

Additional audits:

- no-tools versus retrieval condition;
- context packing policy;
- truncation or summarization policy;
- exact document/dialogue/task manifest hash;
- abstention and contradiction handling semantics where benchmark defines them.

## 10. Report Types

Use report types rather than ad hoc prose labels:

- `runtime_doctor`: environment readiness only.
- `contract`: import, split, verifier, canonical, and report-schema readiness.
- `smoke`: tiny model or fixture run for plumbing.
- `calibration`: base model measurement on a declared evaluation slice.
- `transfer`: trained/adapted policy compared with matched base.
- `external_harness_probe`: install/import/scorer/oracle compatibility.
- `agentic_capability`: model-agent run under upstream harness semantics.
- `audit`: invalidation, contamination, answer-position, or report-integrity
  review.
- `promotion_decision`: machine-computed promote/hold/reject/futility outcome.

Every report should name exactly one primary type.

## 11. Command Shape

Desired CLI shape:

```bash
experiments/scaffold_to_policy/evalctl declare --benchmark <name> --config <file>
experiments/scaffold_to_policy/evalctl doctor --profile <profile>
experiments/scaffold_to_policy/evalctl preflight --declaration <id> --condition <id>
experiments/scaffold_to_policy/evalctl run --condition <id>
experiments/scaffold_to_policy/evalctl audit --attempt <path>
experiments/scaffold_to_policy/evalctl report --attempt <path>
```

Shell runners can remain stable wrappers around this command family. Existing
`run_*.sh` names can continue to work while migrating to the new implementation.

## 12. Failure Model

The infrastructure should fail closed, but classify failures precisely:

- `blocked`: prerequisite absent, access denied, canonical preflight failed, or
  runtime profile not selectable.
- `failed`: infrastructure attempted work but crashed unexpectedly.
- `interrupted`: external stop, timeout at attempt level, or host/process loss.
- `completed/real/score_zero`: model ran and verifier scored zero.
- `completed/invalid`: execution finished but audit invalidated the metric.
- `completed/fixture`: fixture or oracle path completed and is useful only as
  plumbing evidence.

Reports must not aggregate these into one "failure" bucket.

## 13. Audit Examples

Historical GPQA bug:

- Declaration gate passes only after shuffled answer positions are balanced.
- Canonical preflight includes a permutation-invariance test.
- Old correct-first results become `measurement=invalid`.
- Markdown reports remain immutable but the audit report points at every
  invalid artifact and excludes those metrics from promotion.

BigCodeBench-Hard zero:

- Runtime gate passes.
- Canonical solutions pass.
- Model outputs execute and fail tests.
- Result is `completed/real` with score zero, not a blocker.
- Promotion state is `reject` or `hold` depending on campaign criteria.

Terminal-Bench oracle probe:

- Harness and verifier run.
- Agent is oracle, noop, or scripted.
- Result is `completed/fixture` or `external_harness_probe`.
- It cannot be reported as a model-agent capability score.

## 14. Implementation Phases

### Phase A: Contract consolidation

- Define JSON schemas for declarations, conditions, audit results, and report
  inputs.
- Add a validator command that checks existing typed attempts and report inputs.
- Add report-type metadata to new reports.
- Document which existing runs are legacy imports.

Exit: the full-matrix report can be revalidated by command without rerunning
benchmarks.

### Phase B: Lifecycle migration

- Migrate all legacy report-only smokes to typed run-attempt bundles.
- Require runtime doctor attachment for every real model/harness run.
- Add artifact producer receipts instead of path-name freshness inference.

Exit: every local runner produces `outcome.json`, `derived/report_input.json`,
and `derived/audit_report.json`.

### Phase C: Benchmark contract gates

- Add generic multiple-choice answer-position and permutation audits.
- Add canonical-solution and parser-fixture gates for every reasoning/coding
  family.
- Add executable-code isolation gate before any untrusted code claim.
- Add external-harness declaration records for tau2 and Harbor/Terminal-Bench.

Exit: GPQA can be repaired and rerun, and coding scores are blocked unless the
execution isolation gate selects.

### Phase D: Model-agent harness coverage

- Replace tau2 and Terminal-Bench oracle/fixture probes with bounded model-agent
  runs.
- Record tool schema, action budget, trajectory, final-state score, and replay
  artifacts.
- Keep official harness score as primary.

Exit: at least one tau2 and one Harbor/Terminal-Bench model-agent row is a real
measurement, even if score zero.

### Phase E: Promotion automation

- Add machine-computed promote/hold/reject/futility decisions.
- Enforce matched base/candidate declarations.
- Add seed/draw replication validators.
- Generate campaign-level audit reports.

Exit: a transfer claim cannot be written unless the promotion audit selects.

## 15. Immediate Next Tickets

1. Add declaration/condition/report-input schema validators and run them over
   `20260814T071500Z-full-matrix`.
2. Migrate legacy report-only smokes to typed lifecycle attempts.
3. Implement multiple-choice answer-position and permutation audits, then
   repair GPQA.
4. Add executable-code isolation gate and make coding runners refuse promoted
   scoring without it.
5. Add tau2 and Harbor model-agent declarations with official-harness score
   ingestion.
6. Add `evalctl audit` and `evalctl report` wrappers that rebuild audit and
   report inputs from existing attempt bundles.

## 16. Acceptance Criteria

The evaluation infrastructure is strong enough when:

- a fresh checkout plus rootfs can validate a committed report against ignored
  local artifacts;
- every real result has a frozen declaration, evaluation ID, runtime doctor,
  attempt bundle, report input, and audit report;
- every missing dependency or gated dataset becomes a typed blocker;
- every valid zero remains a valid zero;
- every invalid measurement is excluded from promotion by machine-readable
  audit, not by reviewer memory;
- every external benchmark score names the model, harness, tools, rootfs,
  benchmark release, evaluator, action budget, and report type;
- adding a new benchmark requires writing a small adapter, not rebuilding
  provenance, lifecycle, audit, and report machinery.
