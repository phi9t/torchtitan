# Scaffold-To-Policy Long-Horizon Research Program

Status: approved research design; implementation and campaign execution remain
gated by the milestones below.

This is the authoritative master plan for the scaffold-to-policy program. It
coordinates the shared execution foundation, offline SFT, online RL, reasoning,
coding, and downstream agentic evaluation without treating plumbing smokes as
scientific evidence.

This program is Phase 4, post-training, of the repository-wide training
research vehicle. Its implementation sequence begins only after the prior
gates in
[`docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md`](../../docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md):

1. the core `Trainer` owns the canonical run-attempt observability contract and
   passes its overhead and fault-injection gates;
2. Qwen3-0.6B and 1.7B pass matched 2-, 4-, and 8-GPU configuration,
   numerical, checkpoint, convergence, performance, and observability gates,
   followed by the required 8B and 30B-A3B foundation promotions; and
3. the native and pluggable multimodal phase reaches its own approved boundary.

The scaffold lifecycle adopts and extends the core evidence schema for
experiment stages; it does not create an earlier or competing observability
foundation. Planning, task-contract audits, and preservation of known-invalid
history may proceed now, but post-training implementation and compute must
respect the repository-level phase gates.

## 1. Document Map And Authority

Each document has one job:

- [`runtime_preflight_roadmap.md`](runtime_preflight_roadmap.md) owns execution
  identity, lifecycle events, rootfs capabilities, preflight, local Temporal,
  tracing, failure semantics, and foundation implementation waves.
- [`training_research_plan.md`](training_research_plan.md) owns verified data,
  TorchTitan SFT, checkpoint recovery, policy export, deterministic RL, async
  RL, and training experiment cells.
- [`reasoning_research_plan.md`](reasoning_research_plan.md) owns the reasoning
  benchmark ladder, task-specific verifiers, calibration bands, hypotheses,
  and promotion gates.
- [`coding_research_plan.md`](coding_research_plan.md) owns executable coding,
  sandboxing, benchmark-preserving harnesses, coding transfer, and agentic
  coding gates.
- [`temporal_durable_execution_research.md`](temporal_durable_execution_research.md)
  records the primary-source rationale and constraints for the local Temporal
  adapter.
- [`spec.md`](spec.md) is the executed historical checkpoint and evidence
  contract. It is not the forward schedule.
- [`README.md`](README.md) is the concise entrypoint plus an inventory of
  implemented commands and timestamped reports.
- The validated repo-local run-attempt bundle and raw artifacts are canonical
  for what an individual run did. Timestamped files under [`reports/`](reports/)
  are immutable compact interpretations linked to that evidence.

If a report conflicts with its validated bundle or raw artifact, the canonical
evidence wins and the discrepancy is recorded. If a historical statement
conflicts with a later scientific correction in this plan, retain the
historical artifact but use this plan's corrected status for future decisions.
No plan may retroactively upgrade a smoke, fixture, reused artifact, or invalid
evaluation into scientific evidence.

## 2. Program Objective

The program tests a sequence of increasingly strong claims:

1. A scaffold can expose verified behavior that the unassisted policy does not
   reliably produce at pass@1.
2. Offline training on verified scaffold traces can compress some of that
   behavior into the policy.
3. The gain is causal, replicable, and not merely stricter formatting, leakage,
   answer-position bias, or a different inference configuration.
4. The compression mechanism transfers from local exact-verifier tasks to
   public reasoning and executable coding benchmarks.
5. Deterministic online RL can improve the same verified objective from a base
   or SFT initialization without numerical or policy-age ambiguity.
6. Controlled asynchronous RL preserves enough of the gain to justify its
   additional throughput and freshness complexity.
7. A selected policy improves a small, pinned downstream agentic suite under
   the official external harness, rather than only the local proxy tasks.

The goal is not to maximize the number of benchmark cells. The goal is to learn
which scaffold information is compressible, under which training regimes, and
where transfer stops.

## 3. Current Evidence And Corrections

### 3.1 Evidence that can guide the next run

- Countdown has replicated task-specific Qwen3-1.7B SFT evidence under its
  historical clean-split gate, including fresh clean and formatting-arm runs.
  Its principal runs couple split draw and training seed rather than fully
  crossing `2 draws x 2 seeds`, so it is a historical replication anchor, not
  a new-ladder M4 result or evidence of cross-task transfer. Its next work is
  lifecycle migration and stop-quality analysis, not another broad rank sweep.
- The expanded modular-sequences transfer is a useful one-draw causal lead:
  raw LoRA moved dev pass@1 from `0.531` to `0.625` and the historically named
  `ood_test` pass@1 from `0.375` to `0.594`. That second split changed the seed,
  not the recurrence parameter band, so it is held-out IID rather than genuine
  distribution-shift evidence. The result improved the observed pass@k curve
  and earns a replicated rerun with explicit length- and modulus-shift OOD
  axes; it does not establish a robust program-level claim.
- Arithmetic words, modular sequences, GSM-style, GSM8K, MATH, AIME,
  MMLU-Pro, ARC-AGI-2, MBPP, HumanEval, LiveCodeBench public cases,
  BigCodeBench-Hard, Harbor/Terminal-Bench, and tau2 have varying amounts of
  import, verifier, rootfs, or model-execution plumbing. Their timestamped
  reports define the exact scope.
- Several hard suites produced zero or low scores. A completed, valid zero is a
  real hard-negative calibration, not an infrastructure blocker. It becomes a
  blocker only when execution or measurement was invalid.
- Existing agentic probes are useful compatibility evidence. No current noop,
  mock, dry-run, or tiny model-policy probe is an agentic capability claim.

Legacy report fields sometimes expose pass@16 or pass@32 after only 4 or 8
rollouts by clamping the requested `k` to the available samples. Preserve those
fields as historical compatibility output, but scientific tables stop at the
actual rollout budget. A split named `ood_test` is likewise not OOD evidence
unless its declared transformation changes the data-generating distribution.

### 3.2 GPQA scientific invalidation

The current GPQA importer places the correct answer first and labels it `A` for
every imported row. Existing GPQA model scores therefore confound reasoning
with answer-position bias and are scientifically invalid. Preserve the old
reports as execution history, but set their measurement status to `invalid` for
promotion purposes.

Before any GPQA result is used:

1. deterministically shuffle choices per stable problem identity and declared
   shuffle seed;
2. demonstrate near-balanced correct-answer positions for each evaluated
   slice;
3. test that reordering choices and relabeling the expected answer preserves
   verifier semantics;
4. rebuild all cached problems and split hashes;
5. rerun base calibration before generating or training on GPQA traces.

The repaired Diamond result is post-hoc/opened evidence, not a locked transfer
confirmation: its questions and defective-mapping outcomes were already
inspected. A confirmatory multiple-choice claim requires an untouched benchmark
or authorized split frozen before policy selection.

This correction also motivates a generic multiple-choice preflight for answer
position histograms and permutation invariance.

### 3.3 Claims not yet supported

The current record does not yet support:

- replicated scaffold-to-policy improvement on a second task family beyond the
  Countdown reference;
- a public reasoning transfer gain from SFT;
- a coding transfer gain from SFT or RL;
- parity-validated deterministic online RL;
- safe retry of a joint trainer/generator RL run;
- an official LiveCodeBench score;
- a SciCode result;
- an agentic capability gain on Terminal-Bench/Harbor or tau2; or
- a scale claim for Qwen3-8B or Qwen3-30B-A3B.

## 4. Non-Negotiable Program Contracts

### 4.1 Surface and dependency direction

This is a repo-local research program. Reusable experiment Python belongs under
`torchtitan/experiments/`; rootfs runners, declarations, compact reports, and
run artifacts belong under top-level `experiments/`. Dependencies flow from
experiments to core.

- Core SFT remains `run_train.sh -> torchrun -> torchtitan.train -> Trainer`.
- Online RL remains `python -m torchtitan.experiments.rl.train` and provisions
  its own Monarch trainer and generator meshes. It is never wrapped in
  `torchrun`.
- Experiment dependencies, including Temporal and external harnesses, do not
  enter the PyTorch-native core.

### 4.2 Evidence identity

- A `run_id` identifies one immutable scientific declaration.
- An `attempt_id` spans one top-level operational execution or retry of that
  run and ends once with an attempt outcome.
- A `stage_invocation_id` identifies each actual stage subprocess/job launch;
  Activity redelivery that only reconciles a receipt creates no new invocation.
- Temporal workflow and activity IDs are correlations, not artifact authority.
- Stage events are append-only and include one start plus exactly one terminal
  outcome per attempt-stage execution.
- Artifacts prove freshness through producer identity, content hashes, and
  before/after evidence. A run-specific pathname alone is insufficient.
- One canonical report input is enriched with execution evidence. There is no
  second competing result-manifest schema.

Three statuses remain independent for every evaluation ID/cell/split:

- execution: `completed`, `blocked`, `failed`, or `interrupted`;
- measurement: `real`, `fixture`, `smoke`, `invalid`, or `not_run`;
- promotion: `promote`, `hold`, `reject`, or `not_evaluated`.

A real regression normally has `execution=completed`, `measurement=real`, and
`promotion=reject`. It must not be mislabeled as a runtime failure.
The enclosing attempt has a separate derived outcome, so a completed condition
and blocked later condition can coexist truthfully.

### 4.3 Scientific evidence

- Train, development, IID, OOD, and locked-transfer problem identities are
  globally disjoint unless an upstream benchmark explicitly defines otherwise.
- Dataset revision, row identity, transformation version, prompt contract,
  verifier version, split hash, and exclusion policy are recorded.
- Task success and strict output-format compliance are separate metrics.
- Base, adapter or full policy, scaffold subset, failure-mode, and
  representative-example evidence are required.
- Smoke runs prove plumbing only.
- Promotion claims require seed or split-draw replication and uncertainty over
  the problem as the primary statistical unit.

### 4.4 Rootfs and security

All real generation, training, export, evaluation, and official harness work
uses the repo-local bwrap rootfs runners. Host-side unit and static checks may
run directly.

The hardened target is a content-identified, read-only base rootfs with
declared writable mounts and explicit capabilities for acquisition network,
local-only distributed networking, host-network compatibility, GPUs, Docker,
host mirrors, and model images. `local_distributed_network` permits required
loopback/Unix-socket/single-node rendezvous traffic while denying external
egress. Networked asset acquisition is a separate stage from offline scientific
execution.

Credentials are explicit stage capabilities delivered through invocation-only
read-only files or file descriptors, never embedded in declarations, argv,
logs, Temporal payloads, or environment dumps. Remote trackers receive only a
versioned allowlist of derived scalars and artifact references after redaction
scanning; local evidence remains sufficient when every exporter is disabled.

Python `subprocess` isolation is not a security sandbox. Untrusted generated
code must not execute until the coding sandbox gate in the coding plan passes.

### 4.5 Durable execution

Temporal is an optional host-side orchestration adapter for long-running staged
campaigns. Workflow code is deterministic and carries only small references.
Filesystem, hashing, subprocess, bwrap, CUDA, and model operations are coarse,
idempotent Activities. Local artifacts, checkpoints, and attempt bundles remain
the source of truth.

SFT can retry from a verified distributed checkpoint. Online RL cannot retry
automatically until joint controller recovery is implemented and fault-tested.

## 5. Two-Dimensional Research Ladder

Every benchmark family advances independently through scientific maturity. A
hard task with only a valid base calibration is not considered more mature than
an easy task with replicated training evidence.

### 5.1 Scientific maturity

| Level | Name | Required evidence |
| --- | --- | --- |
| M0 | Contract | Pinned source, stable identities, split policy, verifier tests, prompt contract, report schema, and semantic preflight. |
| M1 | Base calibration | Real model run, valid difficulty band, pass@k or official metric, failure census, and representative examples. |
| M2 | Scaffold reachability | Verified scaffold improves coverage over pass@1 with useful elicitable cases and an explicit compute budget. |
| M3 | SFT pilot | Matched base-versus-trained causal pilot on disjoint data; plumbing and direction only. |
| M4 | Replicated SFT | Champion, frozen base, and causal anchor across at least two fresh split draws and two training seeds, with cellwise paired uncertainty. This minimum matrix supports a claim conditional on the realized policies, not population-level draw/seed inference. |
| M5 | Deterministic RL | Strict on-policy RL with trainer/generator numerical parity and causal base/SFT initialization comparisons. |
| M6 | Controlled async RL | One async factor changed at a time, policy-age evidence, phase timings, and recovery semantics. |
| M7 | Scale confirmation | Selected mechanism and causal control reproduce at a promoted larger model with preregistered scale-level seed replication and paired final-set uncertainty. A single larger-model run is a scale probe, not M7. |

No benchmark skips M0. A benchmark may stop permanently at any level if the
scaffold has no reachability, the verifier is unreliable, or the effect does
not replicate.

### 5.2 Difficulty and interaction bands

Use these bands for scheduling, not as a claim of a universal ordering:

| Band | Reasoning examples | Coding examples | Purpose |
| --- | --- | --- | --- |
| L0 | Countdown, arithmetic words, modular sequences | Local code fixtures | Unit and mechanism lab. |
| L1 | GSM8K, selected MATH-exact | MBPP | First public transfer with cheap verification. |
| L2 | AIME, MMLU-Pro | HumanEval, canonical BigCodeBench-Hard | Hard no-tool transfer. |
| L3 | GPQA after repair, ARC-AGI-2 | Official LiveCodeBench, SciCode | Distribution, temporal, and domain stress. |
| L4 | tau2 pinned tasks | Terminal-Bench through Harbor | Stateful or environment-mediated downstream evaluation. |

The portfolio target is sparse: establish M4 on one L0 mechanism task, M3-M4
on one L1 public task, and M1-M2 on selected L2/L3 tasks before investing in
M5-M7.

Downstream transfer is orthogonal to model scale and uses its own evidence
gate:

| Gate | Name | Required evidence |
| --- | --- | --- |
| X0 | Harness contract | Official harness identity, task/split policy, secure execution contract, and trajectory/report schema are valid. |
| X1 | Base calibration | The frozen base runs on the pinned suite with complete validity, failure, cost, and trajectory evidence. |
| X2 | Selected-policy pilot | One policy selected without downstream outcomes is compared with base on an opened development subset. |
| X3 | Replicated transfer | A frozen comparison is repeated over prospectively sized independent task draws when claiming task-distribution transfer, or over policy seeds with the claim explicitly limited to one fixed suite. |

Maturity and downstream transfer must be reported separately, for example
`M4/X2`; an official downstream run cannot substitute for scale replication.

## 6. Portfolio And Advancement Policy

### 6.1 Branches

The reasoning portfolio has four branches:

1. mechanism lab: Countdown and modular sequences;
2. exact public reasoning: GSM8K -> `MATH-exact-v1` -> AIME;
3. multiple-choice knowledge/reasoning: MMLU-Pro -> repaired GPQA;
4. structured abstraction: ARC-AGI-2.

The coding portfolio has four branches:

1. function synthesis: MBPP train -> HumanEval locked transfer;
2. repository-independent hard functions: BigCodeBench-Hard;
3. temporal and hidden-test coding: official LiveCodeBench;
4. scientific software: SciCode, followed by Terminal-Bench/Harbor as the
   environment-mediated downstream suite.

tau2 is a downstream reasoning/tool-protocol suite. It is not a replacement for
the no-tool reasoning benchmarks.

### 6.2 Champion/challenger rule

At each maturity transition:

- retain one current champion;
- retain one causal anchor, normally base or raw SFT;
- admit at most two challengers;
- eliminate cells that cannot answer a distinct hypothesis; and
- advance only the champion plus causal anchor to the next expensive tier.

Do not cross every scaffold, data cleaning rule, model size, LoRA rank, prompt,
training seed, and benchmark. Small factorial studies are allowed only where
an interaction is the hypothesis.

### 6.3 Compute funnel

| Tier | Typical resource | Purpose | Evidence ceiling |
| --- | --- | --- | --- |
| C0 | Host CPU | Schema, verifier, importer, lifecycle, and fault unit tests. | M0 |
| C1 | Rootfs CPU | Hermetic imports, report validation, Temporal activities, and sandbox wiring. | M0 |
| C2 | 1 GPU, Qwen3-0.6B | End-to-end SFT/RL/load plumbing and tiny deterministic checks. | Smoke only unless separately powered. |
| C3 | 1 GPU, Qwen3-1.7B | Base/scaffold calibration and selected inference cells. | M1-M2 |
| C4 | Sequential 8 GPU, Qwen3-1.7B | Primary SFT, deterministic RL, recovery, and matched performance. | M3-M6 |
| C5 | Certified Qwen3-8B, selected cells | One selected scale probe after mechanism selection and upstream 8B promotion; M7 only after the replicated scale gate. | Probe or M7 |
| C6 | Certified Qwen3-30B-A3B, separately reviewed | Optional MoE probe or replicated confirmation after upstream MoE promotion and a resource/parallelism review. | Separate campaign |

Routine multi-node execution is out of scope. The default local scheduler
serializes 8-GPU stages. Concurrency is reserved for CPU work or explicitly
non-overlapping 1-GPU leases.

## 7. Program Dependency Graph

```text
research vehicle P1: core Trainer observability and evidence contract
  -> P2: Qwen3 0.6B/1.7B on 2/4/8 GPUs, then 8B and 30B-A3B
     -> P3: native and pluggable multimodal phase
        -> P4 scaffold-to-policy program

P4 F0 truth repair and core evidence-schema adoption
  -> F1 lifecycle and evidence identity
     -> F2 composable doctor and semantic preflight
        -> F3 rootfs identity and capability hardening
           -> F4 runner migration
              -> F5 local Temporal adapter
                 -> F6 experiment views and operations

M0 task contract -> M1 base calibration -> M2 scaffold reachability
                                      \-> verified trace dataset
                                           -> M3 SFT pilot
                                              -> M4 replicated SFT
                                                 -> policy bridge
                                                    -> M5 deterministic RL
                                                       -> M6 async RL
                                                          -> M7 scale

coding M0 -> secure execution gate -> any real generated-code measurement
GPQA M0 -> choice randomization repair -> rebuilt base calibration
external harness X0/X1 -> selected-policy pilot -> X3 downstream claim
```

Foundation waves may overlap task-contract work, but no expensive scientific
campaign starts until its required foundation gates are green.

## 8. Foundation Workstream

The detailed design is in `runtime_preflight_roadmap.md`. The master ordering is:

### F0: Preserve truth

Deliver:

- (landed) nested doctor behavior is fixed so attaching to a run cannot
  initialize or truncate an existing manifest; `scaffold_setup_run_manifest` is
  append-only and records a distinct attempt marker;
- distinguish score regression from absent/invalid measurement;
- reconcile lane/profile names and all declared runner targets with real
  entrypoints;
- repair GPQA answer ordering and invalidate prior GPQA measurements;
- reject pass@k labels above the actual sampling budget and require an explicit
  distribution-shift declaration before reporting an OOD claim;
- define one complete evaluation identity and one canonical report input.
- adopt the already-certified core run-attempt envelope, clock, process/rank/
  mesh, device, phase, checkpoint, and lineage keys rather than defining an
  incompatible experiment authority.

Gate: focused host tests demonstrate that existing execution history cannot be
silently destroyed or upgraded.

### F1: Typed lifecycle

Implement the experiment-only `torchtitan/experiments/execution/` deep module
and a thin `begin`, `stage`, `finish` interface. Keep `run_common.sh` as a
compatibility facade during migration.

Gate: append-only event tests cover success, block, failure, interruption,
duplicate invocation, artifact receipts, and concurrent readers.

### F2: Profiles and preflight

Separate lane, runtime profile, and execution adapter. Compose a profile from
generic clauses plus benchmark-semantic preflight. Host unit tests are never
gated on rootfs or GPU checks.

Gate: CPU, 1-GPU, 8-GPU, coding-sandbox, Harbor, and tau2 profiles each fail on
the right missing capability and omit irrelevant clauses.

### F3: Rootfs hardening

Pin image and dependency inputs, emit build identity, make base mounts
read-only, declare writable binds, and add explicit network/GPU/Docker/image
capabilities. First prove TorchTitan distributed networking, Monarch/vLLM, and
official harness networking with recorded probes.

Gate: the same declaration cannot be resumed under a different rootfs digest
without an explicit new run or approved equivalence record.

### F4: Runner migration

Reordered ahead of Temporal durability so a real runner exercises the typed
lifecycle before any stage is wrapped in a durable Activity. See ADR
0005-reorder-runner-migration-before-temporal. Migrate one host-testable
reference runner (`run_arithmetic_words_smoke.sh`) end to end, then one CPU
runner, one inference runner, one SFT runner, one external harness runner, then
the remaining inventory. A migrated runner must preserve its task semantics and
canonical report input.

Gate: no runner can claim completion without terminal stage events and artifact
receipts, and no serious runner writes the prototype manifest directly.

### F5: Temporal durability

Reordered after runner migration so durability wraps a lifecycle path already
proven on a real runner. Run a loopback-only, version-pinned local Temporal dev
server with persistent SQLite, plus host workers and resource queues. Activities
launch existing rootfs stages and validate local receipts.

Gate: server restart, worker restart, duplicate delivery, cancellation,
ambiguous subprocess ownership, and verified SFT checkpoint resume all pass
fault injection. RL retries remain disabled.

### F6: Experiment views and operations

The core observability foundation is an upstream prerequisite, not work deferred
to F6. At this wave, index experiment lifecycle and task evidence into that
certified schema, add campaign/operator views, and mirror selected scalar
metrics and links to TensorBoard and optionally W&B. Record inference, data,
SFT, RL, weight-sync, and external-harness phase times separately. Preserve the
core Tier 0 overhead and capture-escalation policy.

Gate: a report can be reconstructed offline from local artifacts alone.

## 9. Training Workstream

The detailed cells and invariants are in `training_research_plan.md`.

### 9.1 SFT sequence

1. Establish CPU dataset and config contracts.
2. Prove Qwen3-0.6B end-to-end plumbing on one GPU.
3. Calibrate Qwen3-1.7B LoRA schedule and memory without making a quality
   claim.
4. Run the minimum conditional replication matrix for modular sequences: two
   fresh split draws by two training seeds for the champion and raw causal
   anchor, with matched base evaluations in every final cell. Treat
   champion-minus-base as the primary policy-compression estimand and
   champion-minus-raw as a secondary mechanism estimand.
5. Run selected base, scaffold-subset, raw, clean, and formatting analyses.
6. Test full-SFT only if LoRA saturates or the representation hypothesis
   requires it.
7. Probe only the selected mechanism on Qwen3-8B after its upstream foundation
   promotion. Reach M7 only with prospectively sized scale-level replication,
   never below three fresh training seeds per arm, and paired final-set
   uncertainty. Treat 30B-A3B as a separately reviewed MoE campaign, never a
   routine next cell; a single affordable MoE cell remains a probe.

All SFT configuration moves to an experiment-owned config registry consumed by
`ConfigManager`; environment variables remain launcher conveniences, not a
second source of training truth.

### 9.2 Policy bridge

Keep three artifacts distinct:

- TorchTitan DCP for distributed recovery;
- PEFT adapter for adapter evaluation and lightweight distribution;
- merged full Hugging Face policy for vLLM loading and RL initialization.

The bridge gate validates state-dict mapping, model config, tokenizer and chat
template, TorchTitan load, vLLM load, and base-plus-adapter versus merged-policy
logit/generation equivalence within a declared tolerance.

### 9.3 Deterministic RL

Before a quality experiment, require strict on-policy FIFO execution,
`target_offpolicy_steps=0`, matched trainer/generator seeds and TP, deterministic
and batch-invariant modes, bfloat16 forward, fp32 master weights, no sequence or
pipeline parallelism, compatible prefix-cache reset, and exact logprob/loss
parity through the focused parity tools.

The causal matrix is:

- base evaluation;
- SFT evaluation;
- base -> RL;
- SFT -> RL.

This distinguishes whether RL rediscovers the SFT gain, adds to it, or erases
it.

### 9.4 Async RL

Change one factor at a time:

1. enable production kernels while remaining on-policy;
2. allow controlled off-policy depth with strict FIFO;
3. introduce a bounded look-ahead window;
4. tune throughput only after policy-age and quality effects are understood.

Report generator latency, rollout wait, trainer forward/backward throughput,
weight-sync wait, end-to-end step time, consumed-policy age, rejection/filter
rates, and active-slot occupancy. No automatic Temporal retry is allowed until
the joint recovery contract covers trainer DCP, policy version, dataset cursor,
prompt ledger, buffer/batcher/inflight dispositions, and sync state.

## 10. Reasoning Workstream

The detailed task protocols are in `reasoning_research_plan.md`.

### R0: Replicate the mechanism lab

- Freeze modular generator, IID band, length-shift and modulus-shift OOD axes,
  prompt, scaffold budget, verifier, and evaluation schedules.
- Execute the minimum conditional matrix of two fresh split draws and two
  training seeds for raw and the selected cleaned/formatting challenger, with
  matched base evaluation on every final cell. Expand upper-level replication
  before final opening if prospective sizing is needed for a recipe-general
  claim.
- Analyze base-elicitable items, pass@k compression, output validity, reasoning
  length, and exact failure types.

Gate: the predeclared champion-minus-base pass@1 gain has a paired fixed-policy
interval and consistent direction across draws without an unacceptable pass@k
or format regression; champion-minus-raw remains secondary mechanism evidence.

### R1: Public exact-answer transfer

- Use GSM8K as the first public numeric distribution.
- Define and freeze `MATH-exact-v1`, selecting only items whose released answer
  can be deterministically normalized and verified.
- Use AIME as a locked harder transfer set, not as a source of tuning feedback.

Gate: replicated gain on GSM8K or MATH-exact with no verifier-validity loss;
AIME is reported regardless of direction.

### R2: Multiple-choice transfer

- Calibrate MMLU-Pro with answer-position and prompt invariance checks.
- Repair GPQA import, rebuild caches, then recalibrate it from scratch.
- Use MMLU-Pro for development and repaired GPQA as a harder post-hoc transfer
  test. Reserve an untouched multiple-choice set for confirmation.

Gate: the result survives choice permutation and label remapping and is not
explained by answer-position frequency.

### R3: Structured abstraction

Treat ARC-AGI-2 as a separate structured-output branch. First establish
scaffold reachability and exact grid serialization. Do not train on it merely
because the current base score is low.

Gate: enough elicitable items exist for a powered compression experiment and
the verifier separates correct grids from malformed output.

### R4: Downstream reasoning/tools

Run a small pinned tau2 suite only after a selected reasoning policy exists.
The official harness owns environment dynamics and scoring. Compare base and
one selected policy with trajectories, final-state metrics, errors, latency,
and cost.

## 11. Coding Workstream

The detailed protocols are in `coding_research_plan.md`.

### C0: Secure execution gate

Before executing untrusted generations, prove isolation, limits, teardown, and
evidence capture with adversarial fixtures. At minimum test filesystem escape,
network access, process spawning, fork/CPU/memory/disk exhaustion, timeouts,
signals, and output flooding. Treat the current subprocess executor as unsafe
until this gate passes.

The recommended executor is a host-launched minimal bwrap boundary per
candidate/task unit, separate from the trusted experiment rootfs. It exposes a
content-identified language/dependency root, one evaluator shim, one task, and
one writable scratch mount; it exposes no network, GPU, Docker, checkout, model
assets, caches, or credentials. Enforce descendant-wide resource accounting
with cgroup v2 plus rlimits, add a pinned runtime-specific seccomp policy, and
block the profile rather than falling back when the host cannot enforce it.

### C1: Function-level transfer

- Use MBPP training identities for verified trace generation and training.
- Keep HumanEval as a locked transfer set.
- Compare base, scaffold, raw, clean, and selected policy under the same secure
  executor, sampling budget, and pass@k estimator.

Gate: a replicated MBPP pass@1 gain transfers directionally to HumanEval with
no increase in unsafe, invalid, or timeout outcomes.

### C2: Hard function synthesis

Use canonical BigCodeBench-Hard as a separate distribution. Preserve official
task construction and tests; do not substitute the earlier contract-chat proxy
for the canonical score. A valid zero or regression is a result.

Gate: M2 scaffold reachability before any training allocation.

### C3: Temporal hidden-test coding

Integrate the official LiveCodeBench evaluation path with release-date and
contamination boundaries. Current public-case execution is a local smoke, not
an official LiveCodeBench result.

Gate: official harness provenance, hidden/private test semantics as provided by
the benchmark, temporal slice identity, and reproducible score ingestion.

### C4: Scientific coding

Integrate SciCode only after the executor and official-harness seam are stable.
Its numerical and domain-specific tolerances must come from the benchmark, not
a generic exact-output verifier.

### C5: Agentic coding

Run Terminal-Bench through pinned Harbor. TorchTitan owns model artifacts,
rootfs launch, identity, and ingestion; Harbor owns task execution and scoring.
Compare base with one selected coding policy on a small fixed suite before any
broad run.

## 12. Campaign Declaration And Governance

Every scientific campaign is reviewed before launch and freezes:

- claim and falsifiable hypothesis;
- benchmark revision, split identities, exclusions, and contamination notes;
- model, tokenizer, prompt, scaffold, verifier, and environment identity;
- exact experimental cells and the reason each cell exists;
- primary endpoint and estimand;
- guardrail metrics and qualitative evidence classes;
- draw/seed/randomization plan;
- maximum compute, wall-clock, and retry budget;
- stopping rule and prohibited peeking;
- promotion threshold and downstream action for promote, hold, or reject; and
- named human authorization for the next compute tier.

Machine gates may prevent invalid work and summarize evidence. They do not
autonomously authorize a new scientific claim, a larger model, or an external
benchmark expenditure.

### 12.1 Statistical plan

The problem is the primary unit, not individual rollout samples.

- Use paired comparisons on identical evaluation problems and sampling
  schedules.
- Bootstrap problems, retaining all rollout siblings within the resampled
  problem.
- Before freezing a generic absolute-effect gate such as `+0.05`, run a
  prospective power and precision simulation using blinded or development-only
  estimates of paired discordance, benchmark size, and between-run variance.
  Freeze the number of problems, split draws, and training seeds that can
  resolve the declared effect. If the available design cannot do so, its
  outcome is `hold/inconclusive`, not evidence of no effect.
- The minimum `2 draws x 2 seeds` M4 matrix estimates each fixed policy on
  paired problems and demonstrates cellwise reproducibility. With only four
  trained policy realizations, any pooled interval is conditional on those
  realizations; a hierarchical bootstrap is descriptive and must not be called
  population-level uncertainty over data draws or training seeds.
- A recipe-general claim requires prospectively sized upper-level replication
  based on the simulation, never fewer than four independent data draws by
  three training seeds per arm. If that floor is still underpowered, add the
  simulated number of cells or narrow the claim. Use a prespecified
  small-sample run-level method and report sensitivity to it.
- Report absolute deltas, intervals, and the distribution of draw/seed effects;
  do not rely on pooled rollout counts.
- Freeze locked evaluation sets and analyze them once at the declared campaign
  boundary.
- The primary policy-compression estimand is champion minus frozen base. The
  champion-minus-raw comparison is a secondary mechanism estimand when those
  policies differ. Evaluate base, champion, and raw on every final cell under
  the same problem and sampling keys.
- A failed final gate spends that holdout. Any correction uses development
  evidence only; retesting requires a new campaign declaration and a newly
  frozen, untouched final registry.
- Report pass@1 and the scaffold budget endpoint together so compression is not
  purchased by destroying residual search coverage.
- Treat multiple benchmark endpoints as separate claims; either predeclare one
  primary endpoint or adjust the family-wise interpretation.

### 12.2 Common guardrails

- strict-format success;
- verifier and harness validity;
- pass@k curve and base-elicitable subset;
- output and reasoning length;
- timeout, crash, empty, and overlength rates;
- unsafe execution events for coding;
- latency, tokens, and wall-clock cost;
- representative wins, regressions, unchanged successes, unchanged failures,
  and formatting-only changes.

### 12.3 Stop rules

Stop or redesign when:

- split overlap, identity drift, verifier inconsistency, or artifact ambiguity
  invalidates measurement;
- M2 shows too few elicitable problems for a powered training study;
- two clean replications disagree in direction without an identified moderator;
- a formatting gain explains the endpoint but reasoning/task success does not
  improve;
- a challenger misses its futility boundary;
- safety isolation fails;
- deterministic RL parity fails;
- policy age exceeds the declared async bound;
- an official harness cannot be pinned or its score cannot be ingested without
  semantic alteration; or
- the campaign consumes its declared budget.

A stopped branch remains documented. It does not force substitution of an
easier metric after results are known.

## 13. Initial Campaign Slate

These are declarations to prepare, not blanket authorization to execute every
cell.

| Campaign | Question | Minimum cells | Required maturity before launch | Advance condition |
| --- | --- | --- | --- | --- |
| `FOUNDATION-TRUTH` | Can the evidence lifecycle preserve truth under nesting, retries, and regressions? | unit plus host fault matrix | none | F0/F1 gates pass. |
| `ROOTFS-REPRO` | Can identical stages prove environment identity and explicit capabilities? | CPU, 1-GPU, distributed, Monarch/vLLM, external harness probes | F1 | F2/F3 gates pass. |
| `TEMPORAL-DURABILITY` | Can staged work recover without duplicate or ambiguous publication? | CPU stage, eval shard, SFT checkpoint resume; RL forced no-retry | F1-F4 | F5 fault matrix passes. |
| `TRAIN-MOD-REP` | Does modular raw/selected SFT gain replicate? | Minimum conditional matrix: 2 fresh draws x 2 seeds x champion/anchor plus matched base; expand to the prospectively sized matrix for a recipe-general claim | M2 modular, F0-F5 | M4 gate in reasoning/training plans with claim scope recorded. |
| `REASON-EXACT-TRANSFER` | Does the selected modular mechanism transfer to public exact reasoning? | base/scaffold/selected/anchor on GSM8K and MATH-exact; AIME locked transfer | modular M4 | public transfer primary endpoint passes. |
| `REASON-MC-TRANSFER` | Does it transfer under choice permutation? | MMLU-Pro development, repaired GPQA post-hoc transfer, untouched confirmation set | both task contracts valid | permutation-safe direction plus untouched confirmation. |
| `CODE-FUNCTION-TRANSFER` | Can verified MBPP traces improve policy and HumanEval transfer? | base/scaffold/selected/anchor MBPP; locked HumanEval | secure executor, both M1/M2 | replicated MBPP gain plus directional transfer. |
| `RL-MOD-DETERMINISTIC` | Does RL add to base and SFT without off-policy ambiguity? | base, SFT, base->RL, SFT->RL | modular M4, bridge, parity | M5 causal comparison valid. |
| `RL-MOD-ASYNC` | Which async factor buys throughput at acceptable quality/policy age? | deterministic control plus one-factor kernel/off-policy/window cells | deterministic RL M5, recovery | declared quality/throughput frontier. |
| `REASON-STRUCTURED` | Is ARC scaffold behavior reachable and compressible? | base/scaffold first; training only after M2 | ARC M0/M1 | sufficient elicitable set. |
| `CODE-HARD` | Is canonical BigCodeBench-Hard scaffold behavior reachable? | base/scaffold first | secure executor, M0 | sufficient elicitable set. |
| `DOWNSTREAM-AGENTIC` | Does one selected policy improve official environment-mediated work? | base versus selected on pinned tau2 and/or Terminal-Bench subsets | relevant upstream M4+, official harness X1 | X2 fixed-suite pilot or X3 replicated transfer, with claim scope and valid trajectories. |
| `SCALE-CONFIRM` | Does the selected mechanism persist at larger model scale? | winning control/treatment cells across a prospectively sized seed matrix, minimum three fresh training seeds per arm | prior branch M4/M5 | paired final-set uncertainty and guardrails pass M7; one run is reported only as a scale probe. |

## 14. Temporal Campaign Shape

Use Temporal for long-lived operational sequencing, not scientific decision
making.

```text
ResearchCampaignWorkflow
  -> freeze and validate declarations
  -> ExperimentRunWorkflow(draw 0, seed 0)
  -> ExperimentRunWorkflow(draw 0, seed 1)
  -> ExperimentRunWorkflow(draw 1, seed 0)
  -> ExperimentRunWorkflow(draw 1, seed 1)
  -> validate canonical reports
  -> wait at human promotion gate
  -> optionally start the next campaign phase
```

Within one run, Activities align with durable artifact seams: acquire/validate
assets, materialize splits, generate rollout shards, build verified dataset,
train SFT, validate DCP, export policy, evaluate a condition, and finalize the
report. Never create Activities per prompt, batch, optimizer step, or rollout.

Task queues describe capabilities such as `cpu`, `gpu1`, `gpu8`,
`coding-sandbox`, `harbor`, and `tau2`. A local lease still owns actual GPU or
exclusive resource admission. Use Continue-As-New only at campaign phase
boundaries where event history would otherwise grow without bound.

## 15. Risk Register

| Risk | Early signal | Required response |
| --- | --- | --- |
| Evidence schema forks | Two files can independently claim final status. | Stop migration; make execution evidence enrich the canonical report input. |
| Mutable environment | Same declaration observes a different rootfs/package/model identity. | New run or explicit equivalence review; never silently resume. |
| Duplicate activity publication | Existing artifact has missing or mismatched receipt/hash. | Fail safely; do not overwrite or assume success. |
| Ambiguous live process after retry | No durable supervisor/official job API can prove ownership and terminal status. | Retain the lease, mark blocked, and require operator reconciliation; never rely on PID alone. |
| GPQA answer-position leakage | Correct labels are imbalanced or performance changes under permutation. | Invalidate and rebuild the evaluation. |
| Coding sandbox escape | Adversarial fixture reaches host/network/resource outside policy. | Halt all generated-code execution. |
| SFT resume drift | Post-resume loss/state diverges from uninterrupted matched run. | Reject recovery path and inspect full DCP state. |
| RL numerical mismatch | Trainer/generator logprobs differ in parity configuration. | Stop RL quality work; diagnose before optimization. |
| Async staleness | Consumed policy age or queue occupancy exceeds declaration. | Reject cell or reduce capacity/window/off-policy depth. |
| Benchmark overfitting | Locked set informs prompt, cleaning, or stop decisions. | Invalidate locked claim and create a fresh held-out evaluation. |
| Sparse hard task | Scaffold has almost no successes. | Record hard negative; improve scaffold or stop branch before training. |
| Compute fragmentation | Too many underpowered challenger cells. | Return to champion/anchor rule and terminate low-value cells. |

## 16. State Tracker

Update this table only from validated reports; plans do not self-promote.

| Program area | Current state | Next gate |
| --- | --- | --- |
| Execution foundation | Design approved; current runners expose manifest and doctor correctness gaps. | F0 truth-preserving fixes. |
| Rootfs | Real bwrap/GPU path exists; F3a landed content-addressed identity, fail-closed dest resolver, and ownership-gated replacement; build inputs and mounts remain mutable/broad. | F3b read-only mounts, capability policy, and immutable build inputs. |
| Temporal | Primary-source design complete; no adapter implementation implied. | F5 prototype plus fault injection after F0-F4. |
| Countdown reasoning | Historical replicated anchor; principal runs are not a fully crossed new-ladder M4. | Lifecycle migration and stop-quality ablation; fill cross cells only if a new Countdown M4 claim is needed. |
| Modular reasoning | M2 plus a one-draw exploratory SFT lead; historical `ood_test` is held-out IID. | `TRAIN-MOD-REP` with genuine OOD axes to reach or reject M4. |
| GSM8K/MATH/AIME | M0/M1-style calibration and plumbing vary by task; no trained transfer claim. | Freeze exact task contracts and establish M2. |
| MMLU-Pro | Base calibration exists; no trained transfer claim. | Permutation-safe M1/M2 declaration. |
| GPQA | Prior execution exists but measurement is invalid due correct-first import. | Repair M0 and rebuild M1. |
| ARC-AGI-2 | Valid hard calibration/plumbing; reachability remains weak. | Establish M2 or stop before training. |
| Function coding | MBPP/HumanEval plumbing exists; current executor is not a security boundary. | Secure executor, M0/M1, then M2. |
| BigCodeBench-Hard | Hard-negative calibration; proxy and canonical conditions must remain distinct. | Canonical secure M1/M2. |
| LiveCodeBench | Public-case local smoke only. | Official temporal harness contract. |
| SciCode | Not integrated. | M0 after official coding seam is stable. |
| Deterministic RL | Architecture exists elsewhere in the repo; no task-specific parity campaign. | Policy bridge plus deterministic M5 gate. |
| Async RL | Not scientifically authorized for this program. | M5 and joint recovery before M6. |
| Harbor/Terminal-Bench | Compatibility probes only. | Official pinned M1, then selected-policy comparison. |
| tau2 | Noop/mock/model-policy probes only. | Official pinned M1, then selected-policy comparison. |

## 17. Ordered Near-Term Backlog

Execute in this order unless a new review changes the dependency graph:

1. (nested manifest truncation landed: `scaffold_setup_run_manifest` is
   append-only with a regression test.) Add regression tests that reproduce
   score-regression misclassification and GPQA correct-first behavior.
2. Fix those truth-preservation failures and mark prior GPQA report inputs
   invalid without deleting historical reports.
3. Freeze the canonical run declaration, attempt bundle, stage-event, artifact
   receipt, and enriched report-input schemas.
4. Implement the experiment-only lifecycle module and compatibility facade.
5. Split the current broad doctor into composable clauses and add semantic
   preflights for reasoning, coding, Harbor, and tau2.
6. Pin and identify rootfs construction; prove required networking and device
   paths; then enforce explicit capabilities and read-only defaults.
7. Migrate one host-testable reference runner
   (`run_arithmetic_words_smoke.sh`) end to end onto the lifecycle, then
   representative CPU, inference, SFT, and external harness runners, then finish
   the runner inventory. See ADR 0005-reorder-runner-migration-before-temporal.
8. Implement local resource leases and the Temporal CPU/reference workflow.
9. Fault-test duplicate delivery, cancellation, worker/server restart, receipt
   validation, and ambiguous subprocess ownership.
10. Add verified SFT DCP resume to the Temporal path; keep RL retry disabled.
11. Freeze modular replication declarations and execute `TRAIN-MOD-REP`.
12. Advance only if the M4 gate passes; otherwise diagnose draw/seed,
    formatting, and scaffold-reachability mechanisms before adding tasks.
13. Establish secure coding execution and public reasoning/coding M0-M2 in
    parallel where resources do not conflict.
14. Build and validate the full-policy bridge, then run deterministic modular
    RL parity and causal cells.
15. Run one-factor async RL only after deterministic evidence and joint recovery
    are complete.
16. Promote a single selected policy to the official downstream agentic pilot.
17. Spend larger-model compute only on a predeclared scale confirmation.

## 18. Program Completion Criteria

The long-horizon program is complete when it can answer, including with a
negative answer, all of the following from immutable evidence:

- Which verified scaffold behaviors compressed into pass@1 policy behavior?
- Which data treatment and training mechanism caused the gain?
- Did it replicate across draws and seeds?
- How much of the scaffold pass@k frontier remained after compression?
- Did the mechanism transfer to public reasoning and executable coding?
- Did deterministic RL add to or replace SFT gains?
- What quality, freshness, and throughput tradeoff did async RL introduce?
- Did any selected policy improve an official downstream environment-mediated
  suite?
- Were all claims reproducible from pinned rootfs, declarations, checkpoints,
  artifact receipts, and canonical reports?

Completion does not require positive results on every branch. It requires that
each attempted branch end in a valid promotion, hold, reject, or documented
stop decision, with infrastructure failures kept distinct from scientific
outcomes.
