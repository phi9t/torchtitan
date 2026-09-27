# Qwen3 Formal Verification Vehicle

Status: in-progress -- Phase 1; tickets 01 and 02 resolved, 03 in review
Route: gated, two-phase tracer-bullet program
Authority: implementation, commit, push, pull-request creation, and merge each
require their normal separate authorization; a human performs every merge

Ticket status vocabulary, in lifecycle order:

- `ready-for-agent`: open, specified, and available to claim.
- `needs-info`: open but blocked on a decision or an external answer.
- `claimed`: an agent is actively implementing it.
- `review-pending`: implementation and the executable gate are complete;
  independent Standards and Spec review has not yet returned PASS.
- `resolved`: reviewed, closed, and evidence recorded.

## Problem Statement

The existing branch demonstrates useful pieces of an accelerator trace schema, a
semantic tracer, a logical emulator, a Qwen3 DPxTP trace builder, and generated
TLA+ and Lean text. It does not yet demonstrate that those pieces form a real
formal-verification path. The TLA+ artifact has not been accepted by TLC, the
Lean artifact has not been compiled by Lean, and the current trace is synthetic
rather than evidence from an executed TorchTitan Qwen3 training step.

TorchTitan also has no repository-level Bazel foundation for declaring and
caching the large formal toolchains. Running ad hoc host tools would violate the
repository's Insula boundary, make results environment-dependent, and risk
placing large Lean artifacts in the checkout.

The work therefore needs a deliberately narrow end-to-end scout before the
formal model or runtime integration grows. Once that tracer bullet proves the
complete path, a second phase can deepen the semantics, runtime capture, proofs,
and operational evidence without guessing which interfaces work in practice.

## Solution

Develop the vehicle in two phases.

### Phase 1: end-to-end scout

Build the path as two sequential scout steps. Scout A establishes the smallest
honest end-to-end path; Scout B widens the same path to the distributed topology
that the existing experiment intends to formalize.

Scout A:

1. Enter through an Insula-aware repository command.
2. Use a minimal repository-level Bazel module and formal configuration.
3. Run exactly one optimizer step through the core `Trainer` with the Qwen3
   debug model on one CUDA device: one rank, DP=1, TP=1, bfloat16, AdamW, local
   batch 1, sequence length 128, seed 42, and deterministic mode.
4. Observe that execution through experiment-owned PyTorch contexts and hooks,
   preserving both raw observed evidence and a deterministic normalized
   `qwen3.formal.scout.v0` trace without changing core-training defaults.
5. Generate deterministic TLA+ and Lean artifacts from that trace.
6. Run TLC on a bounded valid model and on a negative model that must violate a
   named invariant.
7. Compile and kernel-check the generated Lean facts against separately
   maintained step-lifecycle definitions and proofs.
8. Run Python contract and artifact-synchronization tests with pytest, then run
   both formal checks with Bazel. The Insula wrapper orchestrates both without
   making Bazel the owner of TorchTitan's Python suite.
9. Submit Scout A to a fresh clean-context Codex reviewer for separate
   Standards and Spec verdicts.

Scout B, blocked by Scout A:

1. Run the same one-step core `Trainer` path on four CUDA ranks with a 2x2
   DPxTP mesh: data-parallel replicate degree 1, data-parallel shard degree 2,
   tensor-parallel degree 2, and every other mesh degree 1. Preserve the same
   deterministic model, local batch 1, sequence length 128, precision, seed,
   and optimizer contract. This yields global batch 2, 256 tokens per optimizer
   step, and one gradient-accumulation step.
2. Extend raw observation and normalized facts to real process groups,
   rank/mesh coordinates, tensor placements, collective work lifecycle,
   stream/executor assignment, producer correlation, and step completion.
3. Check separately maintained TLA+ and Lean DPxTP invariants against the
   observed trace, including one named negative mutation per checker.
4. Repeat synchronization, cache, focused/broader tests, lint, and a fresh
   clean-context Codex review gate from the final Scout B state.

The scout phase proves plumbing, single-rank step-lifecycle capture, and one
bounded 2x2 DPxTP execution only. It does not claim a complete model of
TorchTitan training, arbitrary-topology correctness, convergence, performance,
or broad correctness.

### Phase 2: progressive design and refinement

Use the working scout to refine one contract at a time:

- use the two scout traces to define the stable trace vocabulary and
  compatibility/versioning rules;
- generalize beyond the one observed 1x1 and 2x2 profiles without silently
  treating fixture facts as runtime evidence;
- deepen the TLA+ transition system and its safety/liveness invariants;
- deepen Lean definitions and proofs over the same normalized trace facts;
- establish trace-refinement checks between observed execution and the abstract
  models;
- add controlled corruptions and fault cases that prove each checker rejects
  each named mutation actually exercised;
- measure capture overhead and artifact size before proposing any core Trainer
  adoption;
- promote only interfaces that satisfy the repository's observability evidence
  contract and preserve existing training behavior.

Each refinement must remain executable end to end. Scout B cannot begin until
Scout A passes its full gate, and Phase 2 cannot begin until Scout B passes its
full gate.

## User Stories

1. As a TorchTitan researcher, I want one command to exercise the complete
   Qwen3-to-formal-checker path, so that I can tell whether the architecture is
   viable before investing in deeper models.
2. As a TorchTitan developer, I want the scout to execute a real Qwen3 forward,
   backward, and optimizer step, so that synthetic fixtures are not mistaken for
   runtime integration evidence.
3. As a formal-methods developer, I want each normalized Python trace to be the
   single source for generated facts, so that TLA+ and Lean do not drift while
   the scout schema is still provisional.
4. As a TLA+ reviewer, I want TLC to accept the valid bounded model, so that
   generated syntax alone is not reported as formal verification.
5. As a TLA+ reviewer, I want a deliberately invalid model to violate a named
   invariant, so that the TLC gate proves it rejects that exact mutation.
6. As a Lean reviewer, I want the generated module compiled and kernel-checked,
   so that string snapshots are not reported as Lean evidence.
7. As a Lean reviewer, I want mesh and lifecycle propositions to contain real
   definitions and proofs, so that placeholders such as a proposition equal to
   `True` cannot pass the gate.
8. As a build maintainer, I want Bazel to own the formal toolchain dependencies,
   so that downloads, extraction, and invalidation are reproducible.
9. As a developer using Insula, I want formal commands to re-enter or fail
   closed on the rootfs boundary, so that host state cannot silently affect the
   result.
10. As a developer with limited disk space, I want Lean and Bazel artifacts kept
    in a reusable cache outside the checkout, so that large generated files do
    not pollute Git or get copied repeatedly.
11. As a contributor, I want the repository to be Bazel-enabled without
    replacing pytest, torchrun, or existing launchers, so that the formal
    experiment remains additive.
12. As a contributor, I want ordinary core imports and tests to remain
    independent of formal dependencies, so that optional tooling cannot break
    core training.
13. As a reviewer, I want every formal claim mapped to an executed checker and
    artifact, so that a skipped or unavailable tool is never presented as PASS.
14. As a reviewer, I want tool versions, integrity hashes, and source references
    recorded, so that evidence can be reproduced later.
15. As a researcher, I want deterministic artifact generation, so that a trace
    or exporter change produces an inspectable diff.
16. As a researcher, I want raw-to-normalized provenance plus trace identities
    and causal edges preserved across Python, TLA+, and Lean, so that findings
    can be joined back to runtime events.
17. As a distributed-training developer, I want DP and TP coordinates and group
    membership validated explicitly, so that a plausible-looking but malformed
    mesh is rejected.
18. As a distributed-training developer, I want collective work lifecycle and
    producer correlation validated, so that missing or reordered work cannot be
    hidden by a final-state-only check.
19. As an experiment owner, I want the scout's limits stated prominently, so
    that plumbing evidence is not promoted into a distributed-correctness
    claim.
20. As an experiment owner, I want Phase 2 changes delivered as independently
    reviewable refinements, so that the system can evolve without a second
    speculative rewrite.
21. As a core maintainer, I want capture overhead and artifact volume measured
    before Trainer integration, so that formal observability does not violate
    the Tier 0 performance budget.
22. As a core maintainer, I want experiment-to-core dependency direction
    preserved, so that the core never imports optional formal tooling.
23. As a reviewer, I want Standards and Spec review for every refinement, so
    that correct-looking formal output does not mask repository violations or
    scope creep.
24. As the branch owner, I want fresh clean-context Codex review, so that an
    independent reviewer gates Scout A, Scout B, and each later refinement.

## Implementation Decisions

- The Phase 1 public test seam is one Insula-aware repository command that
  drives the complete Bazel formal suite. Direct Bazel targets remain available
  for focused diagnosis, but the wrapper is the supported user entrypoint.
- Scout A runs exactly one deterministic optimizer step through the core
  `Trainer`, not an experiment-owned reconstruction of trainer orchestration.
  Its fixed profile is one CUDA device and rank, DP=1, TP=1, bfloat16, AdamW,
  local batch 1, global batch 1, sequence length 128, 128 tokens per optimizer
  step, one gradient-accumulation step, seed 42, and deterministic mode.
- Scout B runs the same one-step `Trainer` path on four CUDA ranks with
  data-parallel replicate degree 1, data-parallel shard degree 2, and
  tensor-parallel degree 2. It retains local batch 1 and sequence length 128,
  giving global batch 2, 256 tokens per optimizer step, and one
  gradient-accumulation step. If runtime validation disproves a required
  divisibility assumption, Scout B returns to design review rather than
  silently changing this comparison profile.
- Both scouts select `qwen3_debugmodel` through `ConfigManager` and apply their
  overrides through the normal CLI precedence. They use the repository-local
  `c4_test` data and test tokenizer, record the exact first-batch identity,
  start from seed-42 model initialization with checkpoint loading and saving
  disabled, and create no experiment-owned configuration path. Scout A
  explicitly sets every mesh degree to 1. Scout B explicitly sets data-parallel
  replicate degree 1, data-parallel shard degree 2, tensor-parallel degree 2,
  and every other mesh degree 1; `DP=2` never leaves the replicate-versus-shard
  choice to inference.
- The runtime adapter stays experiment-owned during Phase 1 and observes the
  unchanged `Trainer` through PyTorch contexts and hooks. Core never imports the
  tracer or formal packages. If required evidence cannot be observed without a
  new core interface, the active scout returns to design review instead of
  adding a hidden experiment-specific hook to core.
- Synthetic traces remain useful as unit fixtures and negative controls. They
  are explicitly secondary to the observed scout trace.
- Phase 1 emits a provisional `qwen3.formal.scout.v0` normalized trace with
  stable run/attempt, process/rank/mesh, device, clock/order, step, phase,
  lineage, event, work, and causal keys. It is not promoted as the repository's
  canonical observability schema.
- Raw observed evidence is immutable. Normalization records a provenance map
  from every normalized event back to its raw source.
- The sealed source identity covers executable and checked-in source only.
  Process and tracker documents under `.scratch/` and `.superpowers/` have their
  **bytes** recorded informationally at seal time and deliberately excluded from
  `source_id`, because a sealed attempt reports its own evidence identity into
  its ticket and hashing that ticket into the identity it reports is circular.
  A ticket status transition or a recorded gate-evidence section therefore
  cannot invalidate the bundle that justifies it. Path-set **membership** is a
  separate matter and is still covered: adding or removing any path, process
  paths included, changes `status_sha256` and therefore `source_id`, because the
  attempt's path set is part of its identity. The captured path set must not
  change while a gate runs, lint covers both partitions, and a process path may
  not carry executable or formal source.
- Scout B normalization preserves each rank's observed source order and creates
  cross-rank order only from explicit synchronization or causal evidence. A
  four-rank bundle is incomplete, and cannot reach either formal checker, when
  any rank is missing or duplicated or when run, attempt, topology, model, or
  configuration identity disagrees across ranks. Input identity is validated
  at the correct mesh scope: TP peers with the same DP coordinate must report
  the same local first-batch identity, while the attempt records an ordered
  global input-bundle identity keyed by DP coordinate. Different DP coordinates
  are not required to consume the same local batch.
- TLA+ and Lean exporters consume the same normalized trace and generate facts
  only. Separately maintained TLA+ semantics and Lean definitions state the
  invariants, so a generator cannot emit both the input and the proposition
  that approves it.
- Scout A's TLA+ model is a finite bounded single-rank step transition system;
  Scout B adds the bounded 2x2 mesh and collective lifecycle. Both have an
  initial state, next-state relation, named invariants, and a
  stuttering-complete specification accepted by TLC.
- The negative TLC model changes one controlled fact or action and must produce
  the expected named invariant violation. A checker error is not an acceptable
  negative result.
- Each Lean scout defines the same finite facts as its TLA+ counterpart and
  checks them against separately maintained lifecycle and mesh definitions. The
  primary theorem may not depend on `sorryAx` or project-declared axioms; the
  check records its printed axiom dependencies. Definitions that reduce the
  main contract to unconditional truth are forbidden.
- Each Lean negative control changes one specified fact and must fail to
  establish its named proposition. This proves rejection of that mutation only,
  not a general failure class.
- Bazel is introduced as an additive repository-level build surface for formal
  work. Existing Python packaging, pytest, torchrun, and shell launchers remain
  authoritative for their current responsibilities.
- Bazel pins its own expected version, the official TLA+ tools artifact, a
  modern JDK, and the official Lean distribution. Network resolution may occur
  only while Bazel materializes declared external repositories.
- The TLA+ setup follows Ferric Continuum's hermetic pattern: integrity-pinned
  `tla2tools`, Bazel's JDK 17, and an explicit formal configuration with good
  and bad model checks.
- The Lean version begins at `leanprover/lean4:v4.34.0`, matching the reviewed
  TorchLean toolchain pin. TorchLean informs version pinning, external build
  placement, profile isolation, and cache discipline; it is not treated as a
  Bazel reference because TorchLean uses Lake and elan.
- The Lean distribution is a Bazel external dependency. No archive, extracted
  toolchain, package cache, object tree, or generated proof cache is committed
  or placed under the source tree.
- Bazel's output user root and repository cache live in a writable,
  Insula-visible cache outside the checkout, keyed by repository identity and
  tool versions. Lean archives, extracted toolchains, build products, TLC state
  spaces, and checker caches all remain outside the checkout. After one
  networked materialization, an explicit no-fetch second run proves reuse.
  Cleanup is explicit and never part of normal tests.
- Formal targets are opt-in and tagged so ordinary TorchTitan unit tests do not
  download large toolchains. The supported formal wrapper requests them
  explicitly and never converts an unavailable dependency into PASS.
- Scout A and Scout B each end only when their observed trace, deterministic
  exports, TLC positive/negative checks, Lean compilation/proofs, Python tests,
  lint, no-fetch cache rerun, and fresh clean-context Codex review pass from the
  same final worktree state.
- Phase 2 begins with a human-approved design checkpoint over both immutable
  scout attempt bundles. It defines the stable schema before compatibility
  promises are made. Every deeper invariant states which runtime events support
  it, which corruption violates it, and which checker establishes that specific
  result.
- Core Trainer instrumentation is a promotion decision, not an automatic
  consequence of the scout. It requires compatibility with the versioned
  run-attempt evidence contract, measured overhead, and an independently
  reviewed seam.
- Every per-ticket review uses a fresh clean-context Codex reviewer over the
  current uncommitted delta. The reviewer must return separate Standards and
  Spec verdicts, and every blocking finding must be fixed and re-reviewed before
  the ticket can close. The final assembled branch receives a separate
  clean-context Codex defect review. Missing review output or unavailable
  reviewers block the gate; self-review by the implementer is not a substitute.

## Testing Decisions

- Tests observe public behavior: the supported Insula command, core `Trainer`
  execution, declared Bazel targets, raw-to-normalized provenance, emitted trace
  contract, checker classification, and deterministic generated artifacts.
- Scout A starts red at the highest seam: the supported end-to-end command or
  formal Bazel target does not exist. It turns green only when the real 1x1
  Qwen3 step reaches both formal checkers. Scout B then starts red on the absent
  observed 2x2 path and turns green only on a four-rank execution.
- The real-step tests assert observed forward, backward, gradient readiness,
  optimizer mutation, and step completion. Scout B additionally asserts
  observed rank/mesh, process-group, placement, collective work lifecycle,
  executor, and producer-correlation evidence. Tests do not infer those events
  solely from configured intent.
- Scout B merge tests reject a missing rank, duplicate rank, inconsistent
  run/attempt or topology identity, inconsistent local input identity among TP
  peers at the same DP coordinate, and unsupported cross-rank total ordering;
  no partial bundle may be exported to TLC or Lean.
- Exporter tests compare deterministic complete artifacts and semantic fields,
  not fragments that merely restate implementation strings.
- TLC tests distinguish success, named invariant violation, parse/semantic
  error, missing dependency, and infrastructure failure.
- Lean tests distinguish successful kernel checking, compiler failure, missing
  toolchain, `sorryAx` or project-axiom dependence, and one deliberately false
  named proposition per scout.
- Negative trace tests mutate one dimension at a time. Scout A covers phase and
  causal order, gradient readiness, and optimizer completion. Scout B adds mesh
  membership, tensor placement, producer correlation, and collective lifecycle.
- Cache and wrapper tests verify that formal dependencies and outputs stay
  outside the checkout and that an explicitly offline/no-fetch second run can
  reuse Bazel-managed state.
- Existing focused Python tests remain the nearest prior art for trace and
  exporter contracts. Ferric's good/bad TLC Bazel test is the prior art for
  checker classification. TorchLean's isolated build-root strategy is the
  prior art for keeping large Lean artifacts outside the source tree.
- Phase 2 adds a regression test before each semantic refinement and preserves
  both real Phase 1 end-to-end scout gates throughout. From every refinement's
  final state, the supported Scout A and Scout B commands must re-execute the
  real 1x1 and 2x2 Trainer paths and both formal backends; checked-in or
  replayed fixtures alone do not satisfy this regression gate. Every refinement
  receives its own fresh clean-context Codex review gate.
- GPU or distributed claims require the repository's integration runner and
  matched topology evidence. CPU, fake-backend, or local-tensor runs are labeled
  as plumbing or simulation evidence only.
- Each ticket receives separate Standards and Spec review from a fresh Codex
  context. Each phase also receives an independent Codex defect review,
  followed by fresh final-state verification.
- Every required command records its command line, final-state commit, exit
  status, and produced evidence path. A skipped check, an empty or stale
  artifact, a missing dependency, a checker crash, an infrastructure failure,
  or a wrapper that catches failure cannot satisfy an acceptance criterion.
  Expected negative tests pass only on the specified named invariant or
  proposition rejection.

## Out of Scope

- Converting all TorchTitan Python packages or tests to Bazel.
- Replacing `uv`, pytest, torchrun, or existing experiment runners.
- Treating the Phase 1 1x1 and 2x2 observations as proof of arbitrary-topology
  or complete distributed-training correctness.
- Claiming Lean theorem coverage beyond the propositions actually compiled and
  kernel-checked.
- Claiming TLA+ liveness or safety properties that are not named in the checked
  configuration.
- Full Qwen3 0.6B/1.7B/8B/30B training, convergence, checkpoint, or performance
  certification.
- Online RL, multimodal, checkpoint-lineage, or automatic recovery modeling.
- Adding formal-tool dependencies to TorchTitan core runtime imports.
- Core Trainer adoption before the Phase 2 promotion gate and Tier 0 overhead
  evidence.
- Committing downloaded toolchains, Bazel caches, generated state spaces, Lean
  build trees, or large run artifacts.
- Commit, push, pull-request creation, or merge without separate authority.

## Further Notes

- Existing branch base: `843948e9f9df21984f026b1abf8f7da8617c0851`;
  existing implementation commit: `366737c37478d2253491742a74179febcf4ea54f`.
- Scout audit finding: the current generated TLA+ redeclares names as both
  constants and definitions, lacks a TLC-executable specification/configuration,
  and therefore has not established TLC acceptance.
- Scout audit finding: the current Lean artifact is not compiled, and its main
  mesh well-formedness definition is a placeholder truth rather than a useful
  proposition.
- Ferric reference: the `formal-tla-hermetic` worktree at commit `8c44b3d`
  demonstrates pinned TLA+ 1.8.0, Bazel JDK 17, and good/bad TLC targets.
- TorchLean reference: upstream commit
  `de3192df4779877ceb65cfe470a4d4ce9d480a5b` pins Lean 4.34.0 and keeps Lake
  build products outside the checkout. TorchLean currently does not provide a
  Bazel integration, so only its Lean and cache policies are adopted.
- The initial missing-target probe ran inside Insula and failed because Bazel
  was not installed. This is valid red evidence for the scout, not a product
  failure or formal result.
- The current worktree's automatic rootfs build encountered a pre-existing mise
  asset download failure. A known-good shared Insula rootfs exists; the scout
  must record which rootfs identity it uses and must not hide provisioning
  failures.
- The original plan required direct Trae CLI review. On 2026-09-24 the user
  explicitly replaced that gate with Codex review after the current stable Trae
  client repeatedly emitted a terminal review result but failed to exit. The
  retained Trae diagnosis is historical evidence rather than an active gate.
- Checked-in generated formal modules are compact review fixtures. Raw observed
  traces, TLC state spaces, Lean build trees, and checker logs are ignored
  evidence artifacts referenced through their run-attempt provenance.
