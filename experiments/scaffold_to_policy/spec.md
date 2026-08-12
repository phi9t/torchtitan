---
title: Scaffold-To-Policy Experiment Registry And Reasoning Expansion
labels:
  - ready-for-agent
status: ready
---

# Scaffold-To-Policy Experiment Registry And Reasoning Expansion

## Problem Statement

The Countdown pilot produced a real scaffold-to-policy signal: verified
best-of-32 Countdown behavior from Qwen3-1.7B can be compressed into better
first-sample behavior with TorchTitan LoRA SFT. The clean-split full evaluation
shows the strongest arm improving held-out OOD pass@1 and pass@32, and every
full arm improves both metrics.

The current execution state is not yet broad or reproducible enough to support
larger reasoning or agentic benchmark claims. The run artifacts are partly
manual, the manifest uses a mutable `current` file, provenance does not yet
distinguish fresh work from reused checkpoints in a machine-readable way, and
strict output-format compliance is not separated from arithmetic-trace success.
The next program needs to preserve the rootfs discipline and exact-verifier
strengths of Countdown while making runs auditable, repeatable, and portable to
new reasoning tasks before adding external agentic harnesses.

## Solution

Build a reusable scaffold-to-policy experiment registry and reporting contract,
then use it to run a staged reasoning-first expansion.

The solution keeps Countdown as the reference task and turns its current
contracts into reusable infrastructure: run-scoped manifests, split registries,
artifact provenance, evaluator metadata, strict-format metrics, failure-mode
aggregation, and report generation from structured result files. After that,
the program adds Countdown mechanism ablations and adjacent exact-verifier
reasoning tasks. Public reasoning benchmarks follow only after local synthetic
transfer is visible. Terminal-Bench/Harbor and tau2-bench enter later as
rootfs-managed harness smokes, not as immediate capability claims.

The primary implementation seam is the experiment registry and report contract:
one high-level interface should describe a run family, declare split and
artifact expectations, validate completed artifacts, and render reports. Task
generators, verifiers, TorchTitan training, vLLM evaluation, and external
harnesses should plug into that seam rather than each inventing their own
artifact layout.

## User Stories

1. As an experiment owner, I want every run to have an immutable run ID, so that
   later audits do not confuse fresh artifacts with stale artifacts.
2. As an experiment owner, I want run manifests to record every stage command,
   environment, start time, end time, return code, and artifact root, so that I
   can reconstruct what happened without reading terminal logs.
3. As an experiment owner, I want the manifest to record whether a stage did
   fresh work or reused existing artifacts, so that reports do not overclaim
   retraining or re-exporting.
4. As an experiment owner, I want split registries to be required for serious
   runs, so that train/evaluation leakage is caught before training or
   evaluation.
5. As an experiment owner, I want split hashes and problem-key hashes recorded,
   so that regenerated tasks can be compared to prior runs.
6. As an experiment owner, I want evaluator versions and verifier names
   recorded, so that metric changes are visible in the artifact provenance.
7. As an experiment owner, I want rootfs runtime metadata recorded, so that
   TorchTitan and vLLM results are tied to the actual hermetic environment.
8. As an experiment owner, I want CUDA device count, package versions, model
   asset path, and vLLM backend settings recorded, so that runtime differences
   do not look like model differences.
9. As an experiment owner, I want strict output-format compliance reported
   separately from arithmetic success, so that a model that solves the problem
   but misses `FINAL: <target>` is not silently treated as benchmark-clean.
10. As an experiment owner, I want verifier failure modes aggregated by split
    and arm, so that improvements can be tied to mechanisms rather than only
    pass@k.
11. As an experiment owner, I want representative examples automatically
    selected for wins, regressions, unchanged failures, and format failures, so
    that reports show behavior rather than only tables.
12. As an experiment owner, I want base-elicitable subset metrics, so that
    search compression can be measured on problems where the scaffold actually
    found a solution.
13. As an experiment owner, I want pass@1, pass@2, pass@4, pass@8, pass@16, and
    pass@32 reported consistently, so that policy compression and residual
    search behavior can be compared.
14. As an experiment owner, I want easy, elicitable, and unreached buckets kept
    in every Countdown report, so that coverage movement remains visible.
15. As an experiment owner, I want the current `clean` arm treated as the
    champion arm, so that new variants have a concrete baseline to beat.
16. As an experiment owner, I want a formatting arm, so that output-contract
    improvements can be separated from arithmetic-search improvements.
17. As an experiment owner, I want Countdown ablations to replicate across at
    least two split draws or seeds, so that one lucky split does not drive the
    next benchmark expansion.
18. As an experiment owner, I want training-size and LoRA-rank sweeps for the
    champion arm, so that the cost/performance tradeoff is visible before
    moving to broader tasks.
19. As an experiment owner, I want all real Python and GPU work to run through
    the bwrap rootfs, so that host Python state does not affect results.
20. As an experiment owner, I want rootfs-managed shell entrypoints for every
    run family, so that future agents can reproduce runs from repo-local
    commands.
21. As an experiment owner, I want local generated reasoning tasks before public
    benchmarks, so that exact verification and failure analysis stay cheap
    while the registry matures.
22. As an experiment owner, I want synthetic reasoning task registries to record
    generator source, split policy, verifier, output contract, scaffold budget,
    and training arms, so that they are as auditable as Countdown.
23. As an experiment owner, I want GSM-style public reasoning to enter with
    answer normalization tests, so that final-answer parsing errors are caught.
24. As an experiment owner, I want MATH-style public reasoning to enter only
    after smaller numeric tasks are stable, so that normalization complexity
    does not hide infrastructure problems.
25. As an experiment owner, I want no-tool public reasoning conditions to remain
    no-tool, so that upstream benchmark semantics are preserved.
26. As an experiment owner, I want Terminal-Bench/Harbor to enter first as an
    infrastructure smoke, so that container execution and artifact ingestion are
    proven before capability claims.
27. As an experiment owner, I want tau2-bench to enter first as an
    infrastructure smoke, so that trajectory and final-state score ingestion are
    proven before capability claims.
28. As an experiment owner, I want external harness boundaries documented, so
    that TorchTitan owns launch/registry/reporting while external projects own
    task execution and scoring.
29. As an experiment owner, I want benchmark results labeled as model plus
    adapter plus scaffold plus harness plus environment, so that scores are not
    misrepresented as model-only measurements.
30. As an experiment owner, I want generated checkpoints, rollouts, and large
    results ignored by git, so that the repository records code and reports
    without absorbing runtime artifacts.
31. As a future agent, I want a single spec and ticket set, so that the next
    implementation can proceed blocker-first without reconstructing the
    discussion.
32. As a reviewer, I want every promotion gate documented, so that the program
    cannot silently move from Countdown to public or agentic claims too early.
33. As a reviewer, I want current caveats preserved in reports, so that reused
    checkpoints and permissive verifier behavior are not lost in summaries.
34. As a researcher, I want results to include examples, so that qualitative
    behavior can be inspected alongside aggregate metrics.
35. As a researcher, I want exact-verifier tasks prioritized, so that early
    conclusions depend on deterministic checks rather than generic judges.
36. As a researcher, I want later agentic evaluations to preserve upstream
    metrics, so that Terminal-Bench, Harbor, and tau2-bench results remain
    comparable to their benchmark definitions.

## Implementation Decisions

- Use the bwrap rootfs as the mandatory execution boundary for real setup,
  generation, training, export, evaluation, and external harness smokes.
- Treat the experiment registry as the highest testing and integration seam.
  The registry should declare run family metadata, split expectations, arms,
  budgets, artifact roots, evaluator contracts, and external harness boundaries.
- Keep Countdown as the reference implementation for the registry. New run
  families should reuse the same concepts: split registry, scaffold budget,
  training arms, verifier, matrix validation, and report rendering.
- Make run directories immutable and run-scoped. Avoid mutable-only manifests
  such as a single `current` file for benchmark-grade results.
- Record artifact provenance for split files, training data, checkpoints,
  exported adapters, evaluator outputs, summaries, and reports.
- Record whether a stage produced fresh artifacts, skipped existing artifacts,
  or resumed from existing artifacts.
- Keep large generated data, rollouts, checkpoints, and vLLM/TorchTitan outputs
  out of git. Commit scripts, source, tests, registry examples, docs, and
  reports.
- Preserve the Qwen3 LoRA config path already added for Countdown rather than
  adding experiment-specific branches to core TorchTitan model code.
- Do not pass TorchTitan internal checkpoints directly to vLLM. Export adapters
  to PEFT/vLLM-compatible directories before LoRA evaluation.
- Add strict output-format metrics for Countdown and future exact-verifier
  tasks. Arithmetic success and format compliance are separate outcomes.
- Treat the `clean` arm as the current champion arm because it is strongest
  overall on the clean full evaluation.
- Add a `formatting` arm before broader benchmark expansion. Its purpose is to
  improve verifier-compliant output shape, not to claim new reasoning behavior.
- Require replication across at least two split draws or seeds before promoting
  a Countdown ablation result to the reasoning lane.
- Add synthetic exact-verifier reasoning before public benchmarks. Candidate
  tasks include arithmetic word problems, constraint puzzles, symbolic
  transformations, and structured grid or table reasoning.
- Add public reasoning benchmarks in order: GSM-style numeric subset, small
  MATH subset, then AIME or ARC-AGI as harder branches.
- Keep public benchmark semantics intact. Do not add tools, retrieval, DOM
  access, or judge substitutions unless the condition is explicitly labeled.
- Add Terminal-Bench/Harbor and tau2-bench only as rootfs-managed harness
  feasibility smokes until the reasoning lane has replicated held-out gains.
- For external harnesses, TorchTitan owns rootfs-managed launch, registry
  entries, artifact paths, result ingestion, and report rendering. The external
  harness owns task execution and scoring.
- Report every external benchmark score as a full evaluation object: model,
  adapter, scaffold, harness, tools, environment, benchmark release, evaluator,
  and metric.

## Testing Decisions

- Test the registry at the highest seam possible: given a declared run family
  and artifact tree, validation should accept complete consistent artifacts and
  reject missing, stale, overlapping, or schema-invalid artifacts.
- Keep unit tests focused on external behavior: generated split overlap
  rejection, exclusion behavior, pass@k summary validation, strict-format
  metrics, provenance fields, and report data extraction.
- Extend the existing Countdown unit test file for Countdown-specific behavior
  only when the behavior remains part of the Countdown experiment contract.
- Add registry-focused tests separately when the registry becomes shared across
  reasoning or agentic run families.
- Shell entrypoints should be syntax-checked with `bash -n` and should fail
  early when not run inside the rootfs for real GPU work.
- Rootfs preflight should be tested by a cheap command path that records imports
  and runtime metadata without launching expensive generation.
- Exact verifiers should have fixture tests for valid outputs, missing final
  lines, wrong final answers, unavailable operands, invalid arithmetic,
  non-exact division, and strict-format failures.
- Report generation should be tested from small fixture summaries rather than
  from full generated rollouts.
- External harness smokes should be tested for artifact ingestion and metadata
  capture, not for model capability.
- Performance or full GPU runs are not unit tests. They should remain explicit
  experiment commands with manifests and reports.

## Out of Scope

- No immediate full retraining run is required by this spec.
- No immediate Terminal-Bench, Harbor, or tau2-bench capability claim is in
  scope.
- No generic LLM-judge replacement for exact or upstream benchmark metrics is in
  scope.
- No changes to core TorchTitan training infrastructure are in scope unless the
  registry exposes a narrow reusable need.
- No broad benchmark leaderboard is in scope. The program remains a staged
  scientific ladder.
- No host-side Python path is in scope for real generation, training, or
  evaluation.
- No commit of large generated checkpoints, model assets, rollouts, or runtime
  caches is in scope.
- No claim that the current result is a model-only score is in scope; results
  must stay labeled by model, adapter, scaffold, harness, and environment.

## Further Notes

The current landed commit is `457991949 Add Countdown scaffold-to-policy
experiment`. The key final report is the clean-split final results and
infrastructure report under the Countdown experiment reports directory.

The most important immediate caveats to preserve are:

- clean-split evaluations are fresh, but full checkpoints and exported adapters
  were reused from an earlier full run;
- the current verifier can accept arithmetic traces when the final text is not
  strictly `FINAL: <target>`;
- run manifests need immutable run IDs and artifact hashes before broader
  benchmark claims;
- reasoning benchmarks should precede agentic benchmarks;
- Terminal-Bench/Harbor and tau2-bench should enter as harness smokes before
  scientific adapter evaluations.

Suggested first ticket after this spec: implement the provenance-aware
experiment registry for Countdown, including run-scoped manifests, artifact
hashes, fresh/reused stage status, runtime metadata, and report input
generation.

## Implementation Progress

- 2026-08-11: Implemented the first Countdown registry slice. Summaries now
  report strict-format pass@k and format breakdown separately from arithmetic
  success. Full/reduced pilots emit run-scoped manifest metadata, optional
  fresh/reused stage status for adapter export/eval, and a structured
  `report_input_<run_id>.json` artifact. The new `build-report-input` CLI
  validates manifest stages, split registry selection, base summaries, adapter
  matrix selection, and artifact hashes.
- 2026-08-12: Implemented and ran the full `formatting` arm. The arm trains on
  parsed operation lines plus exact `FINAL: <target>` targets, exports through
  the PEFT/vLLM adapter path, and has fresh dev/IID/OOD evaluations at 32
  rollouts per problem. The full adapter matrix now has five arms and carries
  strict-format metrics plus format breakdowns in machine-readable rows. The
  continuation report input is
  `experiments/countdown_search_distill/results/manifests/report_input_20260812T001300Z-full-formatting-continuation.json`.
