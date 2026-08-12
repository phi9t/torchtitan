---
title: Scaffold-To-Policy Experiment Registry And Reasoning Expansion
labels:
  - ready-for-agent
status: executed-checkpoint
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

The current Countdown implementation has moved from the original pilot to an
executed replication checkpoint. The key reports are:

- `experiments/countdown_search_distill/reports/20260811T222454Z-clean-split-final-results-and-infra-report.md`
- `experiments/countdown_search_distill/reports/20260812T001300Z-formatting-arm-results-and-audit.md`
- `experiments/countdown_search_distill/reports/20260812T002500Z-base-elicitable-and-examples.md`
- `experiments/countdown_search_distill/reports/20260812T043000Z-replication-rank-size-sweep.md`
- `experiments/countdown_search_distill/reports/20260812T054500Z-formatting-replication.md`
- `experiments/scaffold_to_policy/reports/20260812T055500Z-arithmetic-words-vllm-smoke.md`

The most important immediate caveats to preserve are:

- the original clean-split full report reused full checkpoints and exported
  adapters from an earlier full run, while later formatting, clean
  replication, and formatting replication cells produced fresh scoped
  artifacts;
- arithmetic pass@k and strict `FINAL: <target>` pass@k must remain separate;
- run manifests and report inputs now exist, but broader benchmark claims still
  need canonical latest-row selection, stronger fresh/reused stage surfacing,
  and offline dataset-loader hardening;
- reasoning benchmarks should precede agentic benchmarks;
- Terminal-Bench/Harbor and tau2-bench should enter as harness smokes before
  scientific adapter evaluations.

Suggested first ticket after this checkpoint: run and calibrate the new
`modular_sequences` exact-verifier task, then promote the first nontrivial
local reasoning task to a small scaffold-to-policy transfer run before public
reasoning benchmarks.

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
- 2026-08-12: Extended the Countdown report input with artifact-derived
  `analysis`: base-elicitable subset metrics and deterministic representative
  examples for wins, regressions, unchanged failures, and format failures. This
  closes the first reporting gap needed before replication/sweep and reasoning
  transfer work.
- 2026-08-12: Added isolated Countdown replication/sweep infrastructure.
  Countdown runners now support explicit data/results roots, qwen3 Countdown
  LoRA configs read those roots plus the requested LoRA rank, and
  `run_replication_sweep.sh` creates rootfs-managed second-draw or champion-arm
  sweep runs under `experiments/countdown_search_distill/sweeps/replication/`.
  This implements the entrypoint needed to run the required second seed or
  split draw and training-size/LoRA-rank sweeps without overwriting completed
  pilot artifacts.
- 2026-08-12: Executed the clean-arm replication and rank-size sweep under the
  bwrap rootfs. The completed sweep covered seed 43, train sizes 1000 and
  2000, LoRA ranks 32 and 16, and dev/IID/OOD adapter evaluation at 32 rollouts
  per problem. All four cells produced checkpoints, PEFT/vLLM adapters, adapter
  matrices, and report inputs under isolated sweep roots. The strongest cell
  was `seed43_train2000_rank16`, with dev pass@1 0.192, dev pass@32 0.918, IID
  pass@32 0.905, and OOD pass@32 0.914. On its OOD base-elicitable subset,
  `clean` reached 0.351 pass@1 and 1.000 pass@32 over 279 problems. This clears
  the clean-arm replication gate and shifts the next Countdown work to
  formatting-champion replication plus reasoning-task transfer.
- 2026-08-12: Executed the formatting-champion replication under the same
  rootfs-managed isolated sweep layout. The run used seed 43, train size 2000,
  LoRA rank 16, and the `formatting` arm. It produced a fresh TorchTitan
  checkpoint, PEFT/vLLM adapter export, dev/IID/OOD base and adapter
  evaluations, adapter matrix, report input, and timestamped report. The
  formatting adapter reached dev pass@1 0.278, dev pass@32 0.916, OOD pass@1
  0.302, and OOD pass@32 0.914. On the OOD base-elicitable subset it reached
  0.382 strict pass@1 and 0.993 strict pass@32 over 280 problems. This clears
  the formatting replication gate and makes reasoning-task transfer the active
  next lane.
- 2026-08-12: Added the first real-model reasoning smoke for
  `arithmetic_words`. The existing fixture smoke remains a CPU-testable
  registry/verifier path; the new
  `experiments/scaffold_to_policy/run_arithmetic_words_vllm_smoke.sh` entrypoint
  runs Qwen3-1.7B generation through vLLM inside the bwrap rootfs, verifies
  dev/OOD outputs with the strict final-integer checker, and writes a report
  input under `experiments/scaffold_to_policy/results/arithmetic_words_vllm_smoke/`.
- 2026-08-12: Ran the arithmetic-words vLLM smoke. The rootfs/vLLM/verifier
  path completed and report-input checks passed, but the generated task was too
  easy: dev and OOD both reached 1.000 pass@1 and strict pass@1 over four
  problems. This is infrastructure evidence only; the next reasoning step is
  calibrated harder local reasoning generation.
- 2026-08-12: Added `modular_sequences` as the second local exact-verifier
  reasoning task. It generates modular recurrence problems, verifies a strict
  `FINAL: <integer>` answer against deterministic recurrence execution, reports
  pass@k plus easy/elicitable/unreached buckets, and has rootfs-managed fixture
  and vLLM smoke entrypoints.
- 2026-08-12: Calibrated `modular_sequences` with Qwen3-1.7B/vLLM inside the
  bwrap rootfs. The useful band is 3-5 recurrence steps and modulus 37-257 with
  8 rollouts and 512 generated tokens: dev pass@1 0.250, pass@8 0.625, buckets
  easy 2, elicitable 3, unreached 3; OOD pass@1 0.500, pass@8 0.625, buckets
  easy 4, elicitable 1, unreached 3. Harder settings were mostly unreached, so
  the next step is a small scaffold-to-policy transfer run using this calibrated
  band before public reasoning benchmarks.
- 2026-08-12: Implemented and ran the first `modular_sequences`
  scaffold-to-policy transfer smoke. The new rootfs-managed entrypoint collects
  base train rollouts, builds SFT examples only from verifier-successful
  rollouts, evaluates base dev/OOD, trains a TorchTitan Qwen3-1.7B LoRA adapter,
  exports it to PEFT/vLLM format, evaluates the exported adapter through vLLM
  LoRA loading, and writes a run-scoped report input. The minimal completed run
  used 16 train, 4 dev, 4 OOD, 8 train rollouts, 4 eval rollouts, and 2 training
  steps. It produced the expected checkpoint, adapter export, base summaries,
  adapter summaries, stage manifest, and report input under
  `experiments/scaffold_to_policy/results/modular_sequences_transfer_smoke_min/`.
  This clears the modular transfer plumbing gate, but not the scientific
  transfer gate: dev pass@1 improved from 0.250 to 0.500, while OOD pass@k
  regressed from 0.500 to 0.250 on the four-problem smoke. The next step is a
  larger modular transfer run with enough dev/OOD examples to evaluate
  base-elicitable movement and OOD retention before public reasoning benchmarks.
- 2026-08-12: Added modular transfer analysis to report inputs and ran the
  larger `modular_sequences` transfer experiment under the bwrap rootfs. The
  run used 64 train, 32 dev, 32 OOD, 8 rollouts, and 24 TorchTitan LoRA steps.
  It collected 44 verifier-successful train examples, trained and exported a
  PEFT/vLLM-compatible Qwen3-1.7B LoRA adapter, evaluated base and adapter
  dev/OOD through vLLM, and wrote a report input with base-elicitable subset
  metrics plus representative examples. Dev pass@1 improved from 0.531 to
  0.625 and OOD pass@1 improved from 0.375 to 0.594. On base-elicitable
  subsets, adapter pass@1 reached 0.750 over 4 dev problems and 0.800 over 10
  OOD problems while pass@8/pass@32 stayed at 1.000. This clears the local
  synthetic reasoning transfer gate and makes the next stage public
  GSM-style answer-normalization and fixture smoke work, followed by a small
  public reasoning run if the verifier semantics are stable.
- 2026-08-12: Added `gsm_style` as the first public-reasoning-adjacent verifier
  stage. It ingests JSONL problems with question/answer fields, normalizes
  final answers from strict `FINAL:` lines and GSM8K `####` markers, and covers
  commas, currency markers, boxed answers, integers, decimals, simple
  fractions, and negative numbers. The rootfs-managed fixture smoke prepares
  checked-in train/dev/OOD fixture splits, validates split overlap, evaluates
  fixture rollouts, and writes a report input. This clears the GSM-style
  normalization smoke gate; the next stage is a small no-tool GSM-style
  real-model evaluation using the same verifier before any MATH-style or
  agentic harness expansion.
- 2026-08-12: Added and ran the no-tool GSM-style vLLM smoke with Qwen3-1.7B
  inside the bwrap rootfs. The new rootfs-managed entrypoint prepares the
  checked-in GSM-style fixture splits, validates split overlap, prompts vLLM
  directly with chat-formatted no-tool math prompts, verifies generations with
  `gsm_style_normalized_final_v1`, and writes a report input. The smoke used
  3 dev and 3 OOD fixture problems with 4 rollouts per problem. Dev reached
  1.000 pass@1 and strict-format pass@1; OOD also reached 1.000 pass@1 and
  strict-format pass@1, with one non-first-sample OOD format failure where the
  model emitted `Final: <answer>`. This clears the first real-model GSM-style
  plumbing gate only. It is not a public benchmark result because it uses
  checked-in local fixtures rather than a pinned public GSM8K/GSM-style dataset
  revision.
- 2026-08-12: Added and ran the first pinned public GSM8K no-tool smoke. The
  new `import-gsm8k-split` command imports `openai/gsm8k` rows at explicit
  dataset revision `740312add88f781978c0658806c59bc2815b9866`, converts the
  GSM8K `####` answer into the shared `gsm_style` problem format, records
  per-split provenance, validates split overlap, evaluates Qwen3-1.7B with
  vLLM inside the bwrap rootfs, and writes a report input with the actual
  scaffold budget. The completed smoke used 8 dev and 8 OOD examples from the
  GSM8K test split, 4 rollouts per problem, and reached dev pass@1 0.625,
  dev pass@4 0.750, OOD pass@1 0.625, and OOD pass@4 0.750. This clears the
  first public reasoning plumbing and calibration gate, but it remains too
  small for a public benchmark claim or adapter-training conclusion.
- 2026-08-12: Added and ran the first pinned public MATH-style no-tool smoke.
  The new `math_style` module imports `EleutherAI/hendrycks_math` algebra rows
  at explicit dataset revision `21a5633873b6a120296cce3e2df9d5550074f4a3`,
  extracts boxed dataset answers, normalizes final answers conservatively, and
  records verifier limitations in report inputs. The completed rootfs/vLLM run
  used 8 dev and 8 OOD examples from the algebra test split with 4 rollouts per
  problem. After rescoring saved generations with observed normalization fixes,
  dev reached pass@1 0.750 and pass@4 0.750; OOD reached pass@1 0.750 and
  pass@4 0.875. This clears the first harder public reasoning smoke, but it
  remains too small for a benchmark claim and the verifier intentionally does
  not score symbolic algebra equivalence, interval/set semantics, matrices, or
  multi-answer semantics.
- 2026-08-12: Added and ran the first coding-lane smoke. The new
  `coding_style` module imports `openai/openai_humaneval` rows at explicit
  dataset revision `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`, prompts
  Qwen3-1.7B through vLLM inside the bwrap rootfs, extracts candidate Python
  code, and scores it by running the benchmark tests in a separate Python
  subprocess with a timeout. The completed smoke used 2 dev and 2 OOD examples
  with 2 rollouts per problem. After rescoring saved generations with a
  corrected prompt-preamble extraction rule, dev reached pass@1 0.500 and
  pass@2 0.500; OOD stayed at pass@1/pass@2 0.000 with assertion failures. This
  clears an executable-code harness gate only; it is not a HumanEval,
  LiveCodeBench, SWE-bench, Terminal-Bench, or Harbor benchmark claim.
- 2026-08-12: Added and ran the first external-harness dry-run ingestion smoke.
  The new `external_harness` module records pinned Harbor, Terminal-Bench 2.1,
  and tau2-bench revisions, rootfs/runtime metadata, installed-package state,
  raw score-like artifacts, trajectory fixtures, ingested artifacts, and a
  shared report input. The completed rootfs run used Harbor revision
  `b7e2f71b4563618af3a42279740f5f412dcf7046`, Terminal-Bench 2.1 revision
  `7131e4375048a0e408a8fb404b5f499d726b695b`, and tau2-bench revision
  `668d3bcd135c02aa3438f987ef45735b7c163ee3`. It confirmed rootfs launch,
  pin capture, dry-run score/trajectory ingestion, and report-input checks, but
  explicitly records that `harbor`, `terminal_bench`, and `tau2` are not
  installed in the current rootfs and that Docker/bwrap binaries are not
  available inside it. This clears only the dry-run ingestion contract; real
  Harbor/Terminal-Bench and tau2 task execution remains incomplete.
- 2026-08-12: Added and ran the installed external-harness package preflight
  smoke. The new rootfs-managed entrypoint creates an isolated virtualenv under
  the ignored results tree, installs `harbor==0.21.0`,
  `terminal-bench==0.2.18`, and `tau2==2.3.3`, writes raw package
  import/version preflight artifacts, ingests them, and builds a shared report
  input. The completed run passed rootfs, import, and version checks for
  Harbor, Terminal-Bench, and tau2 packages, while recording that no `harbor`,
  `terminal-bench`, `tb`, or `tau2` CLI executable is exposed by those package
  installs. This clears the pinned package-install gate only. Real upstream
  Harbor/Terminal-Bench and tau2 task execution, scorer invocation, and
  final-state or executable-test artifact capture remain incomplete.
