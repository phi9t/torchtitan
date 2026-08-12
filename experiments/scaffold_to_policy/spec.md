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
  smoke. The rootfs-managed entrypoint creates isolated virtualenvs under the
  ignored results tree, installs `harbor==0.21.0`,
  `terminal-bench==0.2.18`, and Sierra tau2-bench from revision
  `668d3bcd135c02aa3438f987ef45735b7c163ee3`, writes raw package
  import/version preflight artifacts, ingests them, and builds a shared report
  input. The initial run incorrectly used PyPI `tau2==2.3.3`, which is an
  unrelated magnetic-relaxation package; the corrected run installs tau2-bench
  as package `tau2==1.0.1`. Harbor/Terminal-Bench and tau2-bench require
  separate venvs because their pinned dependencies conflict in one
  environment. The corrected run passed rootfs, import, version, and CLI
  checks for `harbor`, `terminal-bench`, `tb`, and `tau2`. Follow-up tau2
  mock-domain runner probes reached tau2's task loader, retry loop, and results
  writer, but the tested no-external-LLM `dummy_user` pairings ended as
  infrastructure errors with zero evaluated tasks. This clears the corrected
  package-install gate only. Real upstream Harbor/Terminal-Bench and tau2 task
  execution, scorer invocation, and final-state or executable-test artifact
  capture remain incomplete.
- 2026-08-12: Added and ran the first tau2 scorer-ingestion smoke. The new
  rootfs-managed entrypoint installs Sierra tau2-bench from the pinned revision,
  clones the same pinned repo under the ignored results tree for its `data/`
  directory, constructs a deterministic valid mock-domain `create_task_1`
  trajectory, scores it with tau2's own evaluator using `all_ignore_basis`,
  ingests the result, and writes a report input. The completed run reached
  reward 1.0 with DB, ACTION, and COMMUNICATE reward components all equal to
  1.0. This clears tau2 task-definition and scorer ingestion for a fixture
  trajectory. It still does not clear full tau2 agent execution because no
  external benchmark agent, user simulator, local model provider, or non-fixture
  trajectory was run.
- 2026-08-12: Added harder public reasoning gates for AIME and GPQA-style
  multiple choice. The new `import-aime-split` command ingests public AIME rows
  into the conservative MATH-style exact-answer verifier, while the new
  `multiple_choice` module supports exact `FINAL: <A|B|C|D>` scoring for
  GPQA-style rows. The first AIME rootfs/vLLM smoke used
  `HuggingFaceH4/aime_2024`, 2 dev and 2 OOD examples, and 2 rollouts per
  problem. It reached pass@1/pass@2 0.000 on both splits, with failures split
  between missing final answers and wrong final values. This clears the
  hard-reasoning plumbing gate and provides a useful hard-negative calibration
  point, not a benchmark claim. The GPQA Diamond gate is implemented but, in the
  unauthenticated rootfs, writes a blocker report because `Idavidrein/gpqa` is a
  gated Hugging Face dataset requiring access credentials.
- 2026-08-12: Added and ran the next coding-lane public smoke for MBPP. The new
  `import-mbpp-split` command ingests `google-research-datasets/mbpp`
  `sanitized`, infers each entry point from the released assert tests, wraps
  those asserts into the shared executable `check(candidate)` verifier, and
  records provenance and split hashes. The completed rootfs/vLLM smoke used 2
  dev and 2 OOD examples with 2 rollouts per problem. Dev reached pass@1/pass@2
  1.000; OOD reached pass@1/pass@2 0.500, with remaining failures classified as
  type errors from wrong emitted function shape. This clears the MBPP coding
  smoke gate but remains too small for a public benchmark claim.
- 2026-08-12: Added and ran the Terminal-Bench / Harbor execution probe. The
  rootfs-managed entrypoint installs `harbor==0.21.0` and
  `terminal-bench==0.2.18` in an isolated venv, clones the pinned
  Terminal-Bench 2.1 task repo at revision
  `7131e4375048a0e408a8fb404b5f499d726b695b`, ingests a Terminal-Bench
  result-model smoke, and attempts a one-task Harbor oracle run for
  `headless-terminal`. The corrected execution path is Harbor's `harbor run`
  because the pinned task repo uses the newer `task.toml` layout, while direct
  `tb runs create` expects legacy `task.yaml` paths. The execution probe reached
  the environment boundary and failed with `Docker is not installed or not on
  PATH` inside the bwrap rootfs. This clears the package/task-layout diagnosis
  and result-ingestion gate, but full Terminal-Bench/Harbor execution remains
  blocked until Docker or an equivalent Harbor backend is available inside the
  hermetic rootfs.
- 2026-08-12: Tightened the bwrap Docker path and Harbor result ingestion. The
  rootfs now has an explicit opt-in Docker passthrough
  (`TORCHTITAN_ROOTFS_BIND_DOCKER=1`) that exposes `/usr/bin/docker`,
  `/run/docker.sock`, and the Docker Compose CLI plugin directory to the
  bubblewrap rootfs. Inside rootfs, `docker version` reported client/server
  `26.1.4` and `docker compose version` reported `v2.27.1`. The rerun
  `20260812T160000Z-terminal-bench-harbor-fixed-ingest` reached Docker Compose
  container creation, but Harbor recorded one runtime error, zero evaluated
  trials, and a missing verifier bind source path under the trial directory.
  `write_terminal_bench_execution_probe` now parses Harbor's job `result.json`
  and trial `exception_info`, so a Harbor process return code of 0 is no longer
  enough to mark the execution probe successful. The corrected report input has
  `task_execution_probes_succeeded=false`, `score=0.0`, `num_trials=0`,
  `num_errors=1`, and `num_trial_exceptions=1`. Full Terminal-Bench/Harbor
  execution remains blocked on the Harbor verifier bind-mount setup, not on
  Docker or Compose visibility inside rootfs.
- 2026-08-12: Fixed the Harbor verifier bind-mount blocker for the
  Terminal-Bench oracle probe. The Docker daemon resolves bind source paths on
  the host, but the previous rootfs run gave Harbor `/workspace/torchtitan/...`
  paths that exist only inside bwrap. When
  `TORCHTITAN_ROOTFS_BIND_DOCKER=1` is enabled, `enter_rootfs.sh` now also binds
  the checkout at its real host path and exports
  `TORCHTITAN_ROOTFS_HOST_REPO_ROOT`; `run_terminal_bench_oracle_probe.sh` uses
  that host-visible path as Harbor's working directory. The rerun
  `20260812T190000Z-terminal-bench-harbor-hostpath` completed one
  `headless-terminal` oracle trial with `n_trials=1`, `n_errors=0`, mean metric
  1.0, verifier reward 1.0, and `task_execution_probes_succeeded=true`. This
  clears Harbor/Docker/Compose/verifier-mount infrastructure for the oracle
  probe, but it is not a Terminal-Bench model or agent capability result. The
  report is
  `experiments/scaffold_to_policy/reports/20260812T190000Z-terminal-bench-harbor-hostpath.md`.
- 2026-08-12: Added and ran the ARC-AGI-2 exact-grid reasoning smoke. The new
  `arc_grid` module imports upstream ARC task JSON files, prompts with released
  train examples plus one test input, requires an exact `FINAL: <json-grid>`
  line, and scores by exact grid equality with no partial credit or LLM judge.
  The rootfs-managed entrypoint clones `https://github.com/arcprize/ARC-AGI-2.git`
  at revision `f3283f727488ad98fe575ea6a5ac981e4a188e49`. The completed
  rootfs/vLLM smoke `20260812T090000Z-arc-agi2-public-vllm-smoke` used 2 dev
  and 2 OOD training-split tasks with 2 rollouts per problem. Dev reached
  pass@1 0.000 and pass@2 0.500; OOD reached pass@1/pass@2 0.000. Failures
  split between incorrect grids and format-contract misses where the model
  emitted `FINAL: <json-grid>` literally before a grid. This clears a harder
  exact-grid reasoning smoke only; it is not an ARC-AGI benchmark claim.
- 2026-08-12: Added and ran the BigCodeBench-Hard executable coding smoke. The
  new `import-bigcodebench-split` command imports `bigcode/bigcodebench-hard`
  split `v0.1.4`, preserves the released `code_prompt`, `entry_point`, and
  `unittest` suite, and wraps the suite into the shared executable
  `check(candidate)` verifier. The first run
  `20260812T090500Z-bigcodebench-hard-public-vllm-smoke` exposed a missing
  rootfs `matplotlib` dependency on an OOD task. After installing `matplotlib`
  inside the bwrap rootfs, the repaired run
  `20260812T091500Z-bigcodebench-hard-public-vllm-smoke` used 2 dev and 2 OOD
  examples with 2 rollouts per problem. Dev and OOD both reached
  pass@1/pass@2 0.000, and all final failures were benchmark-test assertion
  failures. This clears the BigCodeBench-Hard executable harness smoke and
  rootfs dependency repair, but remains too small for a public benchmark claim.
- 2026-08-12: Added an explicit completion audit and the first unblocked
  BigCodeBench-Hard hardening item. The audit report is
  `experiments/scaffold_to_policy/reports/20260812T123000Z-completion-audit-and-next-steps.md`
  and marks the overall spec incomplete because GPQA auth, Terminal-Bench/Harbor
  Docker or equivalent backend support, and full tau2 agent execution remain
  blocked or incomplete. The new `preflight-coding-style-canonical` command
  runs released canonical solutions through the same executable verifier before
  vLLM generation, writes per-problem preflight artifacts, and lets coding
  report inputs require those artifacts. `run_bigcodebench_hard_public_vllm_smoke.sh`
  now performs that preflight for dev and OOD before spending GPU time, so
  missing rootfs packages are surfaced as early infrastructure failures instead
  of late scoring failures.
- 2026-08-12: Ran the expanded BigCodeBench-Hard calibration requested by the
  completion audit. The run
  `20260812T173000Z-bigcodebench-hard-public-vllm-expanded` used 8 dev tasks
  from offset 0, 8 OOD tasks from offset 72, 4 rollouts per problem, the pinned
  `bigcode/bigcodebench-hard` split `v0.1.4`, and the same no-tool vLLM coding
  prompt/verifier path. To make this slice reproducible from the rootfs
  entrypoint, `run_bigcodebench_hard_public_vllm_smoke.sh` now installs the
  explicit packages exposed by canonical preflight: `flask`, `flask-login`,
  `flask-wtf`, `pycryptodome`, `rsa`, `seaborn`, and `wordcloud`. Both final
  selected splits passed canonical preflight 8/8. Qwen3-1.7B reached dev and
  OOD pass@1/pass@4 0.000, with all 64 sampled candidates failing released unit
  tests as assertion failures. Rejected OOD offset 64 remains documented as a
  preflight-blocked slice because two released canonical solutions failed under
  current rootfs library versions. The report is
  `experiments/scaffold_to_policy/reports/20260812T173000Z-bigcodebench-hard-expanded-calibration.md`.
- 2026-08-12: Added the BigCodeBench-Hard `contract_chat` prompt condition and
  a GPU-memory preflight to the BigCodeBench runner. `contract_chat` keeps the
  no-tool executable-test condition but makes the system prompt explicit about
  preserving the requested function signature and matching side effects,
  returns, error behavior, and library calls. It does not expose tests or
  verifier feedback. The attempted same-slice rerun
  `20260812T183000Z-bigcodebench-hard-contract-chat-blocked` passed canonical
  preflight 8/8 on dev and 8/8 on OOD, then stopped before vLLM generation
  because `preflight-vllm-gpu-memory` found only 4.729 GiB free on cuda:0 versus
  160.516 GiB required at `gpu_memory_utilization=0.9`. At inspection time all
  eight B200s were occupied by unrelated `sglang::scheduler` processes. This is
  a blocked prompt-condition attempt, not a model result. The report is
  `experiments/scaffold_to_policy/reports/20260812T183000Z-bigcodebench-contract-chat-blocked.md`.
- 2026-08-12: Ran the larger ARC-AGI-2 exact-grid calibration requested by the
  completion audit. The run
  `20260812T131000Z-arc-agi2-public-vllm-calibration` used 8 dev and 8 OOD
  training-split tasks, 4 rollouts per problem, the same pinned ARC revision
  `f3283f727488ad98fe575ea6a5ac981e4a188e49`, and the same exact
  `FINAL: <json-grid>` verifier. Because another process occupied most local
  B200 memory, `evaluate-arc-grid-vllm` now supports
  `--gpu-memory-utilization`, and the completed run used
  `GPU_MEMORY_UTILIZATION=0.24`. Because one selected ARC prompt exceeded 4096
  tokens, the completed run used `SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN=8192`.
  Dev reached pass@1 0.000 and pass@4 0.125 with one elicitable problem; OOD
  reached pass@1/pass@4 0.000. The dominant failures were parse/format failures
  around the strict final-grid contract, not rootfs or scorer failures. The
  report is
  `experiments/scaffold_to_policy/reports/20260812T131000Z-arc-agi2-calibration.md`.
- 2026-08-12: Hardened ARC-AGI-2 prompt/context validation. The new
  `preflight-arc-grid-prompts` command tokenizes ARC prompts with the same
  tokenizer and prompt variant used by vLLM, records per-problem prompt and
  total token counts, and lets ARC report inputs require prompt preflight
  artifacts. `run_arc_agi2_public_vllm_smoke.sh` now runs this preflight before
  generation and passes the same `MAX_MODEL_LEN` to preflight and vLLM. On the
  completed calibration slice, the preflight reproduced the 4096-context
  blocker: only 6 of 8 dev problems fit when reserving 768 output tokens, with
  `ARC-AGI-2/009d5c81/0` and `ARC-AGI-2/00d62c1b/0` exceeding context. The
  8192-context preflight selected all dev and OOD problems; OOD max total tokens
  were 7497.
- 2026-08-12: Added and ran a rootfs-managed tau2 upstream execution probe. The
  new `run_tau2_execution_probe.sh` entrypoint installs Sierra tau2-bench from
  revision `668d3bcd135c02aa3438f987ef45735b7c163ee3`, clones the pinned repo
  for its `data/` directory, runs `tau2 check-data`, launches a bounded
  `tau2 run` for mock-domain task `create_task_1`, and ingests tau2's saved
  `results.json`. The completed probe
  `20260812T132500Z-tau2-execution-probe` reached tau2's batch runner and
  results writer, but tau2 recorded `termination_reason: infrastructure_error`,
  zero evaluated tasks, and
  `DummyUser.__init__() got an unexpected keyword argument 'tools'`. This
  improves blocker evidence for the tau2 lane, but it is not a successful tau2
  benchmark result and does not clear the full agent-execution gate.
- 2026-08-12: Added an ARC-AGI-2 `strict_chat` prompt condition and attempted
  the same 8 dev / 8 OOD calibration slice as run
  `20260812T134500Z-arc-agi2-strict-chat-calibration`. The prompt condition
  keeps the same no-tool exact-grid verifier but asks for exactly one line,
  `FINAL: <json-grid>`, with no reasoning, markdown, labels, or code fences.
  Prompt/context preflight passed all selected problems at 8192 context, with
  max total tokens 5260 for dev and 7516 for OOD. Generation did not run because
  the visible GPU had insufficient free memory for vLLM. A new
  `preflight-vllm-gpu-memory` command and ARC runner hook now write a JSON
  blocker artifact before vLLM initialization; on the blocked host state it
  recorded 6.50 GiB free versus 42.80 GiB required at
  `GPU_MEMORY_UTILIZATION=0.24`. The report is
  `experiments/scaffold_to_policy/reports/20260812T134500Z-arc-strict-chat-blocked.md`.
- 2026-08-12: Re-ran the ARC-AGI-2 `strict_chat` condition after GPU memory
  became available as run `20260812T150000Z-arc-agi2-strict-chat-calibration`.
  The run completed through rootfs/vLLM and wrote report input under
  `experiments/scaffold_to_policy/results/arc_agi2_public_vllm_calibration_strict_chat_rerun/`.
  Prompt/context and GPU-memory preflights all passed. The strict prompt reached
  dev/OOD pass@1/pass@4 0.000. It eliminated missing-final failures but
  converted most failures into wrong-grid outputs and lost the one dev success
  from the earlier `chat` calibration. This argues against replacing the ARC
  prompt with `strict_chat`; a format-repair or two-stage final-emission
  condition is the better next ARC branch. The report is
  `experiments/scaffold_to_policy/reports/20260812T150000Z-arc-strict-chat-results.md`.
- 2026-08-12: Fixed the Terminal-Bench/Harbor bwrap-to-Docker host-path
  boundary. The rootfs entrypoint now binds the checkout at its real host path
  when Docker passthrough is enabled, and Harbor runs from that host-visible
  path so verifier bind mounts exist from the Docker daemon's point of view.
  The run `20260812T190000Z-terminal-bench-harbor-hostpath` completed one
  pinned `headless-terminal` oracle trial with Harbor `n_trials=1`,
  `n_errors=0`, mean metric 1.0, and verifier reward 1.0. This clears the
  Harbor/Docker/Compose/verifier-mount infrastructure smoke, but it remains an
  oracle harness result rather than a model or agent capability claim.
- 2026-08-12: Continued tau2 upstream execution probing. The initial solo
  default exposed a tau2 revision mismatch: `DummyUser` is required for solo
  mode but its constructor does not accept the `tools` argument that the runner
  now passes. The runner default now uses tau2's constructor-compatible
  `llm_agent` plus `user_simulator` pair and exposes `TAU2_USER_LLM`. In a
  hermetic offline run with `fake` model names, tau2 reaches the simulation loop
  but LiteLLM rejects `model=fake` as an unknown provider, producing one infra
  error and zero evaluated tasks. The next tau2 step is a real compatible
  provider endpoint inside rootfs or a benchmark-preserving deterministic
  provider/agent path that still writes and scores tau2's official
  `results.json`.
- 2026-08-12: Hardened the shared vLLM memory controls for harder reasoning and
  coding. The `preflight-vllm-gpu-memory` command now writes structured blocker
  JSON even when CUDA memory discovery itself fails under extreme contention,
  and the vLLM evaluators for arithmetic, modular sequences, GSM-style, MATH /
  AIME style, coding, and multiple choice now accept and forward
  `--gpu-memory-utilization`. AIME and GPQA rootfs runners now run the same GPU
  memory preflight before model launch. Retried BigCodeBench-Hard
  `contract_chat` and GPQA runs remained blocked by shared-machine state:
  GPQA is gated without `HF_TOKEN`, and unrelated root-owned SGLang processes
  consumed nearly all eight B200s, in one case causing CUDA memory-query OOM.
- 2026-08-12: Unblocked the tau2 upstream execution probe without adding a
  provider dependency. The runner now invokes a small explicit Python wrapper
  that imports `torchtitan.experiments.scaffold_to_policy.tau2_probe_agent`
  before dispatching to `tau2.cli.main`, because this virtualenv layout did not
  import `sitecustomize.py` at startup. The probe module registers a
  non-solo deterministic mock-domain agent plus a static deterministic user,
  then still runs Sierra tau2-bench's pinned `tau2 run` path and ingests
  tau2's official `results.json`. Run
  `20260812T105500Z-tau2-deterministic-execution-probe` completed one
  `create_task_1` simulation with return code 0, `num_evaluated=1`,
  `num_infra_errors=0`, and `task_execution_probes_succeeded=true`. This
  clears tau2 task-execution plumbing only; it is a deterministic harness probe,
  not a model capability score.
- 2026-08-12: Rechecked harder reasoning/coding blockers after the tau2 fix.
  GPQA Diamond still stops at import because `Idavidrein/gpqa` is gated and no
  `HF_TOKEN` is available inside the rootfs; the runner wrote blocker manifest
  `experiments/scaffold_to_policy/results/gpqa_public_vllm_auth_blocker/manifests/report_input_20260812T110000Z-gpqa-auth-blocker.json`.
  The current vLLM GPU preflight also remains blocked: device 0 had 3.93 GiB
  free versus 133.76 GiB required at `GPU_MEMORY_UTILIZATION=0.75`, and
  host-side `nvidia-smi` showed all eight B200s occupied by SGLang scheduler
  processes. No new BigCodeBench-Hard, ARC, AIME, or GPQA model score should be
  inferred from this blocker refresh.
- 2026-08-12: Advanced Terminal-Bench/Harbor beyond the oracle-only
  infrastructure probe by running Harbor's built-in `nop` agent on the same
  pinned `headless-terminal` task. The Terminal-Bench runner now accepts
  `HARBOR_AGENT` and the ingestion layer separates completed task execution
  from benchmark reward via `task_execution_probes_completed`. Run
  `20260812T111500Z-terminal-bench-harbor-nop-baseline` completed one non-oracle
  trial with Harbor `n_trials=1`, `n_errors=0`, no trial exceptions, mean metric
  0.0, `task_execution_probes_completed=true`, and
  `task_execution_probes_succeeded=false`. This is a valid bounded baseline and
  verifier-path check, not a policy capability success.
- 2026-08-12: Added the matching tau2 non-oracle baseline. The tau2 probe
  registration module now includes `torchtitan_noop_agent`, a half-duplex agent
  that stops without tool calls, while the execution-ingestion layer reports
  tau2's official average reward separately from completed execution. Run
  `20260812T113000Z-tau2-noop-baseline` completed one upstream tau2
  `create_task_1` simulation with return code 0, `num_evaluated=1`,
  `num_infra_errors=0`, `average_reward=0.0`,
  `task_execution_probes_completed=true`, and
  `task_execution_probes_succeeded=false`. The official reward breakdown was
  DB 0.0 and COMMUNICATE 1.0. This is a bounded baseline and evaluator-path
  check, not a learned policy result.
- 2026-08-12: Hardened public hard-reasoning and coding report-input
  provenance. The new shared `report_artifacts` helper records per-artifact
  path, existence, size, sha256, mtime, and whether the artifact is run-scoped
  by path or JSON payload. `math_style`, `multiple_choice`, `arc_grid`, and
  `coding_style` report inputs now include `artifacts.details`,
  `artifacts.freshness`, and an `artifact_provenance_labeled` check. This does
  not mark reused artifacts as failures; it makes fresh-vs-reused status
  machine-readable so later audits do not overclaim stale or unscoped outputs.
- 2026-08-12: Revisited the BigCodeBench-Hard `contract_chat` hard coding
  condition after GPU memory became briefly available. A completed 2 dev / 2
  OOD rerun, `20260812T235500Z-bigcodebench-hard-contract-chat-rerun`, passed
  canonical preflight on both splits and reached dev/OOD pass@1 through pass@4
  of 0.000, with all 16 candidates failing released tests as assertion
  failures. An expanded 8 dev / 8 OOD attempt initially exposed a canonical
  preflight timeout on `BigCodeBench/17`; rerunning with
  `TIMEOUT_SECONDS=30` selected all 16 canonical solutions and completed the
  8-problem dev half with pass@1 through pass@4 of 0.000 over 32 candidates.
  The OOD half remained infrastructure-blocked: after a successful memory
  preflight, shared-GPU state changed and vLLM saw only 6.6 GiB free at startup
  versus 21.4 GiB requested at `GPU_MEMORY_UTILIZATION=0.12`. This leaves the
  completed small contract-chat result and a partial expanded dev result, not a
  completed expanded BigCodeBench-Hard result.
- 2026-08-12: Factored the common exact-verifier report-input shell into
  `report_artifacts.build_report_input`. Arithmetic words, GSM-style, MATH /
  AIME style, GPQA-style multiple choice, ARC-AGI-2 grid, executable coding,
  and modular-sequence transfer reports now share split-registry validation,
  summary count matching, optional preflight count checks, artifact hashing,
  and fresh-vs-reused artifact labeling. Task modules still own their verifier
  metadata, scaffold labels, task-specific preflight names, and transfer
  analysis. This materially advances the shared registry/reporting seam without
  changing benchmark semantics or treating reused artifacts as failures.
- 2026-08-12: Added canonical latest-report selection for scaffold report
  inputs. `report_artifacts.build_latest_report_index` scans
  `report_input_*.json` files, extracts embedded `run.run_id`, optionally
  filters by task, records whether each candidate's checks passed, and writes a
  JSON index with the selected latest candidate plus all candidates. The CLI
  command `write-latest-report-index` exposes this path for rootfs audits. A
  rootfs smoke over the BigCodeBench-Hard `contract_chat` manifests selected
  `20260812T235500Z-bigcodebench-hard-contract-chat-rerun` with a fresh
  run-bound report artifact and passing checks. This avoids mutable-only
  `current` pointers while preserving auditable candidate history.
- 2026-08-12: Extended artifact provenance to external-harness report inputs
  without forcing them into the split-summary report builder. The
  `external_harness.build_report_input` path now records artifact hashes,
  mtimes, fresh-vs-reused labels, and an `artifact_provenance_labeled` check
  for ingested Harbor/Terminal-Bench and tau2 artifacts while preserving
  harness-specific completion and reward semantics. This closes the remaining
  provenance-surface gap for external harness smokes; model or learned-policy
  agent execution remains a separate incomplete requirement.
