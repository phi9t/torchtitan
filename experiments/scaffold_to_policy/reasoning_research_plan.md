# Scaffold-To-Policy Reasoning Research Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to execute this plan task by task. Checkboxes
> track future work; the evidence ledger explicitly identifies work that has
> already run.

**Goal:** Determine when verified inference-time reasoning can be compressed
into first-sample policy behavior by TorchTitan SFT, whether deterministic
online RL adds value, and how far the effect transfers across exact-verifier
reasoning tasks.

**Architecture:** Treat benchmark difficulty and scientific maturity as
independent ladders. Each task advances from contract proof through calibration,
scaffold reachability, matched SFT, replication, deterministic RL, bounded
asynchronous RL, and selected scale confirmation. Repo-local run-attempt
evidence is canonical; Temporal provides durable orchestration for coarse,
idempotent stages but does not own scientific truth.

**Execution stack:** Repo-local split registries and report inputs, bwrap
rootfs runners, vLLM generation, TorchTitan core SFT, the separate Monarch
online-RL surface, exact task verifiers, local Temporal Python SDK workflows,
and optional TensorBoard or W&B projections.

**Status:** Approved research design translated into an execution plan. This
document is not a claim that unchecked experiments have run.

**Document boundary:**
[`followup_experiment_plan.md`](followup_experiment_plan.md) owns the
cross-program dependency graph and promotion state;
[`runtime_preflight_roadmap.md`](runtime_preflight_roadmap.md) owns lifecycle,
rootfs, Temporal, and preflight implementation; and
[`training_research_plan.md`](training_research_plan.md) owns shared SFT,
policy-bridge, deterministic-RL, and async-RL mechanics. This document owns the
reasoning-task hypotheses, datasets, verifiers, experiment cells, analyses,
and task-level promotion decisions.

This is a Phase 4 post-training program under
[`docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md`](../../docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md).
Implementation and scientific compute follow the core observability,
Qwen3-0.6B/1.7B 2-/4-/8-GPU foundation certification, Qwen3-8B and 30B-A3B
promotion, and multimodal phase gates. The reasoning lifecycle consumes the
certified run-attempt schema rather than defining a parallel one.

## Global constraints

- Preserve the three execution surfaces. Offline generation/SFT/evaluation is
  a repo-local research program, core SFT runs through `run_train.sh ->
  torchrun -> Trainer`, and online RL runs through the Monarch controller. Do
  not turn one surface into a mode of another.
- Keep the dependency direction `experiments -> core`. Temporal, vLLM, public
  datasets, and benchmark packages remain optional experiment dependencies.
- Run real generation, training, export, evaluation, and benchmark work through
  the repo-local shell entrypoints and `scripts/rootfs/enter_rootfs.sh`. Host
  unit tests and static checks remain directly runnable.
- Use the versioned repo-local run-attempt evidence contract from
  [`docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md`](../../docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md).
  TensorBoard is the compact scalar dashboard. Remote services are optional
  mirrors, never the source of artifact or promotion truth.
- Use the capture and overhead policy in
  [`docs/research/2026-08-12-training-observability-paper-closure.md`](../../docs/research/2026-08-12-training-observability-paper-closure.md).
  Tier 0 observability must stay within a measured 1% median steady-state
  throughput regression.
- Preserve globally disjoint problem identities, immutable run and attempt
  lineage, artifact hashes, canonical work status and freshness, exact or task-specific
  verifiers, and separate task-success and strict-format metrics.
- A smoke proves plumbing only. A deterministic comparison proves only the
  promised identity. A representative matched run supports convergence or
  performance. Replicated held-out evaluation is required for a quality claim.
- Do not commit generated datasets, checkpoints, rollouts, caches, or large
  result trees. Commit source, tests, registries, compact summaries, and
  timestamped reports.
- Keep final evaluation locked. Opening a final split changes its status to
  development evidence and requires a new locked confirmation set for future
  claims.
- Use ASCII in new code, comments, and rewritten documentation.

## 1. Research questions and permitted claims

The program answers five ordered questions:

1. **Reachability:** Does a frozen base policy solve a nontrivial set of
   problems within a fixed scaffold budget while leaving meaningful pass@1
   headroom?
2. **Compression:** Does verified scaffold data improve held-out first-sample
   task success under matched SFT?
3. **Mechanism:** Is the improvement semantic, an output-contract effect, or a
   mixture of both?
4. **Transfer:** Does the selected policy improve an adjacent task without
   training on that task's locked evaluation data?
5. **Post-training:** Does deterministic on-policy RL improve a replicated SFT
   policy, and can bounded asynchronous RL preserve quality while improving
   measured throughput?

The strongest permitted claim at each stage is deliberately narrow:

| Claim class | Required evidence | Disallowed wording |
| --- | --- | --- |
| Plumbing | Contract, rootfs, canonical-solution, and report checks | Capability improvement |
| Calibration | Representative fixed base slice and failure analysis | Benchmark ranking or transfer |
| Exploratory SFT | One matched held-out pilot | Replicated or robust effect |
| Replicated SFT | Predeclared independent draws/seeds and paired analysis | General reasoning improvement |
| Cross-task transfer | Locked adjacent task, no task-specific training leakage | Task-specific learning on the target |
| Deterministic RL | Exact parity setup and replicated causal factorial | Production asynchronous performance |
| Async RL | Bounded policy age, matched quality, and separated timings | On-policy or deterministic identity |
| Scale confirmation | Selected arm/control pair after smaller-model gates | Broad scaling law |

No task is required to reach RL. A task may remain a saturation control,
calibration benchmark, transfer-only benchmark, or hard negative.

## 2. Orthogonal ladders

### 2.1 Scientific maturity

| Level | Name | Minimum evidence |
| --- | --- | --- |
| M0 | Contract | Dataset and split identity, verifier, canonical preflight, rootfs profile, report schema |
| M1 | Base calibration | Representative base pass@1/pass@K, strict-format metrics, failure and example audit |
| M2 | Scaffold reachability | Fixed scaffold improves coverage enough to yield verified training targets |
| M3 | Matched SFT pilot | Base, raw, clean, and output-contract control on one held-out draw |
| M4 | Replicated SFT | Frozen base, champion, and causal anchor on at least two fresh data draws by two training seeds, with cellwise paired uncertainty; this minimum supports a claim conditional on the realized policies |
| M5 | Deterministic RL | Strict on-policy parity plus the base/SFT/base-to-RL/SFT-to-RL factorial |
| M6 | Bounded async RL | Matched quality, bounded consume-time policy age, and a useful throughput gain |
| M7 | Scale confirmation | Selected arm and causal control at a promoted larger model, with prospectively sized scale-level seed replication and paired final-set uncertainty; one larger-model run is only a probe |

Scientific maturity is not task difficulty. A hard M1 zero is not more mature
than a simple M4 replicated result.

The minimum `2 draws x 2 seeds` M4 matrix is a reproducibility floor, not an
estimate of the population distribution of dataset draws or training seeds. A
recipe-general claim requires a prospective power/precision simulation and the
resulting upper-level replication, never fewer than four independent data
draws by three training seeds per arm. If that is unaffordable or still
underpowered, keep the conclusion conditional on the realized policies.

### 2.2 Reasoning difficulty branches

```text
mechanism lab
  Countdown -> modular sequences

quantitative reasoning
  GSM8K -> MATH-exact -> AIME transfer

expert multiple choice
  MMLU-Pro -> GPQA Diamond transfer

structured abstraction
  ARC-AGI-2
```

The branches share evidence contracts, not verifier semantics or promotion
outcomes.

## 3. Current evidence ledger

Validated repo-local bundles and raw artifacts establish what ran. For legacy
runs that predate the new bundle, timestamped reports plus their structured
inputs are imported historical evidence with explicit provenance. The
classification below does not reinterpret old work as stronger evidence.

| Task | Current maturity | What has actually run | Governing limitation / next decision |
| --- | --- | --- | --- |
| Countdown | Historical replicated anchor; M3+ under the new ladder | Clean-split Qwen3-1.7B SFT and fresh clean/formatting replications. The original clean-split evaluation reused earlier checkpoints and exports; later replication cells produced fresh scoped artifacts. | The principal runs couple split draw and training seed, so they do not form the new fully crossed `2 draws x 2 seeds` M4 matrix. Preserve the result as a reference, not proof of cross-task transfer. |
| Arithmetic words | M1, saturated control | Tiny fixture and Qwen3-1.7B rootfs/vLLM smokes reached 1.0 pass@1. | Plumbing and format regression only; do not train at evidence scale. |
| Modular sequences | M2 plus an exploratory raw-SFT signal | One expanded raw-LoRA run used 64 train, 32 dev, 32 held-out rows, eight rollouts, and 24 SFT steps. Dev pass@1 moved 0.531 -> 0.625 and the historically named `ood_test` split moved 0.375 -> 0.594. | It does not satisfy this plan's M3 arm matrix: one split draw, one training seed, one arm, and small base-elicitable subsets. The historical `ood_test` changed only the generator seed, not the parameter band, so it is held-out IID evidence rather than a distribution-shift result. |
| Local GSM fixture | M1, saturated control | Three dev and three OOD fixture rows reached 1.0 pass@1. | Verifier and formatting regression only. |
| GSM8K | M1 calibration | Eight dev and eight OOD rows, both sampled from the official test split, reached pass@1 0.625 and pass@4 0.750. | Opened test-split smoke; not training data, not a leaderboard result, and too small for SFT promotion. |
| MATH | M1 calibration | Eight dev and eight OOD algebra test rows were rescored with the corrected conservative normalizer. | Tiny opened test slice. Current verifier excludes symbolic equivalence, sets, intervals, matrices, and multi-answer semantics. |
| MMLU-Pro | M1 calibration | No-tool Qwen3-1.7B calibrations include a 16/16 validation slice with exact final-letter scoring. | Slice difficulty differs by offset, is not subject-stratified, and has no adapter or replication evidence. |
| GPQA Diamond | M0, invalid and rebuilding | Public-cache runs completed, but `import_gpqa_rows` always places the correct answer first and assigns target `A`. | Every score derived from those imported rows is scientifically invalid. Preserve the artifacts as bug evidence, exclude their metrics from comparisons, rebuild choices, and rescore. |
| AIME 2024 | M1 hard negative | Eight dev and eight OOD rows with eight rollouts produced one easy dev problem, no elicitable problems, and zero OOD success. | Current Qwen3-1.7B best-of-8 condition cannot yield task-specific SFT data. Use as opened development calibration and transfer target. |
| ARC-AGI-2 | M1 hard negative | Packing solved the selected 4096-token context blocker. Packed and packed-strict 8/8 calibrations still scored zero; failures moved from missing final grids to malformed grids. | Context fit is solved for the slice, exact-grid construction is not. No SFT until disjoint training tasks yield verified successes. |

The canonical historical evidence is described in:

- [`experiments/countdown_search_distill/reports/20260811T222454Z-clean-split-final-results-and-infra-report.md`](../countdown_search_distill/reports/20260811T222454Z-clean-split-final-results-and-infra-report.md)
- [`experiments/countdown_search_distill/reports/20260812T043000Z-replication-rank-size-sweep.md`](../countdown_search_distill/reports/20260812T043000Z-replication-rank-size-sweep.md)
- [`experiments/countdown_search_distill/reports/20260812T054500Z-formatting-replication.md`](../countdown_search_distill/reports/20260812T054500Z-formatting-replication.md)
- [`reports/20260812T073500Z-modular-transfer-expanded.md`](reports/20260812T073500Z-modular-transfer-expanded.md)
- [`reports/20260812Taime-8x8-rollouts8.md`](reports/20260812Taime-8x8-rollouts8.md)
- [`reports/20260812Tarc-packed-strict-chat-results.md`](reports/20260812Tarc-packed-strict-chat-results.md)

## 4. Shared scientific contract

### 4.1 Frozen campaign declaration

Before work starts, every campaign records:

- hypothesis ID and one primary estimand;
- task and benchmark release;
- model and tokenizer revisions;
- source, train, development, transfer, and final-evaluation registries;
- problem-identity and near-duplicate policy;
- prompt, renderer, scaffold, and verifier versions;
- arm definitions and target-construction versions;
- generation seeds, rollout budget, temperature, top-p, maximum tokens, and
  stop rules;
- SFT config, effective global batch, valid-target-token exposure, optimizer
  steps, precision, parallelism, and checkpoint policy;
- primary endpoint, guardrails, practical effect threshold, analysis method,
  and futility conditions;
- maximum candidates, target tokens, optimizer steps, GPU-hours, and permitted
  follow-up;
- rootfs profile, source revision, run ID, declaration digest, and Temporal
  workflow correlation when durable execution is enabled.

Changing a scientific field produces a new declaration and run ID. Retrying an
unchanged declaration produces a new attempt ID.

### 4.2 Problem identity and split policy

The problem, not a generated rollout, is the unit of split non-overlap and primary
analysis.

- Synthetic identities hash the normalized generator parameters and canonical
  answer, not only the PRNG seed.
- Public identities use the upstream task ID plus a content digest. If the
  upstream ID is missing or unstable, the normalized prompt and canonical
  answer form the stable identity.
- All examples belonging to one ARC task stay together. Individual ARC test
  inputs from one task may not be split across train and evaluation.
- Exact normalized-question collisions are rejected globally within a task
  family. Benchmark-specific near-duplicate checks are recorded separately;
  they do not silently rewrite upstream data.
- Candidate generation, target construction, SFT, development selection,
  transfer evaluation, and final confirmation each name their permitted
  registries. A row may not move between these roles after outcomes are seen.
- Data from opened GSM8K, MATH, AIME, MMLU-Pro, GPQA, or ARC calibration slices
  remains development evidence. It cannot be relabeled as a locked final set.

Every split registry records source revision, row-source artifact hash, problem
IDs, content hashes, exclusions, overlap results, and selection status.

### 4.3 Verifier contract

Each verifier has a versioned name, source digest, output contract, canonical
preflight suite, adversarial mutation suite, and documented non-equivalences.

Required checks are:

1. Every selected canonical solution succeeds.
2. Wrong answers, unavailable operands, malformed finals, truncated outputs,
   and task-specific invalid states fail for the right reason.
3. Task success and strict format compliance are stored independently.
4. Rescoring saved generations records a new verifier condition and report; it
   never overwrites the original result.
5. A verifier change invalidates comparisons unless both sides are rescored
   from identical saved generations or rerun under the new verifier.

For MATH-exact, the primary verifier accepts only predeclared exact numeric
forms. For multiple choice, scoring uses the answer after the registered choice
permutation. For ARC, the primary score is exact grid equality; serialization
success is a separate metric.

### 4.4 Standard policy and scaffold cells

The standard matrix is intentionally small:

| Cell | Meaning | Required comparison role |
| --- | --- | --- |
| `base_pass1` | Frozen base, first generated sample | Starting policy |
| `base_passK` | Frozen base, fixed best-of-K budget | Reachability and residual search |
| `base_scaffold` | Frozen base plus a deployable critique, repair, search, or serialization scaffold | Inference-time system value |
| `raw_sft` | Verified accepted completion with minimal transformation | Historical and causal anchor |
| `clean_sft` | Deterministically normalized verified reasoning target | Compression candidate |
| `format_control` | The same verified train-problem solution content as the raw anchor, with only the final envelope, signature, or serialization normalized; it adds no corrected reasoning steps | Semantic-versus-format transformation control |
| `champion_passK` | Promoted adapter under the same K and sampling contract | Retained reachability |
| `champion_rl` | Promoted adapter followed by deterministic RL | Incremental post-training value |

Task-specific variants must describe target content rather than relying on an
ambiguous arm name. Countdown's historical `formatting` arm contains parsed
operation lines and correct targets; it is therefore a formatting-oriented
reasoning target, not a pure format-only control.

The first M3 screen evaluates base, raw, clean, and format control. Only the
champion and raw anchor advance to training replication, but frozen base is
evaluated on every final cell. Champion minus base is the primary
policy-compression estimand. Champion minus raw is a secondary mechanism
estimand when those policies differ. Only the champion and one causal control
advance to RL or larger models.

### 4.5 Matched generation and SFT

- Evaluate all arms on identical problem IDs, prompt versions, sampling budgets,
  stop rules, and verifier versions.
- Use keyed per-problem generation seeds. Do not treat stochastic rollouts as
  independent evaluation units.
- Build SFT targets only from train-registry candidates that pass the task
  verifier. Preserve rejected candidates and rejection reasons for analysis.
- Deduplicate targets by problem identity and normalized target digest. Record
  whether multiple candidates were available and the deterministic selection
  rule.
- Match base checkpoint, tokenizer, LoRA target modules, optimizer, schedule,
  precision, sequence length, effective global batch, and optimizer steps
  across arms.
- Primary screens match unique training problem identities and example count.
  A secondary check for the selected arm matches valid assistant-target-token
  exposure because raw and clean traces differ in length.
- Record overlength and dropped-example censuses before training. A comparison
  is invalid if truncation changes the semantic answer or differs silently by
  arm.
- Preserve DCP checkpoint, exported PEFT adapter, and merged/full Hugging Face
  policy as distinct lineage nodes.

### 4.6 Metrics and statistical analysis

The primary SFT estimand is the paired problem-level change in pass@1 between
the promoted policy and base on the locked evaluation registry.

Required outputs are:

- pass@1, pass@2, pass@4, pass@8, pass@16, and pass@32 where the fixed rollout
  budget supports them;
- strict-format pass@k and valid-output rate;
- easy, elicitable, and unreached problem counts;
- base-elicitable subset pass@1 and pass@K;
- accepted train problems, accepted candidates, duplicate rate, overlength
  rate, and target-token exposure;
- per-task difficulty or subject strata fixed before evaluation;
- verifier failure counts and proportions;
- output-length and stop-quality distributions;
- paired deltas versus base and the raw anchor.

Analysis rules:

- Use the problem as the bootstrap cluster. Rollouts, prompt permutations, and
  repeated evaluations of one problem are nested observations, not independent
  samples.
- Report a two-sided 95% paired cluster-bootstrap interval for the primary
  delta. Use at least 10,000 deterministic bootstrap draws recorded by seed.
- Before freezing a generic practical gate such as `+0.05`, simulate power and
  interval precision from development-only estimates of paired discordance,
  benchmark size, and between-run variation. Freeze the required problem,
  draw, and seed counts. An underpowered result is `hold/inconclusive`, not a
  negative effect claim.
- For crossed replication, report every draw-by-training-seed cell, the mean
  within each data draw, and the across-draw mean. With the minimum `2 x 2`
  matrix, a hierarchical bootstrap is descriptive sensitivity conditional on
  four realized policies; it is not population-level draw/seed uncertainty.
  Recipe-general inference requires the prospectively sized upper-level design,
  never fewer than four data draws by three training seeds per arm, and a
  prespecified small-sample run-level method. Do not pool rollouts as rows.
- Choose the champion on development evidence only. Run each locked final set
  once on frozen base, selected champion, and raw anchor using identical problem
  and sampling keys.
- A failed final gate spends that holdout. Corrections use opened development
  data only. Retesting requires a new declaration and a newly frozen, untouched
  final registry.
- Treat secondary endpoints as mechanism and guardrail evidence. Do not search
  them for an alternative success after the primary endpoint fails.
- A multiple-choice permutation audit aggregates within problem first, then
  computes task accuracy. It may expose position sensitivity but may not select
  a favorable permutation as the reported primary condition.
- Never relabel the largest observed budget as a larger pass@k. For example,
  an eight-rollout evaluation reports through pass@8; a derived field that
  clamps pass@32 to eight samples is excluded from scientific tables.

### 4.7 Promotion gates

| Gate | Pass condition |
| --- | --- |
| M0 contract | No exact problem overlap; all canonical solutions pass; adversarial verifier fixtures pass; all provenance and runtime checks select; no invalid source semantics |
| M1 calibration | At least 256 synthetic evaluation problems, or the complete preregistered public set when smaller; nontrivial task-success and format failure audit; no infrastructure failures counted as model failures |
| M2 reachability | Base pass@1 below 0.80; at least 10% and at least 32 evaluation problems are base-elicitable at the fixed K; at least 128 distinct train problems yield verified targets under the fixed collection budget |
| M3 SFT pilot | Finite training; complete checkpoint/export lineage; selected arm reaches the development-only practical threshold chosen before the screen; pass@K drops by no more than 2 points; format and semantic deltas are separately explained |
| M4 replication | Base, champion, and raw run on a minimum two fresh data draws x two training seeds; champion-minus-base direction is positive in every draw mean; the prospectively powered practical threshold and paired fixed-policy interval pass; no declared OOD or pass@K guardrail fails; claim scope is conditional unless the upper-level design was sized for recipe-general inference |
| M5 deterministic RL | Task has exact recomputable reward, sufficient reward variance, verified trainer/generator policy bridge, bitwise parity configuration, and M4 SFT evidence |
| M6 async RL | Deterministic RL replicated; consume-time policy age stays within the declared bound; quality is non-inferior under the declared margin; separated timing shows a meaningful throughput or utilization benefit |
| M7 scale | Smaller-model M4 or M5 gate passed; only champion/control pair runs; at least three fresh scale-level training seeds per arm or a larger prospectively sized design; paired final-set uncertainty, direction, retention, lineage, and reliability reproduce. One pair is a scale probe only. |

For naturally small locked public sets, the declaration may use all available
problems instead of 256. It must then report exact problem-level outcomes and
wide uncertainty rather than claim that sample size is representative.

### 4.8 Futility and stop rules

Stop task-specific advancement when any of these conditions holds:

- **Saturated:** base pass@1 is at least 0.80 and less than 0.10 absolute
  headroom remains at the fixed K. Retain the task as plumbing only.
- **Unreachable:** fewer than 10% or fewer than 32 evaluation problems are
  elicitable, or fewer than 128 distinct training problems yield verified
  targets at the maximum preregistered collection budget. Keep it as a transfer
  or scale-calibration target; do not train on its locked evaluation set.
- **Verifier failure:** any canonical solution fails, choice semantics are
  biased or ambiguous, or more than the preregistered zero tolerance of split
  overlap is observed. Return to M0.
- **No pilot effect:** no arm reaches the declared M3 practical threshold, or all arms lose
  more than 2 points at pass@K. Diagnose and correct using development evidence
  only. The opened final set is never rerun; a second confirmatory attempt needs
  a new declaration and a newly frozen untouched final registry. Otherwise stop
  task-specific SFT.
- **No replication:** either independent data draw has a nonpositive mean
  pass@1 delta, the declared paired fixed-policy interval includes zero, or an
  OOD guardrail fails. Do not promote to RL or larger models. When prospective
  sizing shows inadequate power, record `hold/inconclusive` rather than
  `reject/no effect`.
- **No reward signal:** deterministic RL produces zero reward variance or
  cannot reach full trainer/generator parity. Stop before asynchronous RL.
- **No scale case:** Qwen3-1.7B causal evidence fails. Do not spend a Qwen3-8B
  confirmation budget on that branch.

Infrastructure blockers, real zero scores, and score regressions remain
different outcomes. A regression is a valid measurement with a rejected
promotion decision; it is not a no-score blocker.

### 4.9 Failure and representative-example audit

Every evaluation writes counts and deterministic examples for:

- first-sample win over base;
- first-sample regression from base;
- base-elicitable compression success;
- unchanged semantic failure;
- strict-format-only failure;
- invalid or truncated output;
- verifier or environment failure;
- task-specific failure classes.

Within each category, choose examples by the lowest stable problem-identity
hash after applying the category rule. This prevents manual cherry-picking.
Reports include the prompt condition, expected answer, first relevant base and
policy output, verifier trace, and artifact reference. Evaluation examples are
published only after the corresponding split is opened.

## 5. Model and compute ladder

### 5.1 Model roles

| Model | Reasoning-program role |
| --- | --- |
| Qwen3-0.6B | Contract, checkpoint, export, Temporal recovery, and deterministic-RL plumbing. It is not the primary quality-evidence model. |
| Qwen3-1.7B | Primary causal SFT and deterministic-RL evidence model. |
| Qwen3-8B | Selected scale probe after upstream 8B foundation promotion. M7 also requires a 1.7B M4/M5 result and replicated 8B training seeds. |
| Qwen3-30B-A3B | Optional separate MoE probe or replicated campaign only after upstream MoE promotion, resource review, and a selected smaller-model result. |

Qwen3-8B promotion depends on the core training foundation's certification for
configuration, deterministic numerics, checkpoint resume, observability,
convergence, and matched performance at the topology actually used. Qwen3
30B-A3B MoE first belongs to the foundation research vehicle and then requires
a separate reasoning campaign; this plan does not spend it as a routine
benchmark model.

### 5.2 Compute tiers

1. **CPU/static:** importer, identity, split, verifier, report, workflow replay,
   and fixture tests.
2. **One GPU:** representative base calibration, candidate-yield measurement,
   small LoRA plumbing, adapter export/load, and Qwen3-0.6B recovery checks.
3. **Sequential eight GPU:** promoted Qwen3-1.7B SFT evidence, deterministic RL,
   and matched performance runs. Only one routine eight-GPU campaign runs at a
   time.
4. **Selected scale:** begin with one champion/control probe at certified
   Qwen3-8B. M7 requires the prospectively sized seed design, never fewer than
   three fresh training seeds per arm, with paired final-set uncertainty.
   Qwen3-30B-A3B requires a separate MoE proposal. No broad model x arm sweep
   and no routine multi-node execution.

## 6. Branch A: Countdown and modular mechanism lab

### 6.1 Purpose

Countdown establishes that verified search traces can be compressed on one
exact arithmetic task. Modular sequences tests whether the effect survives a
different generator, recurrence semantics, and OOD parameter band while
retaining deterministic verification.

### 6.2 Countdown anchor work

- [ ] **R-COUNT-01: Import the historical evidence into the immutable
  run-attempt contract.** Mark the initial clean-split training/export stages as
  `work_status=imported` with their evidenced freshness, the clean and
  formatting replication stages as `work_status=produced` with
  `freshness=verified_new`, and retain the original report files as immutable
  interpretations linked to the imported structured evidence.
- [ ] **R-COUNT-02: Requalify the selected formatting-oriented champion.** Run
  one fresh Qwen3-1.7B selected-arm/control evaluation on the canonical clean
  registries under the new rootfs profile and evidence schema. This verifies
  migration, not a third replication claim.
- [ ] **R-COUNT-03: Run a stop-quality ablation.** Compare the replicated
  formatting-oriented target with a shortened target and explicit stop rule on
  identical training problems. Primary endpoint is strict pass@1; guardrails
  are arithmetic pass@1/pass@32 and post-answer repetition rate.
- [ ] **R-COUNT-04: Decide whether a new Countdown M4 claim is needed.** If
  Countdown itself will support an M4 or RL claim under this program, fill the
  missing draw-by-seed cross cells for base, champion, and raw anchor. Do not
  run those cells merely to relabel useful historical evidence, and do not make
  a recipe-general claim from the minimum matrix.
- [ ] **R-COUNT-05: Freeze Countdown as a retention suite.** Every promoted
  reasoning champion is evaluated on a fixed, disjoint Countdown subset to
  detect catastrophic loss of exact-trace and output-contract behavior.

No new Countdown rank sweep is authorized unless R-COUNT-03 identifies a
mechanism that cannot be tested at the selected rank-16 configuration.

### 6.3 Modular replication matrix

The current expanded run is the exploratory anchor. The minimum conditional
confirmation campaign uses two fresh generator draws and training seeds 42 and
43 per promoted arm. Selection and confirmation data are disjoint within every
draw.

| Cell | Data draw | Training seed | Policy |
| --- | --- | --- | --- |
| A | draw 1 | 42 | base, raw, clean, format control on `dev_select`; base/champion/raw on unopened `test_confirm` after selection freezes |
| B | draw 1 | 43 | base, champion, and raw anchor on `test_confirm` |
| C | draw 2 | 42 | base, champion, and raw anchor on `test_confirm` |
| D | draw 2 | 43 | base, champion, and raw anchor on `test_confirm` |

Each candidate draw contains 512 train, 256 IID `dev_select`, 256 IID
`test_confirm`, 128 opened length-shift development, 256 locked length-shift
confirmation, 128 opened modulus-shift development, and 256 locked
modulus-shift confirmation identities. Draw 1 uses generator seeds 5100, 5200,
5250, 5300, 5350, 5400, and 5450 in that order; draw 2 uses 6100, 6200, 6250,
6300, 6350, 6400, and 6450. Train and IID splits use 3-5 recurrence steps and
modulus 37-257. Length-shift splits use 6-7 steps and modulus 37-257.
Modulus-shift splits use 3-5 steps and modulus 263-521. Exact problem-key
exclusion applies across all draws and splits. A prospective power simulation
may increase these sizes or the number of draw/seed cells before registry
digests freeze; it may never shrink or expand them after final outcomes open.

The fixed collection and evaluation budget is K=8, temperature 0.8, top-p
0.95, and 512 maximum generated tokens. Scientific tables report only
pass@1/pass@2/pass@4/pass@8. Any compatibility summary field named pass@16 or
pass@32 is ignored because only eight candidates were sampled.

Experiments:

- [ ] **R-MOD-01: Contract requalification.** Validate generator determinism,
  recurrence execution, strict-final parsing, split identities, and canonical
  solutions. Add arithmetic-slip, unavailable-state, missing-final, extra-final,
  and truncation fixtures.
- [ ] **R-MOD-02: Representative base calibration.** Evaluate Qwen3-1.7B at
  pass@1/pass@8 on the opened `dev_select` and development-shift registries.
  Freeze the calibrated band only if M2 reachability passes on IID; retain each
  locked OOD confirmation axis as an explicit guardrail even when development
  shows it is below the training-yield band.
- [ ] **R-MOD-03: Matched arm screen.** On draw 1/seed 42 (cell A), compare raw
  verified transcripts, deterministic clean recurrence traces, and format-control
  examples. Match unique train problems and optimizer steps; run the
  target-token-matched secondary check for the selected arm.
- [ ] **R-MOD-04: Crossed replication.** After R-MOD-03 freezes the champion,
  run base, champion, and raw anchor on every cell's still-unopened
  `test_confirm` and locked OOD registries. Apply the M4 cellwise paired
  analysis without counting the old expanded run as a confirmatory cell. Label
  the `2 x 2` result conditional unless prospective sizing expanded the
  upper-level design for recipe-general inference.
- [ ] **R-MOD-05: Retention and transfer.** Evaluate the modular champion on
  Countdown retention, GSM8K development, and MATH-exact development without
  task-specific updates.
- [ ] **R-MOD-06: Deterministic RL factorial.** If R-MOD-04 passes, run base,
  SFT, base-to-RL, and SFT-to-RL with exact recurrence reward under the M5
  parity contract. Use Qwen3-0.6B for end-to-end plumbing before Qwen3-1.7B.

Promotion from this branch requires semantic wrong-answer reductions as well as
format improvement. A policy whose pass@1 gain disappears after conditioning
on strict-format-valid outputs does not establish reasoning compression.

## 7. Branch B: GSM8K and MATH-exact quantitative reasoning

### 7.1 Data partition contract

For GSM8K:

- Pin the upstream dataset revision already recorded by the importer.
- Use only the upstream training split for candidate generation and SFT.
- Before generation, make a deterministic content-hash partition of upstream
  training rows into source-train and locked development registries.
- Mark the already opened 8+8 official-test rows as development calibration.
  They never enter training and are excluded from the final confirmation
  estimand.
- Freeze the remaining permitted official-test identities as one-time
  confirmation before champion selection.

For MATH:

- Define `MATH-exact-v1` before generation. It includes signed integers,
  rational numbers, and finite decimals that normalize exactly to a rational
  value. It excludes symbolic expressions, equations, sets, intervals,
  matrices, multiple answers, and unit-bearing answers from the primary metric.
- Record every exclusion category and count; do not call `MATH-exact-v1` a full
  MATH score.
- Use upstream training rows for source-train and development registries and
  preserve the permitted test rows as one-time confirmation.
- Keep subject and difficulty strata in every registry and analysis.
- Run a secondary exact-math scorer or blinded deterministic adjudication on
  every primary-verifier disagreement before broadening the accepted answer
  forms. Generic LLM judging is not permitted.

### 7.2 GSM8K campaign

- [ ] **R-GSM-01: Verifier and registry certification.** Run canonical and
  adversarial fixtures for integers, negatives, commas, currency markers,
  fractions, decimals, boxed answers, multiple final markers, and malformed
  answers. Freeze source-train, development, opened-calibration, and final
  registries.
- [ ] **R-GSM-02: Representative Qwen3-1.7B calibration.** Report pass@1 and
  pass@K by answer magnitude and operation-depth proxy. Separate mathematical
  error, extraction error, missing final, and truncation.
- [ ] **R-GSM-03: Frozen best-of-K collection.** Generate only on the
  source-train registry. Require at least 128 distinct verified problem
  identities and preserve all rejected candidates.
- [ ] **R-GSM-04: Matched SFT pilot.** Compare raw, clean, and format control on
  identical source-train IDs. The clean target keeps a concise verified
  solution plus exact final answer; the format control retains the selected raw
  reasoning verbatim and changes only final-answer serialization.
- [ ] **R-GSM-05: Replication.** Use a minimum two independent source-train
  draws x two training seeds. Select on `dev_select`, then evaluate base,
  champion, and raw on the remaining official-test confirmation once. Scope the
  claim to the realized policies unless prospective sizing authorized the
  larger recipe-general matrix.
- [ ] **R-GSM-06: Transfer probes.** Evaluate Countdown and modular champions
  on GSM8K development, then evaluate the GSM champion on modular and MATH-exact
  development. These are frozen-policy tests.
- [ ] **R-GSM-07: Deterministic RL eligibility.** If R-GSM-05 passes, validate
  exact numeric reward, reward variance, and policy bridge. Run deterministic
  RL before any asynchronous condition.

### 7.3 MATH-exact campaign

- [ ] **R-MATH-01: Build and certify `MATH-exact-v1`.** Publish included and
  excluded answer-form counts, canonical success, mutation failures, and a
  disagreement audit. Preserve the current 8+8 opened algebra rows as smoke
  evidence only.
- [ ] **R-MATH-02: Stratified calibration.** Evaluate Qwen3-1.7B on a fixed
  subject x difficulty development matrix with enough rows to expose each
  reported stratum. Report exact-answer and strict-format success separately.
- [ ] **R-MATH-03: Scaffold comparison.** Compare frozen best-of-K, bounded
  critique/revision, and direct generation on the same development problems.
  The critique path receives the problem and prior candidate but never the
  verifier answer or held-out feedback.
- [ ] **R-MATH-04: Matched SFT pilot.** Collect verified source-train targets
  under the selected scaffold and compare raw, clean, and format-control arms.
  Run both example-matched and valid-target-token-matched analyses.
- [ ] **R-MATH-05: Replication and confirmation.** Run base, champion, and raw
  on a minimum two source-train draws x two training seeds. Select on
  development and open the locked `MATH-exact-v1` confirmation once. Scope the
  claim to the realized policies unless the prospectively sized design adds
  upper-level replication.
- [ ] **R-MATH-06: Bidirectional transfer.** Evaluate the GSM champion on
  MATH-exact and the MATH champion on GSM8K, modular, and Countdown retention.
- [ ] **R-MATH-07: AIME handoff.** Only a replicated MATH champion may enter the
  AIME transfer branch or justify a selected certified-Qwen3-8B AIME
  reachability calibration.

The quantitative branch promotes only if mathematical-error reductions remain
after filtering to outputs that satisfy the format contract.

## 8. Branch C: MMLU-Pro and GPQA expert multiple choice

### 8.1 Shared multiple-choice repair

The current GPQA importer has a fundamental semantic defect:

```text
choices = (correct, incorrect_1, incorrect_2, incorrect_3)
answer = A
```

This makes answer position a label leak. The repair is an M0 blocker, not a
prompt ablation.

- [ ] **R-MC-01: Publish an invalidation audit.** Enumerate every GPQA-derived
  problem artifact, summary, report input, and timestamped report produced by
  the correct-first importer. Mark measurement validity `invalid`, preserve
  files, and exclude their metrics from latest valid result selection.
- [ ] **R-MC-02: Add deterministic choice permutation.** Version the assignment
  algorithm and group rows by frozen `(split, subject_stratum, num_choices)`.
  Within a group, sort by a cryptographic digest of
  `("row-order", algorithm_version, permutation_seed, stable_problem_id)`.
  Derive an independent distractor permutation from a digest namespaced
  `"distractors"`, then insert the correct answer at the position assigned by
  R-MC-03. Never use Python's process-randomized hash. Preserve source order and
  correct-answer text as provenance, then compute the answer letter after
  insertion.
- [ ] **R-MC-03: Balance answer positions.** For each frozen group, derive one
  outcome-blind starting offset from a digest of
  `("position-offset", algorithm_version, permutation_seed, split,
  subject_stratum, num_choices)`. Assign sorted row ordinal `i` to
  `(offset + i) mod num_choices`. This cycles positions so incomplete groups
  differ by at most one count, handles varying choice counts independently, and
  uses neither answer text nor model outcomes. Record counts by split, subject,
  and choice count, plus the algorithm version and seed.
- [ ] **R-MC-04: Add a permutation-robustness audit.** For a problem with `m`
  choices, apply all `m` registered cyclic Latin-square rotations of the same
  choices. GPQA therefore has four rotations and MMLU-Pro rows with ten choices
  have ten. Aggregate within problem first. Report primary balanced accuracy,
  worst-rotation accuracy, answer-position accuracy, and disagreement rate
  without selecting the best rotation.
- [ ] **R-MC-05: Rebuild and rescore.** Raw GPQA caches containing correct and
  incorrect answer columns may be reused after hash validation. All generated
  problem JSONL, prompts, generations, summaries, and reports under the old
  answer mapping are invalid and must be regenerated.

The same permutation implementation applies to MMLU-Pro as an explicitly named
condition. Its existing native-order calibration remains limited M1 evidence,
but it cannot satisfy the new position-robust promotion gate by itself.

### 8.2 MMLU-Pro campaign

Validation rows at offsets `0-15` and `32-47` were already opened by the
16/16 calibration (and the smaller slices are contained within those ranges).
Register those identities as development evidence and exclude them from any new
locked confirmation registry. Build the subject-stratified locked registry only
from untouched identities before evaluating a new policy.

- [ ] **R-MMLU-01: Contract certification.** Freeze a subject-stratified
  registry, validate all answer mappings before and after permutation, and run
  the full choice-count rotation audit.
- [ ] **R-MMLU-02: Representative base calibration.** Evaluate Qwen3-1.7B on
  the fixed subject matrix. Report macro subject accuracy, micro accuracy,
  strict format, answer-position sensitivity, and per-subject uncertainty.
- [ ] **R-MMLU-03: Scaffold reachability.** Compare direct generation with one
  bounded critique/revision pass. The revision prompt receives no correct
  answer or verifier feedback.
- [ ] **R-MMLU-04: Determine training eligibility.** Task-specific SFT may use
  only a pinned upstream source explicitly permitted for training. If the
  registered release supplies no suitable training source, MMLU-Pro remains an
  evaluation-only transfer benchmark and this task-specific SFT branch closes.
- [ ] **R-MMLU-05: Matched SFT when eligible.** On a disjoint authorized
  training registry, compare raw, clean concise rationale, and format control;
  replicate base/champion/raw across a minimum two source draws x two training
  seeds, with the same conditional-versus-recipe-general distinction as M4.
- [ ] **R-MMLU-06: GPQA transfer handoff.** Evaluate the promoted MMLU-Pro,
  MATH, and base policies on the rebuilt GPQA registry without target-specific
  updates.

### 8.3 GPQA Diamond campaign

GPQA Diamond is an opened, post-hoc transfer benchmark first. Diamond questions
and model results were already inspected under the defective answer mapping;
deterministic shuffling repairs measurement validity but does not restore
blindness. Its small evaluation set and expert explanations are not
task-specific SFT data in this plan. No Diamond result may be called a locked
confirmation. A genuinely confirmatory multiple-choice claim needs a separate
untouched benchmark or authorized split frozen before policy selection.

- [ ] **R-GPQA-01: Complete R-MC-01 through R-MC-05.** No GPQA score is valid
  before this gate.
- [ ] **R-GPQA-02: Freeze evaluation semantics.** Pin the authorized raw cache,
  content hashes, primary balanced permutation, four-rotation audit, no-tool
  prompt, exact final-letter verifier, and complete problem registry.
- [ ] **R-GPQA-03: Recalibrate base.** Evaluate Qwen3-1.7B and report exact
  problem-level results. Label the public-cache condition accurately; do not
  call it an official leaderboard score unless the official harness contract is
  independently satisfied.
- [ ] **R-GPQA-04: Post-hoc frozen-policy transfer.** Compare base, MMLU-Pro
  champion, MATH champion, and at most one combined reasoning champion under a
  policy list frozen before the corrected rerun. Label the analysis post-hoc
  because prior Diamond content/outcomes were inspected, and do not train or
  tune on Diamond failures.
- [ ] **R-GPQA-05: Position-bias diagnosis.** Report whether policy rankings
  persist across rotations. A gain driven by one answer position fails
  promotion even if primary balanced accuracy rises.

A GPQA-specific SFT campaign requires a separate, authorized, preregistered
training corpus and a new evaluation registry. Diamond rows, explanations,
model failures, and permutation outcomes may not enter that training corpus.

## 9. Branch D: AIME contest-math transfer

The existing Qwen3-1.7B best-of-8 condition has one easy development problem,
zero elicitable problems, and zero OOD successes. It fails M2 reachability for
task-specific SFT.

- [ ] **R-AIME-01: Freeze opened and locked identities by year and exam.** Mark
  every previously evaluated AIME 2024 problem as opened development evidence.
  Choose the permitted unopened confirmation set before any new policy result
  is inspected. Record year, exam, problem number, source hash, and answer.
- [ ] **R-AIME-02: Exact verifier certification.** Require integer-range,
  leading-zero, multiple-final, missing-final, and truncation fixtures. Report
  each problem, not only aggregate pass@k.
- [ ] **R-AIME-03: Frozen transfer.** Evaluate base, replicated GSM champion,
  and replicated MATH champion on opened development problems. No AIME outcome
  enters training or policy selection.
- [ ] **R-AIME-04: Critique/revision scaffold.** On development only, compare
  direct best-of-K with one frozen critique/revision pass. The critic sees the
  problem and candidate, not the canonical answer or verifier result.
- [ ] **R-AIME-05: Model reachability escalation.** Run Qwen3-8B only after its
  upstream foundation promotion, the Qwen3-1.7B MATH branch reaches M4, and
  AIME remains below M2. This is a selected reachability test, not a way to
  replace a failed 1.7B causal gate.
- [ ] **R-AIME-06: One-time confirmation.** Open the locked set only for base
  and the preselected champion. Report exact successes, pass@1/pass@K, and
  problem-cluster uncertainty. A small-set result remains a transfer result,
  not an AIME training claim.

No task-specific AIME SFT or RL is authorized without a separate training
corpus proven disjoint from all evaluation years and problems.

## 10. Branch E: ARC-AGI-2 structured abstraction

### 10.1 Contract and data policy

- Preserve exact upstream task IDs and keep all train/test examples from one
  ARC task together.
- Public training tasks may support scaffold development and verified SFT only
  after task-level train/development non-overlap is frozen.
- Locked evaluation tasks never appear in prompts, repair feedback, reward
  construction, or training targets.
- Primary task success is exact output-grid equality. Also report valid-grid
  construction, dimensions, palette validity, JSON serialization, and
  truncation separately.
- A deterministic formatter may serialize an already predicted grid. It is a
  deployable scaffold and may not be credited with inferring the
  transformation.

### 10.2 Experiments

- [ ] **R-ARC-01: Serialization certification.** Add exact round-trip tests for
  JSON and packed-row grid forms, adversarial malformed shapes, invalid colors,
  ragged rows, empty grids, and truncation. Run canonical solutions through the
  same prompt packer and verifier.
- [ ] **R-ARC-02: Representative base calibration.** Expand beyond the current
  opened 8+8 slice using a fixed task-level development registry. Compare the
  existing packed and packed-strict conditions only to characterize failure;
  do not choose a prompt from final-evaluation outcomes.
- [ ] **R-ARC-03: Two-stage format scaffold.** Stage one predicts packed rows.
  A deterministic validator either converts a valid packed grid to JSON or
  rejects it without semantic repair. Compare direct exact-grid success,
  valid-packed-grid rate, serializer success, and final exact-grid success.
- [ ] **R-ARC-04: Rule-induction scaffold.** On public training tasks only,
  compare direct prediction with a bounded rule/program proposal followed by
  deterministic execution. Record whether success came from model prediction,
  program execution, or serialization.
- [ ] **R-ARC-05: Reachability decision.** Require M2 coverage on disjoint
  training tasks and at least 128 verified task identities before SFT. If the
  gate fails, keep ARC as a hard-negative transfer benchmark and stop
  task-specific training.
- [ ] **R-ARC-06: Matched SFT when eligible.** Compare raw verified model
  traces/programs, clean canonical programs or grids, and format control.
  Replicate base/champion/raw across a minimum two task draws x two seeds.
  Never split examples from one ARC task across draws, and do not make a
  task-population claim without prospectively sized independent task draws.
- [ ] **R-ARC-07: Frozen transfer and scale.** Evaluate promoted quantitative
  and modular champions before any ARC-specific training. Escalate to certified
  Qwen3-8B only under the selected-scale gate.

ARC deterministic RL is deferred until R-ARC-06 reaches M4 and reward can be
recomputed from exact grid equality without revealing locked task outputs to
the rollout policy.

## 11. SFT and RL eligibility by branch

| Branch | SFT status | Deterministic RL status | Async RL status |
| --- | --- | --- | --- |
| Countdown | Replicated historical anchor; selected stop-quality ablation allowed | Eligible only after optional R-COUNT-04, policy bridge, and parity | Only after deterministic causal value |
| Modular | Eligible for immediate replication | First preferred reasoning RL task after M4 | Only after replicated deterministic RL |
| GSM8K | Eligible after train/dev/final registry repair and M2 | Eligible after M4 exact numeric reward | Deferred |
| MATH-exact | Eligible after verifier subset and M2 | Eligible after M4 and exact reward audit | Deferred |
| MMLU-Pro | Conditional on an authorized training source | Conditional on M4 and exact answer reward; lower priority than exact numeric tasks | Deferred |
| GPQA Diamond | Evaluation-only | Not eligible | Not eligible |
| AIME | Transfer-only under current data/reachability | Not eligible | Not eligible |
| ARC-AGI-2 | Conditional on task-level M2 reachability | Conditional on M4 exact-grid reward | Not eligible until deterministic recovery and reward evidence |

Every first RL campaign uses:

```text
base
SFT
base -> deterministic RL
SFT -> deterministic RL
```

The deterministic configuration requires strict on-policy execution,
`target_offpolicy_steps=0`, `window_fraction=None`, matching trainer and
generator seeds and TP degrees, deterministic and batch-invariant modes,
bf16 forward/generation with fp32 master weights, sequence parallelism off,
and compatible prefix-cache reset. Production-style async RL is a separate
quality/throughput experiment and never supplies parity evidence.

## 12. Durable Temporal campaign mapping

Temporal is the durable outer control plane once the execution foundation is
available. It runs outside the bwrap rootfs and calls coarse rootfs activities.
The local run-attempt bundle remains canonical.

Workflow determinism, Activity idempotence, timeout, heartbeat, cancellation,
and Continue-As-New rationale are documented in
[`temporal_durable_execution_research.md`](temporal_durable_execution_research.md).

Local durable mode uses a version-pinned Temporal CLI and Python SDK, a local
development server started with an explicit persistent database filename, and
a version-pinned worker outside the rootfs. Server history supports durable
control flow but is not a substitute for repo-local artifacts. Loss of the
Temporal database may require a new orchestration attempt; it must not erase or
rewrite completed run-attempt evidence.

### 12.1 Workflow hierarchy

```text
ResearchCampaignWorkflow
  -> ExperimentRunWorkflow(contract)
  -> ExperimentRunWorkflow(calibration cells)
  -> machine-computed gate + explicit promotion approval
  -> ExperimentRunWorkflow(collection shards)
  -> ExperimentRunWorkflow(SFT arms)
  -> ExperimentRunWorkflow(export and evaluation)
  -> ExperimentRunWorkflow(replication cells)
  -> machine-computed gate + explicit promotion approval
  -> ExperimentRunWorkflow(deterministic RL, when eligible)
```

Campaign phase boundaries use Continue-As-New when history grows, while the
scientific run IDs and artifact lineage remain stable. Temporal workflow and
workflow-run IDs are correlation fields, not replacements for run ID or
attempt ID.

### 12.2 Activity boundaries

Use activities for:

- dataset acquisition/import plus provenance publication;
- split generation and validation;
- canonical/verifier preflight;
- one resumable rollout-shard collection;
- deterministic aggregation and target materialization for one arm;
- one complete TorchTitan SFT job;
- one PEFT or full-policy export and validation;
- one model/arm/split evaluation bundle;
- one report-input validation and analysis publication;
- one complete deterministic or async RL job after its eligibility gate.

Do not create activities for individual prompts, batches, optimizer steps, or
RL rollouts.

### 12.3 Queues, heartbeats, and retry safety

- CPU import, validation, analysis, and report work uses the `cpu` queue.
- Calibration and collection use `gpu1` unless the declared model requires the
  selected-scale queue.
- Promoted SFT and RL evidence uses sequential `gpu8` admission.
- Activities heartbeat stage, shard count or optimizer step, last committed
  artifact/checkpoint, subprocess identity, and last progress time.
- Generation commits an append-only shard ledger. SFT resumes only from a
  validated complete DCP checkpoint. Evaluation publishes invocation-specific
  files atomically before registering them.
- Stable logical idempotency keys derive from run ID, stage ID, and declaration
  digest. An attempt ID spans one top-level run execution; each real stage
  launch or relaunch receives a stage invocation ID. A redelivery that only
  validates a receipt creates neither. Duplicate delivery validates and returns
  a completed receipt, observes a durably supervised job, resumes from validated
  state, or stops for operator review. It never blindly launches duplicate GPU
  work.
- Invalid config, split overlap, verifier disagreement, corrupt checkpoint,
  declared-setting OOM, and ambiguous live-process state are non-retryable.
- Automatic RL activity retry remains disabled until joint controller, work
  buffer, dataset, optimizer, and policy-version recovery is proven.

Until Temporal integration passes its restart and duplicate-delivery tests,
the same frozen declarations run through the direct lifecycle executor. Direct
mode does not weaken evidence requirements.

## 13. Dependency-ordered execution waves

### Wave 0: Repair truth before new scores

- [ ] Complete the GPQA invalidation audit and remove invalid measurements from
  valid-result selection.
- [ ] Implement and certify deterministic balanced multiple-choice
  permutations.
- [ ] Freeze opened-versus-locked registries for GSM8K, MATH, AIME, MMLU-Pro,
  GPQA, and ARC.
- [ ] Certify shared task-success/strict-format and blocker/regression status
  semantics.

**Exit:** Every task is honestly classified at M0 or M1 and no known invalid
score can drive promotion.

### Wave 1: Reproduce the mechanism foundation

- [ ] Import Countdown evidence and requalify the selected anchor under the new
  lifecycle.
- [ ] Run modular R-MOD-01 through R-MOD-04.
- [ ] Publish the crossed replication report and machine-computed M4 decision.

**Exit:** Modular either becomes the second replicated exact-verifier task or
closes as a one-draw positive that did not replicate.

### Wave 2: Quantitative public reasoning

- [ ] Complete GSM8K registry repair, arm screen, replication, and one-time
  confirmation.
- [ ] Build `MATH-exact-v1`, run verifier adjudication, arm screen,
  replication, and one-time confirmation.
- [ ] Publish bidirectional transfer and retention results.

**Exit:** At least one public quantitative task reaches M4, or both branches
close with interpretable futility evidence.

### Wave 3: Expert multiple choice

- [ ] Run subject-stratified, permutation-robust MMLU-Pro calibration.
- [ ] Determine MMLU-Pro training eligibility from pinned source semantics.
- [ ] Rebuild GPQA and run frozen-policy transfer only.

**Exit:** GPQA has a valid balanced evaluation and no correct-first artifact is
cited as evidence.

### Wave 4: Hard transfer and structured abstraction

- [ ] Run AIME transfer from the selected MATH policy before model escalation.
- [ ] Run ARC serialization and rule-induction scaffolds before SFT.
- [ ] Apply M2 futility honestly; retain unreachable tasks as hard negatives.

**Exit:** Each branch either has verified training reachability or an explicit
transfer-only status.

### Wave 5: Deterministic post-training

- [ ] Bridge the selected SFT artifact to a full policy with trainer/vLLM
  loading and logprob equivalence.
- [ ] Run Qwen3-0.6B parity and recovery plumbing.
- [ ] Run Qwen3-1.7B modular factorial first, followed by at most one promoted
  public exact-verifier task.
- [ ] Replicate causal RL value before opening async work.

**Exit:** The program can distinguish base improvement, SFT improvement,
ordinary RL improvement, and an RL advantage caused by scaffold-distilled
initialization.

### Wave 6: Bounded async and scale confirmation

- [ ] Run strict FIFO on-policy, then bounded off-policy FIFO, then a controlled
  look-ahead window only if each prior condition passes.
- [ ] Require matched quality, bounded consume-time policy age, born-fresh slot
  release, and separated throughput timings.
- [ ] Run only the selected arm/control pair at certified Qwen3-8B. Call one
  pair a scale probe; claim M7 only after the prospectively sized design with
  at least three fresh training seeds per arm and paired final-set uncertainty.
  Treat any Qwen3-30B-A3B work as a separate MoE campaign.

**Exit:** Any async or M7 scale claim has a smaller-model deterministic causal
anchor, valid uncertainty at its claim level, and complete checkpoint/policy
lineage.

## 14. Required deliverables per promoted campaign

Each campaign closes with:

1. frozen declaration and split registries;
2. dataset/source provenance and raw-cache hashes;
3. verifier specification, canonical preflight, and adversarial fixtures;
4. rootfs/runtime doctor artifact;
5. append-only run-attempt events and stage logs;
6. rollout shard ledger and accepted/rejected candidate manifest;
7. per-arm target dataset statistics and target-token census;
8. TorchTitan resolved config, metrics, DCP checkpoint lineage, and resume
   evidence;
9. PEFT/full-policy export validation and vLLM load evidence;
10. per-split raw generations, evaluation rows, and summaries;
11. paired fixed-policy analysis, cellwise run effects, and only when
    prospectively justified, upper-level recipe-general analysis;
12. failure taxonomy and deterministic representative examples;
13. canonical `report_input.json` with execution, measurement, and promotion
    states kept separate;
14. immutable timestamped Markdown report;
15. machine-computed promotion or futility decision;
16. Temporal workflow references when durable mode was used.

## 15. Program completion criteria

This reasoning program is complete when one of these two honest outcomes is
reached:

### Positive closure

- At least two task families, including one public or public-distribution task,
  show M4 replicated first-sample improvement under clean split and verifier
  contracts.
- At least one cross-task frozen-policy transfer is measured on a locked target.
- At least one exact-verifier task completes the deterministic
  base/SFT/base-to-RL/SFT-to-RL factorial.
- Async and scale claims, if made, pass their separate quality, policy-age,
  throughput, and lineage gates.

### Negative but useful closure

- Every investigated branch has a valid contract, calibrated reachability, and
  a machine-readable reason for saturation, unreachable coverage, no SFT
  effect, failed replication, or no RL reward signal.
- Invalid historical evidence remains preserved but cannot influence valid
  latest-result or promotion selection.
- Failure and representative-example evidence is sufficient to explain where
  scaffold compression stopped working.

Both closures produce a useful research result. Continuing to spend compute
after a preregistered futility gate is not progress.
