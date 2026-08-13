# Scaffold-To-Policy Training Research Plan

Status: approved design, implementation and execution pending unless a cited
timestamped report says otherwise.

This document defines the long-horizon training program for compressing
verified scaffold behavior into policy behavior. It covers offline dataset
construction, TorchTitan supervised fine-tuning (SFT), checkpoint and export
proof, the full-policy bridge into online reinforcement learning (RL), strict
deterministic RL, and later asynchronous RL.

The master dependency graph and campaign state live in
`followup_experiment_plan.md`. The benchmark-specific hypotheses live in
`reasoning_research_plan.md` and `coding_research_plan.md`. Runtime, bwrap, and
Temporal requirements live in `runtime_preflight_roadmap.md`. Validated
repo-local run-attempt bundles, checkpoints, and raw artifacts are canonical;
timestamped reports are immutable compact interpretations and cannot override
their source evidence.

This is a Phase 4 post-training plan. It consumes, and does not bypass, the
repository-wide prerequisites in
[`docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md`](../../docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md):
core Trainer observability first; Qwen3-0.6B and 1.7B certification on 2, 4,
and 8 GPUs; promotion to Qwen3-8B and Qwen3-30B-A3B; and the intervening
multimodal phase. The one-GPU and sequential-eight-GPU cells below are
post-training experiment tiers, not substitutes for that foundation ladder.

## 1. Research Claims

The program keeps four claims separate.

1. **Scaffold reachability**: a frozen base policy has higher verified pass@K
   than pass@1 under a declared scaffold and sample budget.
2. **Offline compression**: SFT on verified scaffold products moves some of
   that reachable behavior into first-sample policy behavior without an
   unacceptable loss of residual pass@K capability.
3. **Deterministic RL improvement**: strict on-policy RL improves held-out task
   success beyond the exact checkpoint from which it starts, under a numerically
   audited trainer/generator path.
4. **Asynchronous RL utility**: bounded staleness improves throughput or
   utilization without violating policy-freshness invariants or causing an
   unacceptable quality loss.

A later claim cannot repair a failed earlier claim. A smoke proves plumbing,
not convergence or quality. Training reward is diagnostic; locked external
evaluation is the quality result.

## 2. Ownership And Execution Boundaries

The training program spans three existing TorchTitan surfaces and one optional
orchestration layer. They must not be merged into one trainer.

```text
verified offline data
  -> core TorchTitan SFT through run_train.sh / torchrun
  -> explicit DCP / PEFT / full-HF artifact bridge
  -> online RL through python -m torchtitan.experiments.rl.train
  -> repo-local evaluation, report, and promotion
```

Ownership is fixed as follows.

- `experiments/scaffold_to_policy/` owns rootfs-managed runners, declarations,
  data and result roots, campaign reports, and Temporal launch wrappers.
- `torchtitan/experiments/scaffold_to_policy/` owns reusable task import,
  generation, verification, dataset, evaluation, and report helpers.
- Core `torchtitan/train.py` and `torchtitan/trainer.py` own the SFT loop,
  optimizer, scheduler, dataloader state, distributed training, metrics, and
  DCP checkpointing.
- An experiment-owned config registry constructs ordinary `Trainer.Config`
  values. `ConfigManager --module/--config` plus CLI overrides is the
  authoritative SFT configuration path.
- `torchtitan/experiments/rl/` owns the Monarch controller, trainer and
  generator meshes, prompt-group buffering, response-token packing, RL losses,
  weight synchronization, policy-age accounting, and RL checkpoint semantics.
- The experiment lifecycle layer owns immutable run/attempt evidence and
  artifact lineage. It does not implement training.
- Local Temporal workflows durably order coarse stages. Temporal does not own
  scientific identity, artifacts, checkpoints, optimizer steps, or rollout
  scheduling.

The dependency direction remains `experiments -> core`. Core cannot acquire a
Temporal, vLLM, benchmark, or scaffold dependency.

## 3. Model And Compute Ladder

| Tier | Model | Normal purpose | Evidence status |
| --- | --- | --- | --- |
| upstream 2/4/8 GPU | Qwen3-0.6B and 1.7B | core observability, numerics, checkpoint, convergence, and performance certification | prerequisite supplied by the foundation program |
| CPU | fixtures and schemas | splits, verifiers, lifecycle, report, Temporal, and failure tests | contract only |
| 1 GPU | Qwen3-0.6B or debug model where appropriate | SFT, checkpoint, export, load, and RL launch plumbing | plumbing only |
| 1 GPU | Qwen3-1.7B | calibration and bounded LoRA arm screening | exploratory |
| sequential 8 GPU | Qwen3-1.7B | matched replicated SFT, deterministic RL, and async evidence | primary causal evidence |
| certified Qwen3-8B allocation | Qwen3-8B | selected-protocol dense scale probe, then replicated confirmation if authorized | M7 only after upstream 8B promotion and scale-level seed replication |
| separately reviewed MoE allocation | Qwen3-30B-A3B | optional selected MoE transfer/post-training question | separate probe or replicated campaign after upstream MoE promotion |

Routine multi-node work is out of scope. Only one routine 8-GPU campaign cell
runs at a time. Larger models receive the selected arm and a prespecified
control, not the full screening matrix. No model size can waive a split,
verifier, checkpoint, parity, or replication gate.

Each declaration caps work in scientific units:

- number of source problems and rollouts per problem;
- accepted and attempted response tokens;
- optimizer steps and effective global batch;
- RL prompt groups, sibling rollouts, and response-token budget;
- evaluation problems, samples, checkpoints, and retries;
- maximum GPU count and wall-clock timeout.

GPU-hour estimates are recorded for scheduling, but a retry cannot silently
change the scientific budget.

## 4. Durable Training Workflow

The default staged training path is a local Temporal campaign whose Activities
launch the normal rootfs commands.

```text
freeze declaration
  -> runtime and task preflight
  -> generate rollout shards
  -> verify and assemble dataset arms
  -> pre-tokenization census
  -> base evaluation
  -> one complete SFT job per arm
  -> validate resumable DCP checkpoint
  -> export PEFT adapter
  -> adapter evaluation
  -> select on development evidence
  -> materialize and validate full HF policy
  -> deterministic RL
  -> locked evaluation
  -> optional async RL
```

Activities align with atomic artifact-commit seams, not prompts, batches,
optimizer steps, or RL rollouts. Each Activity uses a stable idempotency key
derived from the run declaration, stage, shard, and input digests. A retry may
reuse a completed artifact only after its receipt, hashes, producer identity,
and terminal state validate.

Training Activities heartbeat semantic progress: optimizer or policy step,
last committed checkpoint, evidence sequence, and last progress time. A
heartbeat is not a commit. The local attempt bundle and TorchTitan checkpoint
remain authoritative.

SFT retries may resume from a semantically complete DCP checkpoint. RL
automatic retries remain disabled until the joint controller recovery contract
in Section 12 is implemented and proven.

## 5. Verified Offline Dataset Contract

SFT accepts only an immutable `VerifiedDatasetRef`. Its manifest contains:

- dataset ID, schema version, JSONL path, digest, and row count;
- source task release and raw-cache hashes;
- globally unique problem, candidate, and rollout identities;
- split-registry digest and exact non-overlap result;
- generator policy, tokenizer, prompt, renderer, scaffold, sampling, and seed;
- raw completion and any normalized or cleaned target;
- verifier implementation identity and component results;
- task success and strict-format success as separate fields;
- selection decision, rejection reason, and arm membership;
- response-token count, full token count, and overlength disposition;
- producer run, attempt, and stage lineage.

Assembly re-runs the declared verifier against retained raw output. It rejects:

- train/development/test overlap;
- duplicate or unstable problem identities;
- missing raw provenance;
- verifier drift without a full rescore;
- unverified positive examples;
- changed producing policy or prompt identity;
- transformations that change the verified answer or executable behavior;
- rows whose arm membership cannot be reconstructed.

Before allocating a GPU, tokenize every arm with the exact tokenizer, chat
template, and training sequence length. Record accepted, truncated, and dropped
examples and assistant tokens. The current chat dataloader can drop overlength
samples; a run is not ready unless every arm has enough surviving packed data
for the declared optimizer steps on every data-parallel rank.

### 5.1 Standard SFT arms

The core causal matrix is:

| Arm | Purpose |
| --- | --- |
| base | frozen starting-policy anchor |
| base plus scaffold | inference-time reachability reference |
| raw | minimally transformed verified rollout anchor |
| clean | normalized verified solution candidate |
| formatting/signature | preserve problem-solving content while isolating output-contract learning |

Task-specific hindsight, curriculum, repair, or negative-data arms enter only
after the standard matrix identifies a mechanism that needs them. Oracle or
evaluation-set traces never enter learned-policy data.

For the primary arm contrast:

- use identical training problem IDs across arms;
- report coverage before intersection;
- train the matched intersection for the causal comparison;
- freeze transformation code before training;
- match effective global batch, optimizer steps, schedule, precision, LoRA
  settings, sequence policy, checkpoint selection, and evaluation budget;
- report both example exposure and valid assistant-token exposure;
- run a secondary token-budget-matched analysis when target lengths differ
  materially.

## 6. SFT Configuration Contract

Experiment-specific SFT configuration belongs in an experiment-owned registry
that returns a normal `Trainer.Config`. Existing environment-variable settings
for scaffold data paths, results paths, steps, LoRA rank, and alpha become
temporary compatibility aliases and are removed only after runner migration.

The resolved configuration artifact records:

- selected module and registry function;
- all CLI overrides after parsing;
- normalized full config and digest;
- model and tokenizer identities;
- dataset manifest and ordering;
- global and local batch, gradient accumulation, and batch-mesh degree;
- sequence length, packing, and valid-token policy;
- optimizer, scheduler, precision, compile, and parallelism;
- LoRA targets, rank, alpha, and conversion identity;
- checkpoint interval, load source, export dtype, and output root;
- deterministic seed and library versions.

Numerical comparisons hold effective global batch and tokens per optimizer step
fixed. Pipeline microbatches are not treated as new training examples. Sequence
length and mesh placements must satisfy TP/CP/PP divisibility contracts.

## 7. Offline SFT Research Ladder

### T0: CPU contract

Prove schemas, split integrity, verifier fixtures, arm construction,
pre-tokenization census, resolved-config capture, report construction, and
Temporal replay with a fake executor.

Promotion requires all invariant tests and no unresolved data or verifier
failure.

### T1: Qwen3-0.6B plumbing

Run a minimal but real rootfs job:

1. load the pinned HF checkpoint;
2. train at least two optimizer steps on a fixture-sized verified dataset;
3. save an intermediate full DCP checkpoint;
4. interrupt and resume;
5. save a terminal inference artifact;
6. export a PEFT adapter if using LoRA;
7. reload through TorchTitan and vLLM;
8. evaluate a fixed probe set;
9. finalize an immutable report.

This proves only plumbing and recovery.

### T2: Qwen3-1.7B LoRA calibration

Use development data to calibrate:

- one fixed learning-rate/schedule family;
- a bounded rank choice, normally one default and at most one challenger;
- training horizon using prespecified checkpoints;
- sequence length sufficient for the selected dataset;
- a dataset size that covers enough elicitable problems.

Do not open locked confirmation data. Choose at most one clean champion and
retain raw as the causal anchor.

### T3: Matched replicated Qwen3-1.7B evidence

The default minimum design is `2 fresh split draws x 2 training seeds` for the
champion and raw anchor, with the frozen base evaluated on every final cell.
Base generations may be shared only when their exact problem, prompt, sampling,
and seed identity is valid for every paired comparison. The primary
policy-compression estimand is champion minus base; champion minus raw is a
secondary mechanism estimand when they are distinct. Report every cell
separately.

Before freezing the final sample size, simulate power and interval precision
from development-only estimates of paired problem discordance and between-run
variation. Freeze the number of final problems, split draws, and training
seeds needed to resolve the declared practical effect. The `2 x 2` matrix is a
minimum reproducibility design: inference from only those four trained policy
realizations is conditional on them. A recipe-general claim requires the
prospectively sized upper-level matrix, never fewer than four independent data
draws by three training seeds per arm. If the planned matrix is underpowered,
expand it before opening final data or narrow the claim; an underpowered null
is `hold/inconclusive`, not evidence of no effect.

Minimum generic promotion requirements are:

- a prospectively powered practical threshold, initially `+0.05` absolute
  held-out pass@1 unless the task declaration freezes a justified
  headroom-adjusted value;
- positive champion-minus-base direction in every split-draw mean and a
  two-sided paired problem-level 95% interval above zero for the declared
  fixed-policy estimand;
- residual pass@K no worse by more than `0.02` absolute;
- strict-format success no worse by more than `0.02`;
- positive movement on the base-elicitable subset;
- no material truncation, invalid-output, verifier-error, or safety regression;
- examples and failure transitions supporting task improvement rather than
  parser exploitation.

With the minimum `2 x 2` matrix, hierarchical resampling is descriptive only;
do not claim separate or population-level estimates of split and seed variance.
Spend each frozen final registry once. A correction after a failed gate may use
development data only; a retest requires a new declaration and a newly frozen,
untouched final registry.

### T4: Full-SFT ablation

Full SFT is not a routine arm. Run it only for the promoted LoRA dataset and
recipe when it answers a declared question about adapter capacity or the RL
artifact bridge. Match valid-token exposure and evaluate the same locked set.

### T5: Qwen3-8B scale probe and confirmation

Run the frozen base, selected SFT champion, scaffold reference, and at most one
matched control. Require consistent direction and guardrail retention. Report
conversion of the base-elicitable subset when larger-model saturation reduces
absolute headroom. Qwen3-8B must already have passed the upstream foundation
promotion; this task-specific confirmation cannot certify the base model. One
trained champion/control pair is a selected scale probe. M7 requires
prospectively sized scale-level replication, never fewer than three fresh
training seeds per arm, plus paired final-set uncertainty and the frozen
guardrails.

### T6: Optional Qwen3-30B-A3B MoE confirmation

Do not treat 30B-A3B as the automatic next cell. Propose a separate campaign
only after the upstream MoE routing, expert-load, communication, checkpoint,
convergence, performance, and observability gates pass. Re-review total trainer
plus generator GPU demand and parallelism for SFT and RL. Advance only the
selected champion and one control; preserve MoE-specific routing and load
evidence from the shared core schema.

## 8. Checkpoint Integrity And Resume Proof

Two checkpoint classes have different contracts.

1. **Resumable DCP checkpoint**: model, optimizer, scheduler, dataloader,
   train-step/token state, and RNG state when supported and proven.
2. **Terminal inference artifact**: model-only or adapter output derived from a
   named validated DCP checkpoint.

Periodic recovery checkpoints must not be weakened into model-only saves to
simplify export.

A semantic checkpoint manifest records:

- checkpoint ID and parent;
- source, resolved config, model, tokenizer, and dataset digests;
- global step and valid-token count;
- world size, mesh axes, placements, precision, and compile settings;
- complete state inventory;
- shard metadata and digests;
- persistence start and completion;
- an atomic commit marker written only after asynchronous writes finish.

Load rejects an incomplete commit, missing or corrupt shard, state inventory
mismatch, declaration drift, unsupported topology change, or wrong parent.

The deterministic recovery proof compares an uninterrupted run with a run
interrupted after a committed checkpoint. With the same supported topology,
the full-precision loss and grad-norm trace after restore must match exactly.
Production semantic-resume claims list any state not restored; they do not
upgrade themselves to bitwise claims.

## 9. PEFT And Full-Policy Artifact Bridge

The bridge has three distinct lineage nodes.

```text
validated SFT DCP checkpoint
  -> PEFT adapter for base-plus-adapter evaluation
  -> merged full Hugging Face policy for RL initialization
```

A DCP directory is not a vLLM model. A PEFT adapter is not the current RL
trainer/generator origin. RL begins from a fully materialized HF policy whose
base and adapter parentage is explicit.

The export gate derives geometry from the pinned model configuration and
requires:

- expected tensor names, shapes, dtypes, LoRA targets, rank, and alpha;
- no missing or unexpected state-dict keys;
- correct fused projection and `lm_head` handling;
- parameter-level merge verification;
- tokenizer, vocabulary, RoPE, and model-config identity;
- immutable hashes for all config, tokenizer, safetensors, and index files;
- successful independent load by ordinary non-LoRA TorchTitan and vLLM paths;
- expected nonzero policy delta from base;
- greedy output and task-score agreement for base-plus-adapter versus merged
  policy on a fixed probe set;
- logit or logprob agreement under the declared precision tolerance.

Only a bridge-passing full HF artifact can enter RL.

## 10. RL Task Adapter Contract

A benchmark becomes an RL task only after its offline SFT gate and artifact
bridge pass. The task adapter owns:

- globally disjoint training, validation, and locked evaluation prompts;
- deterministic problem identity and prompt rendering;
- environment reset semantics;
- exact or upstream-owned reward components;
- task success and format success as separate values;
- rollout limits and termination reasons;
- retained raw trajectories sufficient for reward recomputation;
- failure dispositions for empty, failed, overlength, and invalid outputs.

The reward and verifier version freeze before a campaign. Reward changes create
a new declaration and require rescoring retained trajectories. A stratified
audit samples rewarded successes, rewarded failures, truncations, and any
training-reward/evaluation disagreement.

## 11. Deterministic On-Policy RL Ladder

### D0: Qwen3-0.6B end-to-end smoke

Prove one optimizer update, weight publication, generator pull, validation,
checkpoint, and terminal report. Exercise failed, empty, overlength, and
zero-variance groups and show that every active slot receives a terminal
disposition.

### D1: Numerical parity

The certification configuration requires:

- `target_offpolicy_steps=0`;
- `window_fraction=None`;
- matching trainer and generator seeds;
- deterministic and batch-invariant mode on both;
- matching tensor-parallel degree;
- bf16 generator and bf16 trainer forward with fp32 master weights;
- sequence parallelism disabled;
- pipeline parallelism disabled;
- compatible hot-swap and prefix-cache reset behavior;
- shared model, tokenizer, renderer, attention, and override identity;
- validated rollout, packed width, and model RoPE limits.

Promotion requires at every optimizer step:

- exact zero trainer/generator response-token logprob difference;
- exact zero parity KL under the focused parity metric;
- zero nonfinite logprob fraction;
- policy age exactly zero;
- identical repeated-run rollout IDs, rewards, packed batches, losses, and
  final policy tensors;
- supported interrupted/resumed equivalence at a quiescent checkpoint barrier.

Use full-precision trace artifacts and the focused loss-comparison tooling, not
rounded console output.

### D2: Qwen3-1.7B causal RL quality

The primary matrix is:

| Cell | Estimand role |
| --- | --- |
| frozen base | base anchor |
| frozen SFT champion | primary RL starting checkpoint |
| base -> deterministic RL | dependence-on-SFT control |
| SFT -> deterministic RL | primary candidate |
| continued SFT | optional compute-matched control after an RL signal |

The primary estimand is `(SFT -> deterministic RL) - frozen SFT`. Match the
prompt stream, reward, consumed groups, response-token budget, optimizer steps,
validation schedule, checkpoint rule, and offline evaluation budget.

Run two explicitly exploratory pilot seeds on development evidence to validate
reward variance, horizon, and effect direction. Freeze the final protocol after
that pilot; exclude pilot outcomes from confirmatory inference. If the pilot
passes its prespecified feasibility gate and compute is authorized, run five
new confirmatory RL seeds on the locked evaluation schedule. A robust quality
claim requires positive direction in at least four of those five confirmatory
seeds, a prespecified seed-level small-sample or randomization interval above
zero with problem-level pairing retained within seed, the frozen practical
threshold, and no disallowed pass@K, format, truncation, or verifier regression.
If reusing pilot seeds is scientifically necessary, preregister a valid
group-sequential design and adjusted interval before the first seed instead of
conditioning sample size on observed signs.

## 12. Joint RL Recovery Contract

The latest trainer DCP directory alone is not an RL recovery point. A joint
commit contains:

- trainer DCP and policy version;
- optimizer and scheduler state;
- dataset cursor and RNG;
- admitted prompt-group ledger;
- buffered, packed, trained, filtered, failed, and unresolved dispositions;
- rollout-producing policy versions;
- generator synchronization state;
- config, data, model, reward, and renderer lineage.

For deterministic RL, create a quiescent boundary: pause admission, complete
the current update, finish trainer push and all selected generator pulls,
record unresolved prompt identities, and commit the joint manifest.

Production async resume may explicitly discard in-flight rollouts, but it must
record every discarded and replayed prompt ID and prove that committed training
groups are neither duplicated nor lost. Until this contract passes failure
injection, Temporal uses `maximum_attempts=1` for an RL training Activity.

## 13. Asynchronous RL Ladder

Async RL is a separate scientific stage. Progress one factor at a time:

1. deterministic batch-invariant, strict on-policy reference;
2. production kernels, still strict on-policy;
3. async strict FIFO with one frozen off-policy target;
4. async windowed FIFO with matched active capacity;
5. additional generator replicas or larger model only after quality and
   scheduling evidence passes.

Every report verifies:

- active capacity equals `(target_offpolicy_steps + 1) *
  num_prompts_per_train_step`;
- age is measured when the trainer consumes the batch;
- observed maximum policy age respects the derived FIFO/window bound;
- trained-group slots release only after trainer push and all selected
  generator pulls;
- the prior push completes before the next optimizer mutation;
- a generator pull completes before its store value is overwritten;
- incompatible prefix-cache entries are never reused;
- every admitted group receives a terminal disposition;
- failed, empty, overlength, zero-variance, and partial groups cannot starve the
  trainer silently.

Measure generator latency, rollout wait, batching, trainer forward/backward,
optimizer, weight-sync wait, and end-to-end step time separately. Performance
runs use at least ten steady-state optimizer steps after initialization and
compilation.

Generic async promotion requires:

- no correctness, freshness, capacity, or recovery violation;
- a one-sided paired quality confidence bound above the declared `-0.02`
  noninferiority margin against deterministic RL;
- no material failure or drop-rate regression;
- a prospectively sized number of matched steady-state performance repetitions;
  three runs are a plumbing minimum, not promotion evidence;
- at least 15% median end-to-end throughput improvement, unless the declaration
  freezes another justified threshold, with a paired run-level 95% interval
  above zero; and
- a prospectively sized quality-seed design. When multiple async cells are
  opened sequentially, freeze a Holm family or alpha-spending rule so optional
  stopping and cell selection do not invalidate either quality or throughput
  inference.

## 14. Metrics And Statistical Analysis

The problem is the primary evaluation unit. Sibling rollouts are not
independent benchmark observations.

- Compute pass@K from per-problem outcomes.
- Use paired problem bootstrap intervals for matched model differences.
- Report absolute deltas, relative deltas, intervals, and denominators.
- Report every split draw and training/RL seed before a pooled estimate.
- With enough prospectively sized upper-level replication, use a prespecified
  hierarchical or small-sample run-level analysis over draw/seed, nesting the
  paired problem analysis within runs. For the minimum `2 x 2` matrix, report
  the hierarchical bootstrap only as descriptive sensitivity and scope the
  interval to the four realized policies.
- Adjust multiple screened arms with Holm or label selection exploratory.
- Freeze checkpoint selection, analysis, and stopping rules before locked
  evaluation.

Training reports also contain:

- loss and grad norm at full precision;
- valid tokens and examples per optimizer step;
- dropped and truncated data;
- throughput and phase timing;
- checkpoint size, duration, validation, and resume state;
- base, scaffold, raw, clean, formatting, and champion metrics;
- base-elicitable subsets;
- task/format failure transitions;
- representative win, regression, unchanged-failure, and parser/executor cases.

## 15. Failure And Stop Rules

Stop a cell immediately for:

- split overlap, contamination, or declaration drift;
- invalid or changed verifier/reward semantics;
- insufficient surviving training data;
- nonfinite training state;
- corrupt or ambiguous checkpoint lineage;
- export/load or merged-policy equivalence failure;
- trainer/generator model, tokenizer, or logprob mismatch;
- stale prefix cache or partial weight synchronization;
- active-slot leak, impossible train-batch progress, or policy-age violation;
- duplicate Temporal side effects or ambiguous live execution;
- locked-evaluation leakage.

At a prespecified checkpoint, stop an arm for futility when the upper
confidence bound cannot reach the practical promotion threshold. Stop for harm
when pass@K, format validity, truncation, reward integrity, or task-specific
safety guards cross their limits. Saved compute does not authorize an
unregistered search on locked results.

## 16. Initial Campaigns

### TRAIN-MOD-REP: modular replicated SFT

- Prerequisites: lifecycle, rootfs, SFT recovery, and existing modular verifier
  contracts.
- Reclassify the historical seed-only `ood_test` split as held-out IID; freeze
  separate recurrence-length and modulus-range shift axes for OOD guardrails.
- Cells: base, scaffold, raw, clean, formatting.
- Screen: one draw on Qwen3-1.7B.
- Confirm: base, champion, and raw on a minimum `2 fresh draws x 2 seeds`
  conditional matrix; expand to the prospectively sized upper-level design for
  a recipe-general claim.
- Primary endpoint: held-out exact pass@1.
- Guards: pass@8/pass@K, strict format, truncation, invalid output.
- Next gate: full-HF bridge and deterministic modular RL.

### TRAIN-PUBLIC-REASON: first public reasoning SFT

- Prerequisite: modular replication and the chosen task's M2 reachability gate.
- Initial candidates: GSM8K or the verifier-supported MATH subset.
- Use official training data only; lock development and final evaluation before
  generation.
- Confirm one champion and raw anchor before cross-task evaluation.

### TRAIN-CODE-FUNCTION: first coding SFT

- Prerequisites: secure executor, canonical-solution preflight, and MBPP
  reachability.
- Train on permitted MBPP training tasks; keep HumanEval locked for transfer.
- Cells: raw, clean, and signature/formatting control.
- Next gate: full-policy bridge and deterministic executable-reward RL.

### RL-MOD-DETERMINISTIC

- Prerequisites: TRAIN-MOD-REP, full-HF bridge, D0/D1 parity and recovery.
- Cells: base, SFT, base->RL, SFT->RL.
- Pilot: two seeds; expand only under the rule in Section 11.
- Final evaluation: offline exact verifier on globally disjoint prompts.

### RL-MOD-ASYNC

- Prerequisites: replicated deterministic RL and joint recovery proof.
- Cells vary production kernels, off-policy target, and FIFO window
  sequentially.
- Primary endpoints: locked success noninferiority and steady-state throughput.

## 17. Deliverables And Exit Criteria

The training program is ready for a broad research claim only when it has:

- replicated offline compression on at least one exact-verifier reasoning task;
- replicated transfer on at least one public reasoning or coding distribution;
- an audited DCP -> PEFT -> full-HF lineage path;
- deterministic SFT and RL recovery proof;
- exact trainer/generator parity for the deterministic RL reference;
- a replicated incremental RL result or a clearly reported negative result;
- async quality/performance evidence with bounded policy age, if async is
  claimed;
- a selected Qwen3-8B scale probe only for branches that passed at 1.7B, and
  M7 claimed only after the replicated scale-level gate, with Qwen3-30B-A3B
  reserved for a separately approved MoE campaign;
- complete immutable reports, failure analyses, and representative examples.

The program may conclude successfully with a bounded negative finding. It must
not keep scaling or adding benchmarks merely because a promotion gate failed.
