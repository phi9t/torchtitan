# Coding Research Plan

Date: 2026-08-12
Status: approved-program expansion; execution remains gate-controlled

> This is a long-horizon scientific program, not an instruction to launch the
> entire matrix. Each promoted work package needs a focused implementation or
> experiment plan, a frozen declaration, a compute budget, and review of the
> preceding gate evidence before execution.

## Goal

Determine whether executable coding behavior found by a bounded scaffold can
be compressed into a TorchTitan-trained policy, whether the resulting policy
transfers across progressively harder coding distributions, and whether any
gain survives deterministic RL, bounded asynchronous RL, and downstream
terminal or tool-use evaluation.

The program must distinguish five claims that are easy to conflate:

1. the runner and verifier execute correctly;
2. the base model can reach a solution under a frozen sampling or repair
   scaffold;
3. SFT compresses that reachable behavior into first-sample policy behavior;
4. RL adds value beyond the originating base or SFT policy; and
5. a coding-policy improvement transfers into a different official harness.

No one result establishes all five claims.

## Relationship To The Rest Of The Repository

This plan covers a repo-local research program. Its reusable task import,
verification, evaluation, and reporting code belongs under
`torchtitan/experiments/scaffold_to_policy/`; rootfs-managed runners,
registries, compact reports, and run declarations belong under
`experiments/scaffold_to_policy/`.

The three execution surfaces remain separate:

- Offline generation, SFT-data construction, export, and evaluation are this
  repo-local program.
- SFT uses `torchtitan/train.py -> torchtitan/trainer.py::Trainer`, normally
  through `run_train.sh` and `torchrun`, with an experiment-owned config
  registry.
- Online RL uses `python -m torchtitan.experiments.rl.train`; it is controlled
  by Monarch and must not be wrapped in `torchrun` or folded into the core
  `Trainer` loop.

This coding program is part of Phase 4 post-training. It begins after the
repository-wide core observability, Qwen3-0.6B/1.7B 2-/4-/8-GPU foundation
certification, Qwen3-8B and 30B-A3B promotion, and multimodal phase gates in
[`docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md`](../../docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md).
Its lifecycle and dashboards adopt the already-certified run-attempt schema;
they do not become a second or earlier observability foundation.

The coding program consumes the common run-attempt evidence contract in
`runtime_preflight_roadmap.md`. It must not create a second observability,
checkpoint, or result-manifest authority. TensorBoard is the compact scalar
view; remote tracking is an optional export; the immutable local run-attempt
bundle is canonical.

The approved durable execution shape is:

```text
local Temporal campaign workflow
  -> experiment-run child workflow
     -> coarse idempotent activity
        -> bwrap rootfs runner
           -> import | generate | verify | SFT | export | evaluate | RL
        -> immutable local attempt evidence
```

Temporal is an experiment-only durable control plane. It does not become a core
dependency, a benchmark scorer, a checkpoint format, or the scientific source
of truth.

## Program Principles

- Preserve official executable or final-state scoring. Do not replace tests
  with an LLM judge.
- Run real import, generation, training, export, evaluation, and harness work
  through the repo-local rootfs runners.
- Treat candidate code as untrusted. The current isolated subprocess is useful
  evaluation isolation, but it is not a security boundary.
- Keep task success, output-contract compliance, harness failure, and promotion
  status separate.
- Freeze complete evaluation identities: task release, task manifest, tests,
  evaluator, model, prompt, tools, budgets, rootfs, dependencies, and sampling.
- Keep training, development, locked evaluation, and cross-task transfer
  identities globally disjoint, including near-duplicate and derived-task
  checks where available.
- Use Qwen3-0.6B for contract, recovery, and short RL-parity work; use
  Qwen3-1.7B for primary causal evidence; use certified Qwen3-8B only for
  selected confirmations. Qwen3-30B-A3B requires a separate MoE campaign.
- Prefer a champion-challenger funnel over a full Cartesian product. Only a
  selected arm and one causal anchor advance to replication, RL, agentic
  evaluation, or a larger model.
- Smoke runs prove plumbing only. Zero scores and score regressions are valid
  measurements when execution is valid; they are not infrastructure blockers.
- Stop a branch when its preregistered futility gate fails. Do not spend a
  larger model or RL budget merely because a runner exists.

## Scientific Maturity Ladder

Benchmark difficulty and scientific maturity are orthogonal. A hard task with
a working smoke runner may still be scientifically immature.

| Level | Name | Exit evidence |
| --- | --- | --- |
| M0 | Contract | Pinned source, stable problem IDs, split registry, importer/parser and non-executing verifier fixtures, immutable attempt evidence, and report validation pass. |
| M1 | Base calibration | Independent secure-execution gate C0 plus executable canonical preflight, representative base pass@1 and fixed-budget search coverage, failure taxonomy, runtime metadata, and enough tasks to meet the declared precision design. |
| M2 | Scaffold reachability | A frozen deployable scaffold yields enough verified successes on training-only tasks to build paired SFT arms; held-out evaluation tasks remain unseen. |
| M3 | Matched SFT pilot | Base, raw, clean, and formatting/signature cells train and export successfully under matched budgets; checkpoint restore and adapter loading pass. |
| M4 | Replicated SFT evidence | The selected arm beats the preregistered primary endpoint across independent training draws/seeds, retains search coverage, and passes cross-task and safety guardrails. |
| M5 | Deterministic on-policy RL | The exact-reward RL factorial passes trainer/generator parity, reward-variance, weight-sync, checkpoint, and replicated quality gates. |
| M6 | Bounded asynchronous RL | A matched asynchronous condition preserves quality, respects policy-age and born-fresh invariants, and demonstrates a meaningful throughput or utilization benefit. |
| M7 | Scale confirmation | A selected arm/control pair confirms direction and retention at certified Qwen3-8B across preregistered, power/precision-sized training-seed replication with paired final-set uncertainty. A single comparison is only a scale probe. |

M0 intentionally permits source, importer, split, parser, static verifier-fixture,
and report-contract work that does not execute untrusted code. Secure execution
C0 is an independent prerequisite for every executable canonical preflight,
candidate or model-output evaluation, and M1 transition. Promotion is per task
family. Unopened HumanEval identities can remain a locked transfer set while
the four opened smoke identities remain development regressions and MBPP
reaches M4; after C0 closes, BigCodeBench-Hard can remain an M1 hard negative
while the function-level branch advances.

## Current Evidence Classification

Validated repo-local bundles and raw artifacts are canonical for what actually
ran. The timestamped reports below are immutable compact interpretations of
legacy and current evidence; this table classifies them without upgrading them
into broader claims.

| Branch | Executed evidence | Highest defensible state | Immediate implication |
| --- | --- | --- | --- |
| Candidate execution | `coding_style.py` and `contest_code.py` use `python -I` subprocesses, temporary directories, and wall-time limits. | Pre-M0 security | Useful isolation and deterministic failure classification, but explicitly not a secure sandbox for malicious candidates. The untrusted-execution gate is still open. |
| HumanEval | Qwen3-1.7B, 2 dev and 2 OOD tasks, 2 rollouts each; dev pass@1/pass@2 `0.500`, OOD `0.000` after an extraction repair and rescore. | Pre-M0 plumbing evidence | Pinned import and executable tests work. HumanEval/0, /1, /64, and /65 are opened development regressions; only untouched IDs can enter the locked transfer registry. |
| MBPP | Qwen3-1.7B, 2 dev and 2 OOD sanitized test tasks, 2 rollouts each; dev pass@1/pass@2 `1.000`, OOD `0.500`. | Pre-M0 plumbing evidence | The shared function verifier works, but the slice is tiny and partly saturated. Build a pinned train/dev/test registry before scaffold collection. |
| LiveCodeBench | Qwen3-1.7B, 2 dev and 2 OOD rows, 2 rollouts each; public-test pass@1/pass@2 `0.000` on both splits. | Pre-M0 public-test compatibility | Public stdin/stdout execution works. It is not an official or private-test LiveCodeBench score and is not sufficient for SFT selection. |
| BigCodeBench-Hard | Latest runtime-backed `contract_chat` condition used 8 dev and 8 OOD tasks with 8 rollouts each. Canonical solutions passed 16/16; model pass@1 through pass@8 was `0.000`, with 127 assertion failures and one syntax error. | M1-shaped hard-negative pilot; formal maturity blocked at M0 | The selected rootfs/task slice is environment-clean and currently unreachable by Qwen3-1.7B at the tested budget. Do not create task-specific SFT data from it yet. |
| SciCode | No importer, verifier, runner, registry, or report exists in this checkout. | M0 not started | Integration starts with upstream-contract verification and canonical execution, not a model run. |
| Terminal-Bench via Harbor | One pinned `headless-terminal` oracle scored `1.0`; nop scored `0.0`; a task-specific scripted agent scored `1.0`; Qwen3-1.7B reached the official verifier and scored `0.0`. | Harness plumbing plus one-task hard negative | Docker/Harbor/verifier execution works for one task. Neither the script nor the single Qwen task is general capability evidence. |
| tau2 | Deterministic fixture/probe and noop contracts execute; Qwen3-1.7B made the correct tool call on `mock/create_task_1` but hit `max_steps` before confirmation and received `0.0`. | Harness and model-loop plumbing | Fix the turn loop so tool results return to the policy and a final response can be emitted before any transfer or training claim. |

Current coding evidence is documented in:

- `reports/20260812T101500Z-humaneval-public-vllm-smoke.md`;
- `reports/20260812T084000Z-mbpp-public-vllm-smoke.md`;
- `reports/20260812Tlivecodebench-public-vllm-smoke.md`;
- `reports/20260812T173000Z-bigcodebench-hard-expanded-calibration.md`;
- `reports/20260812T232000Z-hard-runtime-metadata-refresh.md`;
- `reports/20260812Tqwen-harbor-model-agent.md`; and
- `reports/20260812Ttau2-qwen-model-policy.md`.

## Canonical Evaluation Identity

Every evaluation cell gets an `evaluation_id` derived from the complete
condition, including:

- benchmark name, release, source revision, task manifest hash, and split;
- official evaluator/harness revision and test artifact hash;
- candidate-execution image/rootfs and dependency lock;
- model, tokenizer, base checkpoint, adapter, or merged-policy hashes;
- renderer, system prompt, code extraction version, scaffold, and visible
  feedback policy;
- language, compiler/interpreter, package versions, and hardware constraints;
- sampling seed, temperature, top-p, maximum tokens, and rollout budget;
- per-test and per-task timeouts, process/memory/output limits, and retry rules;
- tools, action budget, context policy, and final-answer contract; and
- run declaration, source revision, and dirty-worktree state.

Changing one of these fields creates a new condition. A rescore after an
extractor, dependency, or verifier change is linked to the original generations
but is not silently substituted for the original evaluation.

For each evaluation ID/cell/split, the report must separate:

```text
execution_outcome: completed | blocked | failed | interrupted
measurement: real | fixture | smoke | invalid | not_run
promotion: promote | hold | reject | not_evaluated
```

A candidate that executes and fails every test is a completed real measurement
with score zero. A missing package or invalid canonical solution is an
infrastructure or contract failure and produces no model score.

The attempt outcome is derived separately from all stages and conditions. One
attempt may therefore preserve a real development measurement and a blocked
later split without collapsing either status.

## Data, Split, And Contamination Contract

### Global identity rules

Every problem receives a stable identity based on the upstream identity plus
the pinned content. Registries record both exact IDs and content hashes. Before
training, validate:

- no exact problem ID overlap across train, development, locked evaluation, or
  cross-task transfer sets;
- no duplicate prompt, reference solution, or test-suite hashes;
- no derived variants of an evaluation task in the training pool;
- no near-duplicate natural-language prompt or code template above a frozen
  similarity policy;
- no benchmark tests, hidden outputs, or verifier traces in model-visible
  training targets unless the condition explicitly permits them; and
- no previous evaluation trajectory recycled into later training for the same
  claimed held-out set.

Near-duplicate detection is evidence, not an automatic semantic oracle. All
flagged pairs and adjudications remain in the split registry.

### Benchmark-specific rules

**HumanEval**

- Treat unopened released HumanEval evaluation tasks as locked
  transfer/regression evidence. `HumanEval/0`, `HumanEval/1`, `HumanEval/64`,
  and `HumanEval/65` were already opened by the smoke and extraction
  repair/rescore; register them as development regressions and exclude them
  from every new locked transfer estimand.
- Do not collect scaffold feedback, train, tune prompts, choose checkpoints, or
  select repair policies on those same tasks.
- If a development source is needed, use a separately registered programming
  corpus, not a hidden slice of the HumanEval evaluation set.

**MBPP**

- Verify and pin the upstream split contract before use; replace mutable
  `revision=main` provenance with an immutable revision or cached-row digest.
- Use only an authorized training partition for scaffold collection.
- Create a locked development partition before generation and preserve the
  official test partition for confirmation.
- Keep task IDs and test hashes disjoint from HumanEval and any derived training
  corpus.

**LiveCodeBench**

- Pin the official harness and release semantics, not only the public dataset
  rows.
- Define chronological training, development, and final test windows before
  generation. Record publication timestamps and the model/data cutoff.
- Register `LiveCodeBench/abc301_f`, `LiveCodeBench/abc301_a`,
  `LiveCodeBench/abc307_d`, and `LiveCodeBench/abc307_c` as opened public-test
  development rows. Exclude them from the new locked temporal final window and
  from any official confirmation claim.
- Training and checkpoint selection may use only earlier windows.
- Released public examples are visible feedback only when the named condition
  permits it. Official hidden/private tests remain evaluator-only.
- Report public-test success and official hidden-test success as different
  metrics.

**BigCodeBench-Hard**

- Keep the current canonical-preflight-clean 16-task slice frozen as a
  calibration cell.
- Do not use its released evaluation tests to create training examples and then
  report the same tests as held-out evidence.
- Task-specific SFT requires a separately authorized, globally disjoint task
  pool. A test-visible repair experiment on the frozen slice is an explicitly
  labeled verifier-visible upper bound, not policy generalization evidence.

**SciCode**

- Do not assume a train/evaluation partition. Verify the public release,
  licensing, evaluator, reference artifacts, and split policy first.
- If no disjoint training source exists, SciCode remains evaluation-only and can
  measure transfer from earlier coding champions but cannot supply task-specific
  SFT or RL rewards.

**Terminal-Bench and tau2**

- Environment/task IDs used for training trajectories are disjoint from locked
  agentic evaluation tasks.
- Oracle, scripted, noop, and model-policy trajectories are separate condition
  types and never mixed silently.
- A task's official verifier or final-state reward remains evaluator-owned.

## C0: Untrusted Candidate Execution Gate

Contract-only M0 work may proceed without executing candidate code. C0 is a
separate mandatory gate: no canonical solution, generated candidate, or model
output may execute, and no coding branch may enter M1, until candidate code
executes behind a tested security boundary. `subprocess.run`, `python -I`, a
temporary directory, and a timeout are not sufficient: the process can still
inspect filesystem state, open sockets, spawn descendants, and consume
unbounded resources.

### Required security properties

The selected executor must provide:

- a read-only base filesystem and no writable view of the checkout, model
  assets, checkpoints, caches, or run declarations;
- a fresh, isolated writable directory for exactly one candidate invocation or
  task trial;
- a fresh network namespace with no usable network transport or external
  egress by default; do not share the host network namespace;
- no inherited secrets, tokens, SSH agents, cloud credentials, Docker socket,
  host devices, or unrelated environment variables;
- an unprivileged identity, isolated process namespace, and explicit syscall
  policy appropriate to the language/runtime;
- wall-time, CPU-time, address-space or cgroup memory, process-count, file-size,
  open-file, and output-byte limits;
- bounded interpreter/compiler caches and pinned third-party dependencies;
- full process-group or cgroup termination on timeout, cancellation, worker
  loss, or normal completion;
- no traversal through symlinks, mounts, device nodes, or host paths; and
- structured termination evidence that distinguishes wrong answer, exception,
  timeout, OOM, output limit, process limit, sandbox violation, and evaluator
  failure.

The outer bwrap rootfs and the inner candidate sandbox have different jobs. The
outer rootfs makes the experiment runtime reproducible. The inner sandbox
contains untrusted candidate behavior. A candidate must not inherit broad
capabilities merely because its parent stage needs model or dataset access.

Harbor's Docker capability is reserved for a dedicated trusted harness activity.
Binding `/run/docker.sock` grants a broad host capability; it must never be
available to ordinary candidate-evaluation or model-generation activities.

### Recommended executor architecture

Use a host-side trusted `coding-sandbox` worker to launch a dedicated second
bwrap boundary for each candidate or official task unit. Do not try to secure
candidate execution by nesting it inside the broad experiment rootfs or by
adding filters to the existing `subprocess.run` call.

```text
trusted host coding-sandbox Activity
  -> validate task, tests, dependency image, and limits
  -> create invocation-specific scratch and cgroup
  -> keep hidden tests, expected outputs, and evaluator state in trusted memory
  -> launch minimal bwrap candidate rootfs
       read-only language runtime and pinned dependencies
       candidate entry shim and explicitly public inputs only
       one empty writable scratch mount
       new user, mount, pid, ipc, uts, and network namespaces
       no GPU, Docker, source checkout, model assets, caches, or credentials
  -> trusted evaluator drives bounded stdio/RPC calls and scores outside sandbox
  -> collect bounded result and resource evidence
  -> terminate process group/cgroup and prove cleanup
  -> atomically publish the candidate receipt
```

The candidate rootfs is built and content-identified separately for each
language/dependency profile. It contains only the interpreter/compiler,
declared dependencies, a candidate entry shim, candidate source, and files the
benchmark explicitly makes public. Hidden tests, expected outputs, reference
solutions, and privileged evaluator logic never enter the candidate-readable
mount or environment. A trusted parent evaluator drives a constrained stdio or
RPC protocol across the isolation boundary, or uses the official benchmark's
equivalent hidden-test isolation. The host worker never bind-mounts the
repository.

The exact protocol is task-specific: function tasks may invoke a long-lived
candidate worker with serialized inputs and compare returned values outside the
sandbox; stdin/stdout tasks send hidden stdin and compare captured stdout in the
trusted evaluator. If an official harness requires another design, preserve its
security boundary and score artifact rather than copying hidden tests into the
candidate root. Public tests may be mounted only when the declared benchmark
condition explicitly exposes them.

Use cgroup v2 for memory, process, CPU, and aggregate descendant accounting
when delegated control is available, plus rlimits for file size, open files,
CPU time, and defense in depth. A startup preflight must prove namespace,
cgroup, and cleanup behavior on the actual host. If the host cannot enforce a
declared limit, the profile is blocked; it does not silently fall back to the
current subprocess evaluator. Add a pinned seccomp policy after tracing the
minimum required syscalls for each supported runtime, and treat policy changes
as a new sandbox identity.

Generation and SFT stay in the trusted experiment rootfs. Verification is a
separate Activity that consumes immutable generation references and returns
immutable candidate receipts. This keeps model assets and GPUs outside the
untrusted boundary and lets verification shards retry independently without
rerunning generation.

### Acceptance fixtures

The gate requires automated fixtures for:

- a correct solution and each normal failure class;
- infinite loop and ignored signal;
- fork bomb or rapid descendant spawning;
- memory exhaustion and oversized allocation;
- unbounded stdout/stderr;
- filesystem reads outside the work directory;
- attempts to enumerate or read hidden tests, expected outputs, reference
  solutions, and privileged evaluator files;
- writes through absolute paths, `..`, or symlinks;
- network connection and local socket attempts;
- access to environment tokens and model/cache paths;
- subprocess escape and orphan cleanup;
- timeout during compiler/interpreter startup; and
- cancellation while descendants are active.

Static/non-executing M0 fixtures are necessary but not sufficient for a model
measurement. Executable canonical preflight and M1 require both task semantics
and the independently passed C0 security fixtures.

### Security evidence

Every evaluator report records the sandbox implementation/version, rootfs
identity, dependency image/lock, applied limits, capability set, and cleanup
outcome. Security test results are linked artifacts. The program must not label
the executor "secure" based only on containerization.

## Standard Coding Experiment Matrix

The mandatory cells at a promoted task rung are:

| Cell | Policy and visible information | Purpose |
| --- | --- | --- |
| `base_direct` | Frozen base policy, one sample, no tools or test feedback. | Primary starting point. |
| `base_search` | Same base policy and prompt, frozen K independent samples. | Reachability and search coverage. |
| `base_scaffold` | Same base policy plus one named deployable scaffold with a fixed action/token/test budget. | Inference-time system value and SFT data source. |
| `raw_sft` | SFT on verifier-successful, model-visible scaffold transcripts with failed attempts and allowed feedback preserved. | Verified-trace anchor. |
| `clean_sft` | SFT on normalized final successful code with hidden verifier material removed. | Compression candidate. |
| `format_signature_sft` | SFT on output-only, entry-point/signature, stdin/stdout, or code-fence compliance examples matched to the same problem IDs. | Separate semantic learning from contract learning. |
| `champion_search` | Selected SFT policy at the same K as `base_search`. | Retention of residual search coverage. |

Optional variants, such as critique labels, execution traces, curriculum,
negative examples, or tool traces, enter only after the standard matrix closes.
They receive distinct names and may not replace the raw anchor.

The base, raw, clean, and format/signature SFT arms use the same selected
training problem identities. Reports include both number of examples and valid
assistant-token exposure because raw and clean targets have different lengths.
A selected result receives a secondary valid-token-matched analysis.

Every evaluation also reports the scaffold-elicitable subset: tasks for which
the frozen base scaffold found at least one valid solution. Report champion
pass@1 and fixed-budget search coverage on this subset and on all held-out
tasks. The subset diagnoses compression; it does not replace the all-task
primary endpoint.

For RL, use a separate factorial:

```text
base
SFT
base -> deterministic RL
SFT -> deterministic RL
```

This distinguishes ordinary RL improvement from value attributable to an
SFT-distilled initialization.

## Scaffold Definitions

Scaffolds are evaluation objects, not vague labels. Each declaration pins the
prompt, visible feedback, model calls, K, action count, token budget, timeout,
and stopping rule.

Initial coding scaffolds are:

1. **Best-of-K generation.** No feedback between candidates. Exact tests select
   successful training-only candidates offline.
2. **Syntax and contract repair.** One bounded revision using parser/compiler
   diagnostics and the required entry point or I/O contract. No benchmark test
   failures are shown.
3. **Permitted public-test repair.** One bounded revision using only test cases
   explicitly visible under the benchmark protocol. It is a separate system
   condition.
4. **Training-task test repair.** Bounded revisions against tests from a
   training-only pool. Full transcripts can feed the raw arm; clean code can
   feed the clean arm.

Do not mix these conditions in one rollout set. A direct generation and a
test-guided repair have different deployment costs and leakage risks.

## C1: Function-Level Coding - MBPP To HumanEval

### Research question

Can verifier-successful code and bounded repair behavior collected on disjoint
MBPP training tasks improve first-sample held-out MBPP performance and transfer
to locked HumanEval tasks?

### Contract and calibration

1. Pin immutable MBPP rows, split semantics, test hashes, and canonical code.
2. Create globally disjoint `train_collect`, `dev_select`, and `test_confirm`
   registries. Preserve HumanEval as a separate locked transfer registry.
3. Run the secure-execution and canonical-solution preflights on every selected
   task and dependency profile.
4. Audit entry-point inference, prompt preamble preservation, code extraction,
   timeout classification, and rescore lineage with adversarial fixtures.
5. Run representative Qwen3-1.7B `base_direct` and `base_search` calibrations on
   `dev_select`; freeze K before looking at test results.
6. Measure syntax validity, entry-point compliance, executable success,
   runtime, timeout, and failure category separately.

The current 2+2 HumanEval and MBPP reports remain regression fixtures. They are
not reused as the new calibration or selection set.

### Scaffold and SFT data

On `train_collect`, compare best-of-K, syntax/contract repair, and bounded
training-test repair. For every candidate record:

- problem and split identity;
- producing model/checkpoint and prompt hash;
- sampling and scaffold step identity;
- code before and after each repair;
- only the diagnostics actually shown to the model;
- verifier/test identity and version;
- execution result, resource use, and failure class;
- accepted/rejected reason; and
- raw, clean, or formatting/signature dataset membership.

Build paired arms:

- `raw_sft`: the successful model-visible generation/repair transcript;
- `clean_sft`: only normalized final verified code, with no hidden test output;
- `format_signature_sft`: code-only responses that preserve the requested
  entry point, signature, import preamble, or stdin/stdout shape without adding
  new semantic solutions.

Deduplicate by problem, normalized code, and test hash. Cap the number of
examples per problem so high-yield tasks cannot dominate the dataset.
Materialize each arm as the shared immutable `VerifiedDatasetRef`, including
producer run/attempt/stage lineage, raw and transformed targets, verifier
identity, selection decision, token census, and split-registry digest.

### Evaluation sequence

1. Screen all three SFT arms once on MBPP `dev_select`.
2. Select one champion and preserve `raw_sft` as the causal anchor.
3. Replicate champion and raw anchor across the preregistered draw/seed matrix,
   with matched frozen-base evaluation under every final problem/sampling key.
4. Open MBPP `test_confirm` once for the replicated base, raw anchor, champion,
   and format/signature control.
5. Evaluate the same frozen policies on locked HumanEval without prompt tuning.
6. Re-run `champion_search` to determine whether first-sample gain preserved or
   collapsed base search coverage.

Primary endpoint: paired champion-minus-frozen-base change in executable pass@1
on held-out MBPP. Champion minus `raw_sft` is the secondary mechanism estimand;
it cannot replace the compression endpoint after outcomes are observed.

Guardrails:

- fixed-budget search coverage;
- syntax and entry-point validity;
- timeout and sandbox-violation rates;
- HumanEval transfer;
- output length, tokens, and execution cost; and
- no hidden-test or held-out-task exposure.

### M4 promotion

Promote a function-level champion only if the preregistered minimum practical
pass@1 effect is met, the paired uncertainty interval excludes zero, direction
is consistent across independent training draws, search coverage stays within
its guardrail, and HumanEval does not show a disallowed regression.

If MBPP remains saturated after representative calibration, reduce its role to
training and regression plumbing; use HumanEval only as locked transfer and
advance the scientific endpoint to a harder, disjoint function-level set.

## C2: Temporally Fresh Contest Coding - LiveCodeBench

### Research question

Does the function-level coding champion transfer to temporally fresh contest
problems under the official LiveCodeBench harness, and can training on earlier
windows improve later-window first-sample performance?

### Official-contract gate

The current public-test smoke does not satisfy this gate. Integration must:

- pin the official LiveCodeBench task release, evaluator commit, runtime image,
  language version, and metric implementation;
- preserve publication timestamps and the official private/hidden-test path;
- preflight official canonical solutions under the same environment;
- define chronological training, development, and locked final windows;
- record the base model's plausible training-data cutoff and contamination
  caveats; and
- ingest the harness-owned result artifact without recomputing an alternate
  score.

If official private-test artifacts are unavailable, the branch remains a
public-test development lane and cannot make an official LiveCodeBench claim.

### Experiment cells

Evaluate, in order:

1. `base_direct` with no tools;
2. `base_search` at a frozen K;
3. compile/parser-only repair;
4. public-example repair, only where the official protocol permits it;
5. the replicated MBPP champion with no LiveCodeBench tuning; and
6. a contest-code SFT champion trained only on earlier chronological windows.

Public-example success, compile success, public-test success, and official
hidden-test success are separate outputs. The policy never receives hidden-test
feedback.

### SFT eligibility

Contest-specific SFT begins only if earlier-window training tasks and tests are
authorized, disjoint from the final window, and scaffold reachability yields a
preregistered minimum number and diversity of verified examples. A zero-score
public-test smoke does not meet this gate.

### Promotion and futility

- Promote to M3 only after nonzero scaffold reachability across multiple
  topics/difficulty bands and a locked chronological registry.
- Promote to M4 only on replicated official-harness improvement in a later
  window, with no increased compile/runtime failure rate.
- If Qwen3-1.7B remains universally unreachable after one bounded
  prompt/extraction repair and one permitted scaffold, stop task-specific SFT.
  A single certified-Qwen3-8B reachability calibration may test whether the
  issue is model scale after an adjacent 1.7B coding branch passes; do not
  launch a broad sweep or call it a scale confirmation.
- Never use the final temporal window to choose prompts, checkpoints, K, or
  repair policy.

## C3: Library And API Composition - BigCodeBench-Hard

### Research question

Can a policy trained on simpler verified coding tasks transfer to library/API
composition, and can a disjoint BigCodeBench-style training pool produce a
task-specific gain without evaluation-test leakage?

### Current anchor

Freeze the runtime-backed 8 dev / 8 OOD `contract_chat` condition as the
Qwen3-1.7B hard-negative anchor. All 16 canonical solutions passed under the
recorded dependency environment; all 128 sampled model candidates failed. This
is evidence of current non-reachability, not evidence that the benchmark or
model family is intrinsically impossible.

### Progression

1. Freeze the dependency-complete rootfs profile, exact task manifest, test
   hashes, canonical preflight, prompt, extraction logic, and timeout policy.
2. Re-run a bounded extraction/contract regression only when the implementation
   changes; do not repeatedly sample the locked slice.
3. Evaluate the function-level M4 champion as a pure transfer condition.
4. Compare direct generation with compile/parser-only repair on the same frozen
   condition.
5. Treat any released-test-visible repair on the frozen evaluation slice as an
   explicitly labeled upper bound, never as held-out policy evidence.
6. If Qwen3-1.7B remains unreachable after contract repair and transferred
   policy evaluation, run one certified-Qwen3-8B reachability calibration at
   the same budget only after an adjacent 1.7B coding branch passes.
7. Build task-specific SFT only from a separately authorized, disjoint task
   pool with canonical solutions and executable tests.
8. Expand the locked evaluation slice only after the training pool shows
   nonzero, diverse scaffold reachability.

### Required failure analysis

Report failures by syntax, import/dependency, entry point, API misuse, return
contract, state/side effect, assertion, exception, timeout, OOM, and sandbox
violation. Preserve per-library and task-category strata. The current dominant
assertion failures must be decomposed before choosing a scaffold; "assertion
failure" alone does not distinguish an almost-correct implementation from a
wrong API contract.

### Promotion and futility

- M2 requires verified successes from multiple task/library strata in a
  training-only pool. Success on one repeated template is insufficient.
- M3/M4 use the standard raw, clean, and format/signature arms; dependencies,
  token exposure, and test visibility are matched and recorded.
- If both Qwen3-1.7B and the single selected Qwen3-8B calibration have zero
  reachability after the bounded scaffold, keep BigCodeBench-Hard as a transfer
  and hard-negative benchmark. Do not fabricate SFT data or advance to RL.
- A larger evaluation slice is justified only after reachability, not as a way
  to search for an easier headline result.

## C4: Scientific Programming - SciCode

SciCode is absent from this checkout. Its first milestone is contract
integration, not generation.

### Integration gate

Pin and record:

- the official task release and immutable task manifest;
- evaluator/harness revision and official aggregation;
- allowed dependencies, interpreter/compiler, numerical libraries, and
  hardware assumptions;
- canonical solutions or reference outputs;
- numeric tolerances, units, stochastic seeds, and retry policy;
- per-subproblem dependencies and full-problem composition; and
- licensing and permitted use of tasks for training or evaluation.

Canonical preflight must test exact cases, tolerance boundaries, NaN/Inf,
random-seed sensitivity, nondeterministic libraries, CPU/GPU differences, and
dependency/version failures. Classify code-generation failure separately from
execution, dependency, numerical-validation, and aggregation failure.

### Initial experiment

1. Run oracle/canonical and deliberately wrong controls.
2. Run Qwen3-1.7B `base_direct` and a frozen search budget.
3. Evaluate the function-level and contest-code champions as transfer-only
   conditions.
4. Report per-subproblem success, full-problem success, runtime, tolerance
   margin, and failure taxonomy using the official evaluator.
5. Start task-specific SFT only if a disjoint authorized training source and
   nonzero scaffold reachability both exist.

SciCode RL is ineligible until its reward is deterministic under the pinned
runtime or its stochastic evaluation has an explicit replicated reward
contract. Passing a loose tolerance once is not a stable online reward.

## Cross-Task Transfer Program

Transfer tests answer whether a coding policy learned reusable behavior or
memorized one task contract. They are preregistered before opening the target
set.

| Source policy | Target | Question |
| --- | --- | --- |
| MBPP champion | HumanEval | Does verified function SFT transfer across prompt/test style? |
| MBPP champion | LiveCodeBench | Does function correctness help contest code without target tuning? |
| MBPP champion | BigCodeBench-Hard | Does signature/API discipline transfer to library composition? |
| Contest champion | BigCodeBench-Hard | Does broader algorithmic code generation help API-heavy tasks? |
| Function or contest champion | SciCode | Does general coding transfer to scientific/numerical work? |
| Coding champion | Terminal-Bench | Does code synthesis help a terminal task under Harbor? |
| Coding champion | tau2 | Negative-control transfer unless the selected tau2 tasks genuinely require coding/tool-state skills. |

Every transfer comparison uses the same target prompt, scaffold, sampling, and
harness for base and source policy. No target-set prompt tuning is allowed.
Report both positive and negative transfer. A target score regression is a real
measurement and normally blocks promotion; it is not reclassified as a
non-score result.

## SFT Training Contract

### Configuration and launch

Coding SFT uses an experiment-owned config registry that produces ordinary
TorchTitan `Trainer.Config` values. `ConfigManager` remains the single config
path, with CLI overrides applied after registry defaults. Do not create a
parallel environment-variable configuration system for model, optimizer,
batch, or checkpoint semantics.

Launch through `run_train.sh`/`torchrun`. Hold constant across arm comparisons:

- base checkpoint and tokenizer;
- paired training problem IDs and example cap per problem;
- effective global batch and tokens per optimizer step;
- optimizer steps, schedule, precision, and gradient clipping;
- sequence length, truncation, and overlength rejection policy;
- LoRA targets, rank, alpha, and dropout when LoRA is used;
- parallelism and compilation settings; and
- evaluation prompts, samples, seeds, tests, and harness.

Reports include data token census, examples dropped for length, valid assistant
tokens, realized optimizer steps, and actual work after any filtering.

### Checkpoint and export evidence

Every SFT job registers model, optimizer, scheduler, RNG, dataloader, and train
step state. Before M3:

- an interrupted and restored Qwen3-0.6B debug run matches the uninterrupted
  reference under deterministic comparison;
- the intended Qwen3-1.7B LoRA job saves a complete DCP checkpoint;
- the adapter exports to a distinct PEFT artifact;
- base plus PEFT adapter matches the TorchTitan checkpoint under the declared
  precision on frozen prompts; and
- artifact parentage, config digest, data registry, and source revision are
  immutable and hashed.

DCP, PEFT, and merged Hugging Face policies are distinct lineage nodes. A PEFT
adapter enters online RL only through an explicit policy-artifact bridge that
materializes and validates a full policy loadable by both the RL trainer and
vLLM generator.

### SFT eligibility

A task can enter SFT only when:

- C0 and task-specific M0 pass;
- a legal, globally disjoint training source exists;
- base calibration is neither meaningless saturation nor universal failure;
- the frozen scaffold yields enough verified, diverse training examples to
  fill every paired arm without silently changing problem coverage;
- verifier success and output-contract compliance are separately observable;
- data and test visibility are explicit; and
- the expected training/evaluation budget is declared and approved.

If these conditions fail, keep the task as a plumbing, transfer, or hard-negative
benchmark. A synthetic solution, oracle trace, or evaluation-test-derived target
cannot be substituted to force SFT eligibility.

## Deterministic Online RL

### Entry gates

Coding RL begins only after one coding branch reaches M4 and the full-policy
artifact bridge passes. The reward environment must use training-only tasks and
the secure candidate executor. The official held-out evaluator is never called
inside training.

The first coding RL proof uses Qwen3-0.6B and a small exact-executable training
task. It must satisfy the RL parity contract:

- `target_offpolicy_steps=0`;
- `window_fraction=None`;
- matching deterministic seeds;
- batch-invariant trainer and generator execution;
- matching tensor-parallel degree;
- bfloat16 generator and bfloat16 trainer forward with fp32 master weights;
- sequence parallelism disabled;
- compatible prefix-cache reset on weight synchronization; and
- bitwise trainer/generator logprob and loss comparison through the focused
  parity tools, not rounded console output.

Reward is exact executable task success. Syntax, format, length, or efficiency
shaping must be named, versioned components and evaluated in separate ablations.
Candidate wrong answers, syntax/runtime failures, and benchmark-defined
candidate timeouts produce reward zero. Sandbox-broker, evaluator, or harness
infrastructure failure produces an invalid sample/infrastructure event, not
reward zero.

### Online RL sandbox integration

Temporal does not execute one Activity per rollout. The outer RL Activity
launches the complete Monarch job. A separately supervised host-side
`CodingSandboxBroker` provides the per-candidate reward boundary over a
permission-restricted Unix socket mounted into the trusted RL rootfs.

```text
Monarch rollout environment in trusted RL rootfs
  -> request(request_id, task_id, candidate, verifier_version, limits)
  -> host CodingSandboxBroker
     -> resolve hidden tests in trusted state; accept no arbitrary command/path
     -> run candidate in the minimal bwrap/cgroup sandbox with public inputs only
     -> drive and score the candidate through the constrained protocol outside
        the candidate-readable namespace
     -> atomically commit structured candidate receipt
  <- task outcome or infrastructure disposition
```

The broker never receives arbitrary host paths, shell commands, Docker access,
or model credentials. It resolves `task_id` against an immutable training-only
registry and verifies the candidate/request digest. A stable request ID derives
from run, attempt, rollout, problem, candidate, verifier, and limit identity;
duplicate requests return a validated receipt.

Use a bounded broker queue and fixed sandbox concurrency. The RL environment
awaits the response with a declared deadline and records queue wait separately
from candidate execution. Queue saturation applies backpressure; it does not
become a fabricated zero reward. Broker loss, receipt mismatch, or cleanup
failure marks the group invalid, gives every affected prompt group a terminal
disposition, preserves active-slot accounting, and fails the RL stage when the
declared infrastructure-error threshold is crossed. The outer RL Activity
remains non-retryable until joint controller recovery includes these request
and receipt identities.

The broker is part of the run-attempt process evidence and must be healthy
before admission begins. Deterministic RL fixes broker version, sandbox image,
test registry, concurrency, limits, and request order; repeated reward
recomputation from saved candidate receipts must agree exactly.

### Coding-specific liveness

Before a training step, the controller must be able to accumulate enough
surviving prompt groups. Track failed executions, timeouts, overlength samples,
empty completions, zero-variance groups, packed response tokens, and slot
release. Reward groups with all-zero or all-one outcomes provide no group
advantage; the experiment needs a calibrated task/K combination with usable
reward variance.

### Primary factorial and promotion

Run the matched four-cell factorial on Qwen3-1.7B:

```text
base
SFT
base -> deterministic RL
SFT -> deterministic RL
```

Use identical task registries, reward versions, prompt groups, optimizer-step
budgets, and evaluation objects. Promote only if `SFT -> RL` produces a
replicated held-out improvement beyond SFT alone, while the base-to-RL cell
shows how much of the effect is ordinary RL. Search coverage, cross-task
transfer, execution-failure rate, and KL are guardrails.

## Bounded Asynchronous RL

Async RL is a throughput/stability study after deterministic quality evidence,
not a shortcut around parity.

Progression:

1. strict FIFO and target policy age 0;
2. target off-policy steps 1 with strict FIFO;
3. one controlled look-ahead window;
4. production-style routing; and
5. additional generators or model scale only after the prior condition passes.

Reports separate generator latency, sandbox/execution latency, rollout wait,
batch construction, trainer forward/backward, weight synchronization, and
end-to-end optimizer-step time. They also report consume-time policy age,
reward by policy age, rejected and zero-variance groups, valid response tokens,
padding, prefix-cache reset, and slot-release timing.

The born-fresh invariant remains mandatory: a trained prompt-group slot is
released only after updated trainer weights are published and every selected
generator pulls them.

M6 requires matched held-out quality within its preregistered noninferiority
margin plus a meaningful measured throughput or utilization gain. Automatic
Temporal retry of an RL activity remains disabled until joint controller,
work-buffer, dataset, trainer, generator, and weight-sync recovery is proven.

## Downstream Agentic Boundary

Agentic infrastructure can be hardened independently, but a coding-derived
training or transfer claim requires a coding policy at M4. Harbor and tau2 own
task execution and final scoring; this repo owns pinned launch, policy adapter,
artifact ingestion, lineage, and reports.

### Terminal-Bench through Harbor

The current one-task evidence proves:

- oracle and noop controls reach Harbor's official verifier;
- the bwrap-to-Docker host-path bridge works for the pinned task;
- a task-specific scripted agent can solve that one task; and
- Qwen3-1.7B reaches official scoring and fails with reward `0.0` rather than an
  infrastructure error.

It does not prove general terminal capability. The next sequence is:

1. freeze Harbor, Terminal-Bench task repo, images, verifier, Docker capability,
   policy adapter, context, action, token, and timeout identities;
2. preflight oracle and noop across a stratified development subset;
3. compare frozen base direct execution, plan/execute, and bounded repair;
4. collect successful trajectories only from disjoint training tasks;
5. materialize raw trajectory and cleaned solution SFT arms;
6. evaluate base and coding champion on held-out tasks through Harbor;
7. replicate any task-distribution gain across independent task draws; use
   agent seeds only to characterize stochastic-policy variation within the
   frozen suite; and
8. consider deterministic RL only after multi-step checkpoint/reward replay and
   secure environment reset are proven.

The report includes official reward, task and trial counts, exceptions,
termination, actions, tokens, wall time, retries, environment image, and raw
Harbor result references. Process return code is not task success.

### tau2

The immediate defect is policy-loop semantics: the Qwen agent executed the
correct tool call but did not consume the result and emit the required final
user-facing confirmation before `max_steps`.

The next sequence is:

1. return tool results to the model with stable message/tool identities;
2. require a final response path and explicit stop semantics;
3. prove deterministic database and simulator reset;
4. run noop, scripted, and oracle controls;
5. calibrate a frozen model across tasks, domains, and simulator seeds;
6. compare direct, planning, and critique/retry policies;
7. collect successful trajectories only from disjoint training tasks;
8. run raw-trajectory and cleaned-trajectory SFT;
9. evaluate held-out tasks through upstream `tau2 run` and `results.json`; and
10. consider deterministic final-state RL only after reward recomputation,
    resume, and state-reset equivalence pass.

For coding-policy transfer, select tau2 tasks only when the required behavior
plausibly exercises code, structured tools, or state management. Otherwise
tau2 is a negative-control agentic transfer, not a coding benchmark.

## Statistics And Replication

### Experimental unit

The problem or task is the primary independent unit. Multiple candidates from
one problem are repeated measurements, not independent benchmark examples.
Agentic reruns are nested within task and environment seed. Agent-seed
replication alone supports only a stochastic-policy claim for the exact pinned
task suite. Generalization over an agentic task distribution requires
independent task draws and task-level inference; agent seeds cannot substitute
for task replication.

### Screening and confirmation

- M1/M2 calibration may use one development draw and one training seed, but is
  labeled exploratory.
- M3 screens the standard arms once on the same paired development identities.
- Before M4, use development-only paired outcomes to estimate the plausible
  discordant-pair rate and pilot training runs to bound draw/seed variance.
  Freeze a simulation-based power and precision analysis for the minimum
  practical paired effect, target power, confidence level, and interval width.
  That analysis selects the number of evaluation tasks, independent training
  draws, and training seeds subject to the declared compute cap.
- A fully crossed `2 draws x 2 training seeds` matrix is the minimum replicated
  diagnostic when the dataset permits draws, but its inference is conditional
  on those four trained policies and registered evaluation tasks. It does not
  estimate population-level draw or seed variance and cannot support a general
  training-recipe claim.
- A claim that generalizes the training recipe over data draws and initialization
  seeds requires at least `4 independent training draws x 3 training seeds`,
  fully crossed, and more when the preregistered simulation requires it. With a
  fixed official split, vary training-data draw and training seed without
  resplitting or repeatedly opening the official test set.
- Sampling seeds are fixed and paired between policies. They do not replace
  independent training replication.
- The locked confirmation set is opened once after the arm, checkpoint rule,
  prompt, K, and analysis are frozen. Any correction after opening it is
  development-only. A corrected policy can return to confirmation only under a
  new declaration with a genuinely untouched final registry.

### Estimation

Report:

- paired problem-level pass@1 difference with bootstrap confidence intervals;
- fixed-budget search coverage and its paired difference;
- all-task and scaffold-elicitable-subset estimates, with the subset frozen
  from the base scaffold rather than recomputed per policy;
- an exact discordant-pair or McNemar-style procedure, paired Newcombe interval,
  or another declared paired-score interval when the task set is too small for
  stable paired bootstrap inference; never substitute a one-sample binomial
  interval for a paired policy difference;
- per-draw and per-seed estimates, not only a pooled mean;
- a crossed draw/seed/problem analysis that preserves the registered pairing;
  any resampling interval over fewer than four draws and three seeds is labeled
  conditional to the executed policies rather than population-level evidence;
- task-stratified estimates for difficulty, source, library, or time window;
- format, syntax, runtime, timeout, and sandbox-violation rates;
- token, wall-time, and execution-cost distributions; and
- representative wins, regressions, unchanged failures, and infrastructure
  failures selected by a deterministic rule.

When several arms are screened against one control, adjust the confirmatory
family with Holm's procedure or label the selection analysis exploratory and
reserve the locked test set for the single selected contrast.

The current local summarizers measure whether a problem is solved among the
first K ordered rollouts. Official harnesses may define a different pass@k
estimator. Every report pins the aggregation implementation and must not assume
these metrics are interchangeable.

### Decision rule

Before execution, each task declares:

- primary endpoint;
- minimum practically important effect;
- confidence level and interval method;
- development-only paired-discordance estimate and run-variance assumptions;
- simulation artifact, target power or precision, planned task count, and
  planned independent draw/seed replication;
- whether the claim is conditional to named trained policies or generalizes
  over a declared draw/seed population;
- search-coverage and cross-task regression guardrails;
- maximum task, sample, token, optimizer-step, and GPU budget; and
- one permitted development-only corrective experiment after a failed gate,
  plus a new declaration and untouched final registry if confirmation was
  already opened.

Promotion requires the estimated effect to meet the practical threshold, its
paired interval to exclude zero, and direction to be consistent across
independent draws. A moderator analysis cannot rescue an inconsistent overall
M4 contrast. A moderator frozen before confirmation may instead define a
narrower claim, which requires its own adequately sized analysis and untouched
confirmation evidence. If the campaign cannot attain its preregistered task or
top-level replication requirement, a null or interval crossing zero is
underpowered/inconclusive, not evidence of no effect. It cannot support a
population-level promotion or a negative scientific conclusion; the branch may
still stop for budget or futility with that scope stated explicitly.

## Compute And Model Ladder

The coding program consumes only base checkpoints already certified by the
core training foundation. It does not bypass the numerical, checkpoint,
observability, convergence, or matched-performance gates for the topology
actually used.

| Tier | Model and resource | Purpose |
| --- | --- | --- |
| CPU/static | Fixtures, imports, split/contamination checks, non-executing canonical fixtures, secure-executor tests, report validation, Temporal replay tests; sandboxed CPU canonical execution only after C0. | Contract and failure-proof work. |
| One GPU debug | Qwen3-0.6B generation, short LoRA/checkpoint/export tests, deterministic RL parity and recovery plumbing. | Cheap end-to-end validation, not quality evidence. |
| One GPU primary calibration | Qwen3-1.7B vLLM calibration and small LoRA screens where memory permits. | M1-M3 task selection and arm screening. |
| Sequential eight GPU evidence | Matched Qwen3-1.7B SFT or deterministic RL evidence runs that require the full local node. | M4-M6 causal and reliability evidence. Only one routine eight-GPU activity at a time. |
| Promoted scale probe/confirmation | Certified Qwen3-8B. | One selected arm/control comparison is a scale probe. M7 requires preregistered seed replication and paired final-set uncertainty. |
| Optional MoE campaign | Certified Qwen3-30B-A3B after separate resource and parallelism review. | Narrow scale probe first; any M7 claim must independently meet the replicated confirmation gate. |

No routine multi-node campaign and no broad larger-model sweep is part of this
plan. Qwen3-30B-A3B belongs first to the foundation/MoE vehicle and is not a
coding-lane default.

At Qwen3-8B, a single trained arm/control pair is explicitly a scale probe. M7
requires the frozen arm/control comparison across at least three independent
training seeds, or more when the preregistered power/precision simulation
requires them, plus a paired interval on one untouched final registry. With one
frozen training-data draw, the claim is conditional to that draw. A broader
scale-level training-recipe claim must also meet the `4 draws x 3 seeds` floor
and claim-scope rules above. The larger model may not be used to repair or
select a mechanism after seeing its final-set outcomes.

Matched comparisons keep effective global batch, tokens per optimizer step,
optimizer steps, sequence length, precision, and evaluation budgets fixed.
Performance evidence uses at least ten steady-state optimizer steps after
initialization, compilation, and warmup.

## Promotion And Futility Gates

| Gate | Promote when | Hold or reject when |
| --- | --- | --- |
| M0 contract | Source/importer/parser and non-executing verifier fixtures pass, task IDs/splits are valid, evaluator semantics are pinned, and report evidence is complete. | Mutable/unresolved source, split overlap, parser/fixture failure, or ambiguous scorer semantics. |
| Security C0 | All escape/resource/cleanup fixtures pass under the exact executor profile. | Any candidate can access secrets/host data, use network, leave descendants, or exceed an unenforced limit. |
| Executable preflight | Security C0 passed and canonical solutions pass under the exact candidate profile before M1. | Canonical failure, executor/evaluator mismatch, or any attempt to execute candidates before C0. |
| Base calibration | The sample is representative enough to estimate pass@1, search coverage, and failure strata. | Tiny/saturated/universally failed slice makes the scientific endpoint uninformative. |
| Reachability | Frozen scaffold yields diverse verified successes from training-only tasks at an approved cost. | Zero or template-concentrated success after one bounded scaffold correction. |
| SFT plumbing | Finite matched training, complete checkpoint, deterministic restore, export, and vLLM load/equivalence pass. | Missing state, corrupt lineage, nonfinite training, ambiguous resume, or adapter mismatch. |
| SFT evidence | Replicated paired effect meets the preregistered threshold and guardrails. | Inconsistent draws, interval includes zero, search collapses, or locked transfer regresses beyond guardrail. |
| Policy bridge | Merged/full policy matches adapter behavior and loads in RL trainer and generator. | State-dict, tokenizer, dtype, logits, or logprob mismatch. |
| Deterministic RL | Bitwise parity, usable reward variance, correct weight sync, recovery proof, and replicated incremental quality. | Warning-only determinism, zero-variance starvation, sample leakage, or no gain beyond SFT. |
| Async RL | Quality is noninferior and measured end-to-end throughput/utilization improves with bounded policy age. | Throughput claim is hidden by rollout bottlenecks, quality degrades, or born-fresh/policy-age bounds fail. |
| Scale | Selected 1.7B causal result is already replicated; M7 has simulation-sized seed replication and paired untouched-final uncertainty. | A single 8B comparison is presented as confirmation, or scale is used to rescue a failed contract or causal gate. |
| Agentic | Coding M4 exists, official harness controls pass, and task/train identities are disjoint. | Only oracle/script smoke exists, policy loop is incomplete, or scorer is locally substituted. |

## Durable Temporal Execution

### Ownership

Run a version-pinned local Temporal server with an explicit persistent database
file and a worker outside the rootfs. The worker launches rootfs runners; the
workloads inside bwrap do not need the Temporal SDK. Direct lifecycle execution
remains available for unit tests and one-stage diagnostics.

Use two workflow types:

- `ResearchCampaignWorkflow` sequences gates, fans out approved replicates, and
  waits for human authorization before expensive tiers.
- `ExperimentRunWorkflow` owns one frozen run declaration and coordinates the
  operational attempts needed to reach its terminal run outcome.

`run_id` is the immutable scientific identity; `attempt_id` spans one top-level
operational execution/retry, and `stage_invocation_id` identifies an actual
stage launch. Temporal workflow and workflow-run IDs are correlation fields
only.

### Task queues

Use capability-separated queues:

- `cpu` for imports, registries, data materialization, report validation, and
  workflow replay;
- `coding-sandbox` for untrusted candidate execution without network, model
  assets, or Docker;
- `gpu1` for vLLM generation, calibration, and small LoRA work;
- `gpu8` for sequential evidence-scale SFT or RL;
- `harbor` for trusted Harbor activities with an explicit `harbor-docker`
  resource lease and Docker capability; and
- `tau2` for pinned tau2 harness work.

The worker enforces explicit resource leases. A successful GPU preflight is not
a durable lease; allocation and launch must be one controlled stage so shared
GPU state cannot change unnoticed between them.

### Activity mapping

| Activity | Queue | Idempotent output and heartbeat |
| --- | --- | --- |
| Freeze/import task registry | `cpu` | Cached rows, provenance, task/test hashes; heartbeat rows processed. |
| Validate splits/contamination | `cpu` | Immutable registry and flagged-pair ledger; heartbeat comparison shard. |
| Build sandbox profile | `cpu` | Rootfs/dependency/capability manifest and security-fixture report. |
| Canonical preflight | `coding-sandbox` | Per-task official/canonical result; heartbeat task ID and completed count. |
| Generate rollout shard | `gpu1` | Committed shard ledger and raw generations; heartbeat last committed problem/sample. |
| Verify candidate shard | `coding-sandbox` | Per-candidate structured results; heartbeat problem/sample and completed count. |
| Materialize SFT arm | `cpu` | Hashed JSONL, token census, rejection ledger, and parent rollouts. |
| Train SFT policy | `gpu1` or `gpu8` | TorchTitan checkpoint lineage and metrics; heartbeat optimizer step and last committed checkpoint. |
| Export/validate policy | `gpu1` | PEFT/full-policy artifacts and frozen-prompt parity results. |
| Aggregate evaluated cell | `cpu` | Join immutable generation and candidate receipts into the task/official evaluation artifact. |
| Run deterministic RL | `gpu8` | RL checkpoint/metrics/parity lineage; heartbeat optimizer step and policy version. |
| Run async RL | `gpu8` | Controller metrics and checkpoint lineage; no automatic retry until joint recovery passes. |
| Run Harbor trial bundle | `harbor` | Raw Harbor job/trial results; heartbeat completed trials. |
| Run tau2 bundle | `tau2` | Raw upstream `results.json` and trajectories; heartbeat completed simulation. |
| Validate and render report | `cpu` | Canonical `report_input.json`, outcome, promotion-gate result, and compact Markdown report. |

Activities are coarse stages, never individual prompts, tests, microbatches, or
optimizer steps. One Activity polls exactly one queue. An evaluation workflow
composes `gpu1` generation, `coding-sandbox` verification, and `cpu`
aggregation Activities; it is not a multi-queue Activity. A training Activity's
declaration resolves either `gpu1` or `gpu8` before scheduling.

### At-least-once and cancellation semantics

Every activity reconciles a stable logical stage key derived from `run_id`,
`stage_id`, and the declaration digest. A repo-local `attempt_id` spans one
top-level run execution; each actual stage launch or relaunch gets a distinct
`stage_invocation_id`. A Temporal redelivery that only validates an existing
receipt creates neither. A retry:

1. returns an existing result only after validating output hashes;
2. observes a live job only through a durable supervisor or official harness
   API with verified ownership and terminal receipts; an arbitrary non-child
   PID is ambiguous and blocks rather than relaunches;
3. resumes generation from a committed shard ledger;
4. resumes training only from a complete validated checkpoint;
5. resumes an external harness only through its official job/trial identity; or
6. fails safely when the state is ambiguous.

Candidate wrong answers, zero benchmark reward, and expected timeouts are
successful activity completion with measurement records; they are not retried
as infrastructure errors. Lost workers, transient acquisition failure, or
validated resumable interruption may receive bounded retry. Invalid config,
split leakage, canonical failure, corrupt checkpoint, or sandbox escape is
non-retryable.

Cancellation propagates through bwrap to the full subprocess group/cgroup and
waits for cleanup evidence before the activity closes. Long campaigns use
Continue-As-New only at declared phase boundaries, with immutable local
artifact references carrying the scientific history.

## Observability And Incident Evidence

Every coding stage adopts the common correlation envelope:

```text
run / attempt / workflow / activity / stage
task / split / arm / seed / sample
process / rank / mesh / device
checkpoint / policy version / parent artifact
```

Always-on evidence includes declaration/config digests, command and environment,
stdout/stderr, stage events, task/sample progress, candidate execution resource
use, checkpoint lineage, full-precision training metrics, and terminal outcome.

Coding-specific progress sentinels distinguish:

- generation has not produced a committed sample;
- candidate execution is hung or resource-limited;
- the verifier is making progress through tests;
- SFT has not committed an optimizer step/checkpoint;
- RL cannot form a train step because groups are failing or zero variance; and
- an external harness process exited without valid official result artifacts.

Tiered core diagnostics follow the training vehicle: lightweight Tier 0,
scheduled profiler/memory Tier 1, anomaly-triggered Flight Recorder/NCCL RAS/
py-spy evidence Tier 2, and stopped-job deep diagnostics Tier 3. Do not run
active GPU diagnostics concurrently with candidate or training work.

## Dependency Graph And Deliverables

```text
run-attempt and Temporal foundation
  +-> non-executing M0 source/importer/verifier-fixture contract
  +-> secure candidate execution C0

M0 contract + C0 security
  -> executable canonical preflight and representative M1 calibration
        -> scaffold reachability and paired SFT data
           -> replicated MBPP SFT and HumanEval transfer
              +-> deterministic coding RL -> bounded async RL
              +-> selected certified-8B scale probe
                    -> replicated M7 confirmation
              +-> official LiveCodeBench temporal lane
                    -> BigCodeBench-Hard transfer/task-specific gate
                       -> SciCode transfer/integration
              +-> Harbor and tau2 downstream transfer
```

### Wave C0: Freeze truth and declarations

Deliverables:

- a coding evidence inventory that labels every existing report as fixture,
  smoke, blocker, calibration, or scientific evidence;
- immutable registry entries for current HumanEval, MBPP, LiveCodeBench, and
  BigCodeBench-Hard artifacts;
- explicit invalidation/supersession links for rescored or mutable-revision
  conditions;
- one canonical evaluation identity schema; and
- machine-computed promotion and futility gate outputs.

Exit: existing results can be selected without mutable filename order or
overclaiming their scope. This wave may finish non-executing M0 contract work;
it does not authorize canonical or generated candidate execution.

### Wave C1: Secure executable contract

Deliverables:

- candidate-executor interface and capability profile;
- dependency-image/rootfs manifests;
- functional, adversarial, resource, cancellation, and cleanup fixtures;
- canonical-solution preflight integration; and
- `coding-sandbox` Temporal worker/activity proof under duplicate delivery and
  worker cancellation.

Exit: the independent secure-execution C0 gate passes without Docker or secret
exposure, and executable canonical preflight may begin for M1.

### Wave C2: Function-level calibration

Deliverables:

- immutable MBPP train/dev/test and locked HumanEval transfer registries;
- contamination and near-duplicate audit;
- representative Qwen3-1.7B base/search calibration;
- extraction/signature/timeout failure analysis; and
- frozen scaffold and SFT-data declarations.

Exit: MBPP is calibrated into a useful, non-saturated reachability band or is
reclassified as plumbing only.

### Wave C3: Function-level scaffold and SFT

Deliverables:

- raw, clean, and format/signature paired datasets with full provenance;
- Qwen3-0.6B checkpoint/Temporal recovery proof;
- matched Qwen3-1.7B LoRA screen, DCP checkpoint, PEFT export, and parity proof;
- replicated champion/raw evidence; and
- locked MBPP confirmation plus HumanEval transfer report.

Exit: M4 passes or the function-level policy-compression branch closes.

### Wave C4: Deterministic coding RL

Deliverables:

- secure exact-reward training environment;
- full-policy artifact bridge;
- Qwen3-0.6B trainer/generator parity and recovery evidence;
- Qwen3-1.7B four-cell base/SFT/RL factorial; and
- replicated held-out and cross-task analysis.

Exit: M5 passes or async coding RL remains out of scope.

### Wave C5: Official LiveCodeBench lane

Deliverables:

- official harness/runtime integration and canonical preflight;
- chronological registries and contamination statement;
- public versus private-test metric separation;
- base, scaffold, function-champion transfer, and earlier-window SFT cells; and
- replicated later-window report.

Exit: official M4 evidence passes, or the branch remains public-test development
only.

### Wave C6: BigCodeBench-Hard lane

Deliverables:

- frozen dependency-complete hard-negative anchor;
- deeper assertion/API/side-effect failure taxonomy;
- function and contest champion transfer evaluations;
- one bounded certified-Qwen3-8B reachability calibration if the 1.7B task is
  unreachable but an adjacent 1.7B coding branch passed;
- disjoint training-pool registry if available; and
- task-specific SFT evidence only after reachability.

Exit: M2 or M4 passes as applicable, or BigCodeBench-Hard remains a locked hard
negative/transfer benchmark.

### Wave C7: SciCode integration

Deliverables:

- official task/evaluator/environment pins;
- numeric tolerance, stochasticity, and dependency conformance suite;
- canonical and wrong controls;
- base and prior-champion transfer report; and
- explicit decision on whether legal disjoint SFT/RL data exists.

Exit: M1 calibration passes; M2 begins only with valid data and reachability.

### Wave C8: Agentic downstream transfer

Deliverables:

- multi-task Harbor oracle/noop/model preflight;
- fixed tau2 tool-result/final-response loop and state-reset proof;
- base versus coding champion transfer cells;
- disjoint training trajectory registries where justified;
- upstream-harness-owned result ingestion; and
- replicated agentic reports with errors, cost, and reliability.

Exit: agentic promotion requires M4 coding ancestry plus its own task-specific
gate. Infrastructure can pass even when policy reward is zero.

### Wave C9: Async and scale confirmation

Deliverables:

- strict-on-policy to bounded-off-policy async progression;
- matched quality, policy-age, throughput, and recovery evidence;
- one selected certified-Qwen3-8B arm/control scale probe, followed by
  preregistered seed replication and paired untouched-final uncertainty before
  any M7 confirmation claim; and
- Qwen3-30B-A3B only under a separate MoE campaign after compute and
  parallelism review.

Exit: M6/M7 claims remain narrow to the executed policy, task, harness,
environment, training-data draws, and seed population actually supported. A
single larger-model comparison remains a scale probe.

## Per-Experiment Declaration Template

Every promoted experiment freezes the following before compute allocation:

```yaml
hypothesis:
  id: coding-<branch>-<question>-v1
  predecessor_gates: [exact gate ids]
  primary_estimand: paired held-out pass@1 difference

data:
  source_revision: immutable revision
  train_registry: path and sha256
  dev_registry: path and sha256
  locked_test_registry: path and sha256
  contamination_policy: versioned policy and findings artifact

policy:
  base_checkpoint: path/revision and sha256
  tokenizer: revision and sha256
  arms: [base_direct, base_search, base_scaffold, raw_sft, clean_sft, format_signature_sft]
  checkpoint_selection: frozen rule

scaffold:
  name: exact version
  visible_feedback: exact policy
  num_samples: fixed K
  max_actions: fixed value
  max_tokens: fixed value
  stopping_rule: exact rule

execution:
  rootfs: immutable identity
  sandbox_profile: version and digest
  dependencies: lock/image digest
  timeouts_and_limits: exact values
  temporal_queue: cpu | coding-sandbox | gpu1 | gpu8 | harbor | tau2

evaluation:
  evaluator: commit/version and digest
  prompt_and_extractor: hashes
  sampling_seeds: frozen list
  primary_endpoint: exact implementation
  guardrails: exact implementations

decision:
  minimum_effect: preregistered value
  interval_method: exact method
  paired_discordance_pilot: development artifact and estimate
  run_variance_assumptions: development artifact and bounds
  power_or_precision_simulation: artifact, seed, target, and implementation
  planned_task_count: exact count
  planned_training_draws: exact identities and count
  planned_training_seeds: exact values and count
  claim_scope: named policies | draw-conditional | draw-and-seed population
  promotion_rule: machine-computable expression
  futility_rule: machine-computable expression
  maximum_budget: tasks, samples, tokens, steps, GPU-hours
  permitted_correction: one development-only named branch or none
  post_correction_final_registry: new untouched path and sha256, or none
```

Values are concrete in a run declaration. The template is not permission to
leave placeholders in an executable campaign.

## Completion Criteria

The coding research program is complete enough for a broad scaffold-to-policy
claim only when:

1. candidate execution has a tested security boundary;
2. at least one coding task reaches replicated M4 policy-compression evidence;
3. the gain survives one locked cross-task executable evaluation;
4. checkpoint, PEFT, full-policy, and dataset lineage trace back to the base
   checkpoint and verified source rollouts;
5. deterministic RL is either shown to add replicated value or is closed as a
   negative result;
6. any async claim includes matched quality, policy age, stability, and
   component-level throughput evidence;
7. external harness scores come from Harbor, LiveCodeBench, SciCode, or tau2's
   official artifacts rather than a local replacement;
8. all locked task identities remain disjoint from training and selection;
9. every score is labeled by model, adapter/policy, scaffold, harness, tools,
   environment, and evaluation identity; and
10. timestamped reports preserve successes, regressions, blockers, invalid
    attempts, and stopped branches without rewriting history.

Until then, claims remain scoped to the exact maturity level and executed
evaluation object documented by each report.
