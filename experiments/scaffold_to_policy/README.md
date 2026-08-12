# Scaffold-To-Policy Experiments

This directory records the broader experiment family for compressing verified
scaffold behavior into policy behavior. Countdown remains the first implemented
task under `experiments/countdown_search_distill/`; this directory starts as the
shared planning, registry, and evidence-contract home for the next lanes.

See `agentic_eval_benchmarks.md` for the benchmark landscape, complexity ladder,
normalization rules, and adoption plan.

See `followup_experiment_plan.md` for the concrete staged experiment program
that follows from the Countdown pilot: split repair, Countdown ablations,
reasoning transfer, harness feasibility, and later agentic pilots.

See `spec.md` for the ready-for-agent spec that turns the current execution
state into the next implementation program.

## Current Ladder

1. **Countdown repair**: regenerate clean held-out same-regime splits, enforce
   global problem-key de-duplication, and rerun base plus exported adapters.
2. **Reasoning lane**: add small exact-verifier reasoning tasks before broader
   benchmark claims.
3. **Agentic task lane**: add tau2-bench and Terminal-Bench/Harbor after the
   registry supports external harness boundaries and at least one reasoning
   benchmark shows replicated held-out gain.

## Evidence Contract

Every serious run must record:

- runtime preflight;
- split registry and overlap validation;
- base metrics;
- adapter metrics;
- scaffold subset metrics, such as base-elicitable subsets when applicable;
- verifier failure-mode breakdown;
- representative examples;
- seed or split-draw replication before robust claims.

## Initial Reasoning Benchmarks

The first reasoning lane should include:

- `arithmetic_words`: a local synthetic exact-verifier task that generates
  multi-step arithmetic word problems with strict `FINAL: <integer>` checking;
- `modular_sequences`: a local synthetic exact-verifier task that generates
  modular recurrence problems with strict `FINAL: <integer>` checking;
- `gsm_style`: a fixture-backed GSM-style exact verifier that normalizes
  `FINAL` and GSM8K `####` answers, commas, currency markers, boxed answers,
  integers, decimals, and simple fractions;
- a small MATH subset with task-appropriate answer normalization;
- verifier-first synthetic reasoning tasks where exact verification is cheap.

The first scaffold remains best-of-N sampling. Tool-assisted scaffolds are a
second-tier variant and must be reported separately.

Run the first local reasoning smoke through the bwrap rootfs:

```bash
experiments/scaffold_to_policy/run_arithmetic_words_smoke.sh
```

The smoke writes generated train/dev/OOD problems, a split registry, fixture
evaluations, summaries, and a report input under:

```text
experiments/scaffold_to_policy/data/arithmetic_words_smoke/
experiments/scaffold_to_policy/results/arithmetic_words_smoke/
```

Run the first real-model reasoning smoke through the same rootfs boundary:

```bash
experiments/scaffold_to_policy/run_arithmetic_words_vllm_smoke.sh
```

This evaluates Qwen3-1.7B with vLLM on tiny dev/OOD arithmetic-word splits,
then verifies the outputs with the strict `FINAL: <integer>` checker. It writes
artifacts under:

```text
experiments/scaffold_to_policy/data/arithmetic_words_vllm_smoke/
experiments/scaffold_to_policy/results/arithmetic_words_vllm_smoke/
```

The first completed smoke is summarized in:

```text
experiments/scaffold_to_policy/reports/20260812T055500Z-arithmetic-words-vllm-smoke.md
```

Run the second local reasoning smoke through the bwrap rootfs:

```bash
experiments/scaffold_to_policy/run_modular_sequences_smoke.sh
```

Run the real-model modular recurrence smoke through the same rootfs boundary:

```bash
experiments/scaffold_to_policy/run_modular_sequences_vllm_smoke.sh
```

The vLLM smoke writes artifacts under:

```text
experiments/scaffold_to_policy/data/modular_sequences_vllm_smoke/
experiments/scaffold_to_policy/results/modular_sequences_vllm_smoke/
```

The first useful modular calibration used smaller recurrences:

```bash
RUN_ID=20260812T064500Z-modular-sequences-small-calibration \
DATA_ROOT=experiments/scaffold_to_policy/data/modular_sequences_small_calibration \
RESULTS_ROOT=experiments/scaffold_to_policy/results/modular_sequences_small_calibration \
NUM_ROLLOUTS=8 \
MAX_NEW_TOKENS=512 \
MIN_STEPS=3 \
MAX_STEPS=5 \
MIN_MODULUS=37 \
MAX_MODULUS=257 \
PROMPT_VARIANT=chat \
experiments/scaffold_to_policy/run_modular_sequences_vllm_smoke.sh
```

It produced nontrivial dev and OOD buckets with Qwen3-1.7B: dev pass@1
`0.250`, pass@8 `0.625`, buckets easy 2, elicitable 3, unreached 3; OOD
pass@1 `0.500`, pass@8 `0.625`, buckets easy 4, elicitable 1, unreached 3.
The report is:

```text
experiments/scaffold_to_policy/reports/20260812T064500Z-modular-sequences-calibration.md
```

Run the first modular scaffold-to-policy transfer smoke through the same rootfs
boundary:

```bash
experiments/scaffold_to_policy/run_modular_sequences_transfer_smoke.sh
```

The smoke generates calibrated modular splits, collects base train rollouts,
builds SFT examples from verified rollouts, evaluates base dev/OOD, trains a
Qwen3-1.7B LoRA adapter with TorchTitan, exports it to PEFT/vLLM format, and
evaluates the exported adapter with vLLM LoRA loading. A minimal completed run
used 16 train, 4 dev, 4 OOD, 8 train rollouts, 4 eval rollouts, and 2 training
steps. It wrote artifacts under:

```text
experiments/scaffold_to_policy/data/modular_sequences_transfer_smoke_min/
experiments/scaffold_to_policy/results/modular_sequences_transfer_smoke_min/
```

The minimal run is plumbing evidence, not a scientific transfer claim: dev
pass@1 moved from `0.250` to `0.500`, while OOD pass@k regressed from `0.500`
to `0.250`. The report is:

```text
experiments/scaffold_to_policy/reports/20260812T071500Z-modular-transfer-smoke.md
```

The first larger modular transfer run used the same calibrated task band with
64 train, 32 dev, 32 OOD, 8 rollouts, and 24 LoRA training steps:

```bash
RUN_ID=20260812T073500Z-modular-transfer-expanded \
DATA_ROOT=experiments/scaffold_to_policy/data/modular_sequences_transfer_expanded \
RESULTS_ROOT=experiments/scaffold_to_policy/results/modular_sequences_transfer_expanded \
TRAIN_PROBLEMS=64 \
DEV_PROBLEMS=32 \
OOD_PROBLEMS=32 \
NUM_ROLLOUTS=8 \
EVAL_ROLLOUTS=8 \
MAX_NEW_TOKENS=512 \
TRAIN_STEPS=24 \
CHECKPOINT_STEP=step-24 \
NGPU=1 \
MIN_TRAIN_EXAMPLES=20 \
LORA_RANK=16 \
LORA_ALPHA=32 \
experiments/scaffold_to_policy/run_modular_sequences_transfer_smoke.sh
```

It produced a positive local synthetic transfer signal: dev pass@1 moved from
`0.531` to `0.625`, OOD pass@1 from `0.375` to `0.594`, and base-elicitable
subset pass@1 from `0.000` to `0.750` on dev and `0.800` on OOD while
base-elicitable pass@8 stayed at `1.000`. The report is:

```text
experiments/scaffold_to_policy/reports/20260812T073500Z-modular-transfer-expanded.md
```

Run the first GSM-style verifier smoke through the bwrap rootfs:

```bash
experiments/scaffold_to_policy/run_gsm_style_smoke.sh
```

The smoke prepares checked-in fixture JSONL splits, validates overlap, evaluates
fixture rollouts, and writes a report input under:

```text
experiments/scaffold_to_policy/data/gsm_style_smoke/
experiments/scaffold_to_policy/results/gsm_style_smoke/
```

The first completed report is:

```text
experiments/scaffold_to_policy/reports/20260812T075500Z-gsm-style-smoke.md
```

Run the first no-tool GSM-style real-model smoke through the same rootfs
boundary:

```bash
experiments/scaffold_to_policy/run_gsm_style_vllm_smoke.sh
```

This evaluates Qwen3-1.7B with vLLM on the checked-in GSM-style dev/OOD
fixtures, then verifies the outputs with the exact GSM-style final-answer
normalizer. It writes artifacts under:

```text
experiments/scaffold_to_policy/data/gsm_style_vllm_smoke/
experiments/scaffold_to_policy/results/gsm_style_vllm_smoke/
```

The first completed no-tool vLLM smoke used 3 dev and 3 OOD fixture problems,
4 rollouts per problem, and reached 1.000 pass@1 and strict-format pass@1 on
both splits. One non-first-sample OOD rollout emitted `Final: <answer>` and was
rejected by answer normalization, while pass@1/pass@4 stayed at 1.000. The
report is:

```text
experiments/scaffold_to_policy/reports/20260812T080500Z-gsm-style-vllm-smoke.md
```

Run the first pinned public GSM8K no-tool smoke through the rootfs:

```bash
experiments/scaffold_to_policy/run_gsm8k_public_vllm_smoke.sh
```

This imports small dev/OOD slices from `openai/gsm8k` at dataset revision
`740312add88f781978c0658806c59bc2815b9866`, evaluates Qwen3-1.7B with vLLM
without tools, and verifies outputs with the same exact GSM-style normalizer.
The generated public-slice artifacts live under:

```text
experiments/scaffold_to_policy/data/gsm8k_public_vllm_smoke/
experiments/scaffold_to_policy/results/gsm8k_public_vllm_smoke/
```

The first completed public smoke used 8 dev and 8 OOD examples from the GSM8K
test split, 4 rollouts per problem, and reached dev pass@1 `0.625`, dev
pass@4 `0.750`, OOD pass@1 `0.625`, and OOD pass@4 `0.750`. The report is:

```text
experiments/scaffold_to_policy/reports/20260812T083000Z-gsm8k-public-vllm-smoke.md
```

Run the first pinned public MATH-style no-tool smoke through the rootfs:

```bash
experiments/scaffold_to_policy/run_math_public_vllm_smoke.sh
```

This imports small dev/OOD slices from `EleutherAI/hendrycks_math` algebra at
dataset revision `21a5633873b6a120296cce3e2df9d5550074f4a3`, evaluates
Qwen3-1.7B with vLLM without tools, and verifies outputs with
`math_style_normalized_final_v1`. The generated public-slice artifacts live
under:

```text
experiments/scaffold_to_policy/data/math_public_vllm_smoke/
experiments/scaffold_to_policy/results/math_public_vllm_smoke/
```

The first completed MATH smoke used 8 dev and 8 OOD examples from the algebra
test split, 4 rollouts per problem, and reached dev pass@1 `0.750`, dev
pass@4 `0.750`, OOD pass@1 `0.750`, and OOD pass@4 `0.875` after rescoring
saved generations with the corrected normalizer. The exact verifier remains
conservative: it does not score symbolic algebra equivalence, interval/set
semantics, matrices, or multi-answer semantics. The report is:

```text
experiments/scaffold_to_policy/reports/20260812T091500Z-math-public-vllm-smoke.md
```

Run the first public AIME no-tool smoke through the rootfs:

```bash
experiments/scaffold_to_policy/run_aime_public_vllm_smoke.sh
```

This imports small slices from `HuggingFaceH4/aime_2024`, evaluates Qwen3-1.7B
with vLLM, and scores exact final integer answers with the MATH-style verifier.
The first completed smoke used 2 dev and 2 OOD examples with 2 rollouts per
problem. It reached pass@1/pass@2 `0.000` on both splits, with failures split
between missing final answers and wrong final values. The report is:

```text
experiments/scaffold_to_policy/reports/20260812T083500Z-aime-public-vllm-smoke.md
```

Run the GPQA Diamond gate through the rootfs:

```bash
experiments/scaffold_to_policy/run_gpqa_public_vllm_smoke.sh
```

GPQA Diamond is gated on Hugging Face. In an unauthenticated rootfs, this script
writes a blocker report input instead of substituting a different dataset or
claiming a score. With `HF_TOKEN` configured and access granted, the same
entrypoint imports `Idavidrein/gpqa` `gpqa_diamond`, evaluates no-tool
multiple-choice prompts, and verifies exact `FINAL: <A|B|C|D>` outputs.

Run the first ARC-AGI-2 no-tool smoke through the rootfs:

```bash
experiments/scaffold_to_policy/run_arc_agi2_public_vllm_smoke.sh
```

This clones `https://github.com/arcprize/ARC-AGI-2.git` at revision
`f3283f727488ad98fe575ea6a5ac981e4a188e49`, imports small task slices from the
released JSON files, preflights tokenizer/context fit, evaluates Qwen3-1.7B
with vLLM, and scores exact JSON grid outputs with `FINAL: <json-grid>`. The
first completed smoke used 2 dev and 2
OOD examples with 2 rollouts per problem. Dev reached pass@1 `0.000` and
pass@2 `0.500`; OOD reached pass@1/pass@2 `0.000`. The dominant failure mode
was format-contract failure where the model emitted `FINAL: <json-grid>`
literally or emitted an incorrect grid. The report is:

```text
experiments/scaffold_to_policy/reports/20260812T092000Z-arc-bigcodebench-hard-smokes.md
```

The larger ARC calibration used 8 dev and 8 OOD training-split tasks with
4 rollouts per problem. Because the local B200s were partially occupied, the run
used `GPU_MEMORY_UTILIZATION=0.24`; because one selected prompt exceeded
4096 tokens, it used `SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN=8192`. It reached
dev pass@1 `0.000`, dev pass@4 `0.125`, and OOD pass@1/pass@4 `0.000`. The
report is:

```text
experiments/scaffold_to_policy/reports/20260812T131000Z-arc-agi2-calibration.md
```

## Initial Coding Benchmarks

The first coding lane uses `coding_style`, a repo-owned executable-test harness
for small HumanEval slices. It is a stepping stone toward LiveCodeBench,
SWE-bench, Terminal-Bench, and Harbor-backed runs, not a substitute for their
official harnesses.

Run the first pinned public HumanEval no-tool smoke through the rootfs:

```bash
experiments/scaffold_to_policy/run_humaneval_public_vllm_smoke.sh
```

This imports small dev/OOD slices from `openai/openai_humaneval` at dataset
revision `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`, evaluates Qwen3-1.7B
with vLLM, extracts candidate Python code, and runs the benchmark tests in a
separate Python subprocess with a timeout. The generated artifacts live under:

```text
experiments/scaffold_to_policy/data/humaneval_public_vllm_smoke/
experiments/scaffold_to_policy/results/humaneval_public_vllm_smoke/
```

The first completed coding smoke used 2 dev and 2 OOD examples, 2 rollouts per
problem, and reached dev pass@1 `0.500`, dev pass@2 `0.500`, OOD pass@1
`0.000`, and OOD pass@2 `0.000` after rescoring saved generations with a
corrected prompt-preamble extraction rule. The report is:

```text
experiments/scaffold_to_policy/reports/20260812T101500Z-humaneval-public-vllm-smoke.md
```

Run the first pinned public MBPP no-tool smoke through the rootfs:

```bash
experiments/scaffold_to_policy/run_mbpp_public_vllm_smoke.sh
```

This imports `google-research-datasets/mbpp` `sanitized`, infers each task's
entry point from released assert tests, evaluates Qwen3-1.7B with vLLM, and
scores candidates with the same executable Python subprocess verifier. The
first completed smoke used 2 dev and 2 OOD examples with 2 rollouts per
problem. It reached dev pass@1 `1.000`, dev pass@2 `1.000`, OOD pass@1
`0.500`, and OOD pass@2 `0.500`. The report is:

```text
experiments/scaffold_to_policy/reports/20260812T084000Z-mbpp-public-vllm-smoke.md
```

Run the first pinned BigCodeBench-Hard no-tool smoke through the rootfs:

```bash
experiments/scaffold_to_policy/run_bigcodebench_hard_public_vllm_smoke.sh
```

This imports `bigcode/bigcodebench-hard` split `v0.1.4`, wraps each released
`unittest` suite into the shared executable `check(candidate)` verifier,
preflights the released canonical solutions under the same rootfs executable
verifier, evaluates Qwen3-1.7B with vLLM, and executes candidates in an
isolated Python subprocess. The canonical preflight is intentionally before
vLLM generation, so missing rootfs packages or broken released tests are
environment failures rather than model failures. The first attempted smoke
exposed a missing rootfs `matplotlib` dependency on one OOD task; after
installing `matplotlib` inside the bwrap rootfs, the repaired run used 2 dev
and 2 OOD examples with 2 rollouts per problem. Both splits reached
pass@1/pass@2 `0.000`, with all final failures classified as assertion
failures from the benchmark tests. The report is:

```text
experiments/scaffold_to_policy/reports/20260812T092000Z-arc-bigcodebench-hard-smokes.md
```

The current completion audit and blocker map is:

```text
experiments/scaffold_to_policy/reports/20260812T123000Z-completion-audit-and-next-steps.md
```

## Agentic Benchmarks

tau2-bench and Terminal-Bench/Harbor are in scope, but they enter after the
reasoning lane because they add external harnesses, tool/user or terminal
interaction semantics, and slower artifact loops.

For Harbor/Terminal-Bench, this repo should own rootfs-managed launch,
registry entries, artifact paths, and result ingestion. Harbor should own task
execution.

For tau2-bench, this repo should own rootfs-managed install/run pinning,
trajectory and score ingestion, and report generation. The tau2 harness should
own task execution.

Run the first external-harness dry-run ingestion smoke through the rootfs:

```bash
experiments/scaffold_to_policy/run_external_harness_dry_run_smoke.sh
```

This records pinned Harbor, Terminal-Bench 2.1, and tau2-bench revisions,
captures rootfs/runtime metadata, writes dry-run harness-owned score and
trajectory fixtures, ingests them, and builds a shared report input. It does
not install or execute the external harnesses and must not be cited as a
benchmark result. The generated artifacts live under:

```text
experiments/scaffold_to_policy/results/external_harness_dry_run_smoke/
```

The first completed dry-run smoke used Harbor revision
`b7e2f71b4563618af3a42279740f5f412dcf7046`, Terminal-Bench 2.1 revision
`7131e4375048a0e408a8fb404b5f499d726b695b`, and tau2-bench revision
`668d3bcd135c02aa3438f987ef45735b7c163ee3`. It confirmed the rootfs launch and
ingestion contract, while recording that `harbor`, `terminal_bench`, and
`tau2` are not installed in the current rootfs and Docker/bwrap binaries are
not available inside that environment. The report is:

```text
experiments/scaffold_to_policy/reports/20260812T103500Z-external-harness-dry-run-smoke.md
```

Run the installed external-harness package preflight through the rootfs:

```bash
experiments/scaffold_to_policy/run_external_harness_preflight_smoke.sh
```

This creates isolated virtualenvs under the ignored results tree, installs
`harbor==0.21.0`, `terminal-bench==0.2.18`, and Sierra tau2-bench from the
pinned git revision, verifies that the expected Python modules import with
matching package versions, ingests the raw preflight artifacts, and builds a
shared report input. It does not complete Terminal-Bench, Harbor, or tau2 tasks
and must not be cited as a benchmark result. The generated artifacts live under:

```text
experiments/scaffold_to_policy/results/external_harness_preflight_corrected/
```

The first completed installed preflight used the same repo revision pins as the
dry-run smoke. It passed rootfs, import, and version checks for all pinned
packages after splitting Harbor/Terminal-Bench and tau2-bench into separate
venvs. The corrected run found the `harbor`, `terminal-bench`, `tb`, and `tau2`
CLI entrypoints. A follow-up tau2 mock-domain runner probe reached tau2's
runner and results writer, but ended with infrastructure errors for the tested
no-external-LLM dummy-user pairings. The report is:

```text
experiments/scaffold_to_policy/reports/20260812T105500Z-external-harness-preflight-smoke.md
```

Run the first tau2 scorer-ingestion smoke through the rootfs:

```bash
experiments/scaffold_to_policy/run_tau2_mock_score_smoke.sh
```

This installs Sierra tau2-bench from the pinned revision, uses its pinned
`data/` checkout for the mock domain, constructs a deterministic valid
`create_task_1` fixture trajectory, scores it with tau2's own evaluator, and
ingests the result. It is a scorer-preserving fixture smoke, not a model or
agent benchmark result. The generated artifacts live under:

```text
experiments/scaffold_to_policy/results/tau2_mock_score_smoke/
```

The first completed run scored the mock task with `all_ignore_basis` and
reached reward `1.0`, with DB, ACTION, and COMMUNICATE reward components all
equal to `1.0`. The report is:

```text
experiments/scaffold_to_policy/reports/20260812T121500Z-tau2-mock-score-smoke.md
```

Run the Terminal-Bench / Harbor oracle execution probe through the rootfs:

```bash
experiments/scaffold_to_policy/run_terminal_bench_oracle_probe.sh
```

This creates an isolated Harbor/Terminal-Bench virtualenv, clones the pinned
Terminal-Bench 2.1 task repo, ingests a Terminal-Bench result-model smoke, and
attempts to run the `headless-terminal` task through Harbor's current CLI. The
first corrected probe reached Harbor's environment boundary and failed with:

```text
Docker is not installed or not on PATH. Please install Docker and try again.
```

It also established that the pinned Terminal-Bench 2.1 task repo uses Harbor's
newer `task.toml` layout, while `terminal-bench==0.2.18` expects the older
`task.yaml`/`docker-compose.yaml` layout. Use `harbor run`, not direct
`tb runs create`, for this pinned task corpus. The report is:

```text
experiments/scaffold_to_policy/reports/20260812T084500Z-terminal-bench-harbor-probe.md
```

## Registry Shape

Each run family should be described by a machine-readable registry with:

- task name and lane;
- verifier type;
- model and asset path;
- split registry path;
- scaffold type and rollout budget;
- training arms;
- training budget;
- evaluation budget;
- success thresholds;
- artifact roots;
- external harness boundary, if any;
- report template.
