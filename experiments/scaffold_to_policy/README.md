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
