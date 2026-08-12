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

The larger AIME calibration used 8 dev and 8 OOD public AIME 2024 examples,
8 rollouts per problem, and the same no-tool exact final-integer verifier:

```bash
RUN_ID=20260812Taime-8x8-rollouts8-lowmem \
DATA_ROOT=experiments/scaffold_to_policy/data/aime_public_vllm_8x8_rollouts8 \
RESULTS_ROOT=experiments/scaffold_to_policy/results/aime_public_vllm_8x8_rollouts8 \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 NUM_ROLLOUTS=8 \
GPU_MEMORY_UTILIZATION=0.05 \
MAX_NEW_TOKENS=1024 \
PROMPT_VARIANT=chat \
experiments/scaffold_to_policy/run_aime_public_vllm_smoke.sh
```

It reached dev pass@1/pass@8/pass@32 `0.125` and OOD
pass@1/pass@8/pass@32 `0.000`. The only solved dev problem was `AIME/67`,
solved in all eight rollouts; no OOD problem was elicitable at eight samples.
The report is:

```text
experiments/scaffold_to_policy/reports/20260812Taime-8x8-rollouts8.md
```

Run the first public MMLU-Pro no-tool calibration through the rootfs:

```bash
RUN_ID=20260812Tmmlu-pro-8x8-rollouts4-promptfix \
DATA_ROOT=experiments/scaffold_to_policy/data/mmlu_pro_public_vllm_8x8_rollouts4 \
RESULTS_ROOT=experiments/scaffold_to_policy/results/mmlu_pro_public_vllm_8x8_rollouts4_promptfix \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 DEV_OFFSET=0 OOD_OFFSET=32 \
NUM_ROLLOUTS=4 MAX_NEW_TOKENS=1024 \
GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_mmlu_pro_public_vllm_smoke.sh
```

This imports `TIGER-Lab/MMLU-Pro` validation rows, preserves up to ten answer
choices, evaluates Qwen3-1.7B with vLLM, and scores exact final letters with
the multiple-choice verifier. The prompt-fixed run reached dev
pass@1/pass@4/pass@32 `0.250` and OOD pass@1/pass@4/pass@32 `0.750`; strict
format pass@k matched answer pass@k because the repaired prompt eliminated
missing-final failures. The report is:

```text
experiments/scaffold_to_policy/reports/20260812Tmmlu-pro-8x8-rollouts4.md
```

The larger MMLU-Pro calibration used 16 dev and 16 OOD validation examples,
4 rollouts per problem, and the same exact final-letter verifier:

```bash
RUN_ID=20260812T203000Z-mmlu-pro-16x16-rollouts4-ood32 \
DATA_ROOT=experiments/scaffold_to_policy/data/mmlu_pro_public_vllm_16x16_rollouts4_ood32 \
RESULTS_ROOT=experiments/scaffold_to_policy/results/mmlu_pro_public_vllm_16x16_rollouts4_ood32 \
DEV_PROBLEMS=16 OOD_PROBLEMS=16 DEV_OFFSET=0 OOD_OFFSET=32 \
NUM_ROLLOUTS=4 MAX_NEW_TOKENS=1024 GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_mmlu_pro_public_vllm_smoke.sh
```

It reached dev pass@1/pass@4/pass@32 `0.500` and OOD pass@1 `0.5625`,
pass@2 `0.625`, and pass@4/pass@32 `0.6875`. Strict-format pass@k matched
answer pass@k. The report is:

```text
experiments/scaffold_to_policy/reports/20260812T204500Z-hard-reasoning-coding-freegpu.md
```

The current metadata-backed MMLU-Pro refresh used the same 16/16 slice and
four-rollout no-tool condition, and writes rootfs, CUDA, package, model, and
vLLM settings directly into the report input:

```bash
RUN_ID=20260812T230000Z-mmlu-pro-16x16-runtime \
DATA_ROOT=experiments/scaffold_to_policy/data/mmlu_pro_public_vllm_16x16_runtime \
RESULTS_ROOT=experiments/scaffold_to_policy/results/mmlu_pro_public_vllm_16x16_runtime \
DEV_PROBLEMS=16 OOD_PROBLEMS=16 DEV_OFFSET=0 OOD_OFFSET=32 \
NUM_ROLLOUTS=4 GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_mmlu_pro_public_vllm_smoke.sh
```

It reproduced the same metrics and records `runtime_metadata_present=true` in:

```text
experiments/scaffold_to_policy/results/mmlu_pro_public_vllm_16x16_runtime/manifests/report_input_20260812T230000Z-mmlu-pro-16x16-runtime.json
experiments/scaffold_to_policy/reports/20260812T232000Z-hard-runtime-metadata-refresh.md
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

If live Hugging Face access is not available, provide authorized raw row caches
through repo-visible paths and run the same entrypoint offline:

```bash
OFFLINE=1 \
DEV_RAW_CACHE=experiments/scaffold_to_policy/data/gpqa_authorized/raw/dev.jsonl \
OOD_RAW_CACHE=experiments/scaffold_to_policy/data/gpqa_authorized/raw/ood.jsonl \
DATA_ROOT=experiments/scaffold_to_policy/data/gpqa_authorized \
RESULTS_ROOT=experiments/scaffold_to_policy/results/gpqa_authorized \
experiments/scaffold_to_policy/run_gpqa_public_vllm_smoke.sh
```

To verify either access path without launching vLLM generation, run the
rootfs-managed GPQA access preflight:

```bash
experiments/scaffold_to_policy/run_gpqa_access_preflight.sh
```

For authorized offline rows:

```bash
OFFLINE=1 \
DEV_RAW_CACHE=experiments/scaffold_to_policy/data/gpqa_authorized/raw/dev.jsonl \
OOD_RAW_CACHE=experiments/scaffold_to_policy/data/gpqa_authorized/raw/ood.jsonl \
DATA_ROOT=experiments/scaffold_to_policy/data/gpqa_authorized_preflight \
RESULTS_ROOT=experiments/scaffold_to_policy/results/gpqa_authorized_preflight \
experiments/scaffold_to_policy/run_gpqa_access_preflight.sh
```

The preflight writes `gpqa_access_preflight_<run_id>.json` plus a command-stage
manifest under `RESULTS_ROOT/manifests/`. It records whether dev and OOD rows
were accessible, their row source, problem counts, raw-cache hashes when used,
and the rootfs state.

The offline-cache path was smoke-tested with synthetic GPQA-shaped rows in run
`20260812T143600Z-gpqa-offline-cache-smoke`: rootfs imports used
`--offline --raw-cache`, provenance recorded raw-cache hashes, split validation
passed, and the run stopped at an intentionally impossible GPU-memory preflight
without launching vLLM. Authorized real GPQA rows are still required for a
benchmark-preserving GPQA result.

To walk through either access path interactively, use the rootfs-aware setup
wizard:

```bash
experiments/scaffold_to_policy/setup_gpqa_access_wizard.sh
```

The wizard stores a live Hugging Face token in ignored
`.cache/huggingface/token` or records authorized raw-cache paths in `.env`,
then verifies access through `run_gpqa_access_preflight.sh` without launching
vLLM.

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

To test whether those failures are mostly final-grid formatting failures, run
the strict prompt condition on the same slice:

```bash
RUN_ID=20260812T134500Z-arc-agi2-strict-chat-calibration \
DATA_ROOT=experiments/scaffold_to_policy/data/arc_agi2_public_vllm_calibration_strict_chat \
RESULTS_ROOT=experiments/scaffold_to_policy/results/arc_agi2_public_vllm_calibration_strict_chat \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 NUM_ROLLOUTS=4 \
GPU_MEMORY_UTILIZATION=0.24 \
SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN=8192 \
MAX_MODEL_LEN=8192 \
PROMPT_VARIANT=strict_chat \
experiments/scaffold_to_policy/run_arc_agi2_public_vllm_smoke.sh
```

The first attempt passed prompt/context preflight for all 8 dev and 8 OOD
problems, but generation did not run because the visible GPU had insufficient
free memory for vLLM. The ARC entrypoint now writes
`eval/vllm_gpu_memory_preflight.json` before vLLM initialization, so this
failure is caught as a cheap JSON preflight on future runs. The blocked report
is:

```text
experiments/scaffold_to_policy/reports/20260812T134500Z-arc-strict-chat-blocked.md
```

After GPUs became available, the same strict prompt condition completed as run
`20260812T150000Z-arc-agi2-strict-chat-calibration`. It reached dev/OOD
pass@1/pass@4 `0.000`. Missing-final failures dropped to zero, but the original
`chat` prompt's one dev success disappeared and most failures became wrong-grid
failures. The completed report is:

```text
experiments/scaffold_to_policy/reports/20260812T150000Z-arc-strict-chat-results.md
```

To keep the same 8 dev / 8 OOD slice inside a 4096-token context, use the
packed-grid prompt:

```bash
RUN_ID=20260812Tarc-agi2-packed-strict-chat-calibration \
DATA_ROOT=experiments/scaffold_to_policy/data/arc_agi2_public_vllm_calibration_packed_strict_chat \
RESULTS_ROOT=experiments/scaffold_to_policy/results/arc_agi2_public_vllm_calibration_packed_strict_chat \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 NUM_ROLLOUTS=4 \
GPU_MEMORY_UTILIZATION=0.05 \
SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN=4096 \
MAX_MODEL_LEN=4096 \
MAX_NEW_TOKENS=256 \
PROMPT_VARIANT=packed_strict_chat \
experiments/scaffold_to_policy/run_arc_agi2_public_vllm_smoke.sh
```

The completed packed runs selected all 8 dev and 8 OOD problems at 4096
context. `packed_chat` reached dev/OOD pass@1/pass@4 `0.000` with all failures
classified as missing final grids. `packed_strict_chat` also reached
dev/OOD pass@1/pass@4 `0.000`, but shifted failures to malformed final JSON
grids. This means packing solved the context blocker and strict packing solved
the missing-`FINAL` blocker, but neither solved exact-grid output construction.
The report is:

```text
experiments/scaffold_to_policy/reports/20260812Tarc-packed-strict-chat-results.md
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

Run the first LiveCodeBench public-test smoke through the rootfs:

```bash
RUN_ID=20260812Tlivecodebench-public-vllm-smoke \
DATA_ROOT=experiments/scaffold_to_policy/data/livecodebench_public_vllm_smoke \
RESULTS_ROOT=experiments/scaffold_to_policy/results/livecodebench_public_vllm_smoke \
DEV_PROBLEMS=2 OOD_PROBLEMS=2 DEV_OFFSET=0 OOD_OFFSET=32 \
NUM_ROLLOUTS=2 MAX_NEW_TOKENS=1536 \
GPU_MEMORY_UTILIZATION=0.05 TIMEOUT_SECONDS=10 \
experiments/scaffold_to_policy/run_livecodebench_public_vllm_smoke.sh
```

This uses the `contest_code` stdin/stdout verifier for
`livecodebench/code_generation` rows. It executes released public tests only,
so it is not an official LiveCodeBench score. The completed smoke reached
dev/OOD pass@1/pass@2/pass@32 `0.000`; failures were indentation errors and
wrong public-test outputs. The report is:

```text
experiments/scaffold_to_policy/reports/20260812Tlivecodebench-public-vllm-smoke.md
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

The expanded calibration run uses the same verifier and prompt condition on an
8-task dev and 8-task OOD slice, with 4 rollouts per problem:

```bash
RUN_ID=20260812T173000Z-bigcodebench-hard-public-vllm-expanded \
DATA_ROOT=experiments/scaffold_to_policy/data/bigcodebench_hard_public_vllm_expanded_clean \
RESULTS_ROOT=experiments/scaffold_to_policy/results/bigcodebench_hard_public_vllm_expanded_clean \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 DEV_OFFSET=0 OOD_OFFSET=72 \
NUM_ROLLOUTS=4 MAX_NEW_TOKENS=1024 TIMEOUT_SECONDS=60 \
experiments/scaffold_to_policy/run_bigcodebench_hard_public_vllm_smoke.sh
```

The runner now installs the explicit packages exposed by canonical preflight
for these slices (`faker`, `flask`, `flask-login`, `flask-wtf`, `pycryptodome`,
`rsa`, `seaborn`, and `wordcloud`) inside rootfs before preflight. The selected
dev and OOD canonical solutions pass 8/8. Qwen3-1.7B reached pass@1/pass@4
`0.000` on both splits; all 64 sampled candidates failed released unit tests
with assertion failures. The expanded report is:

```text
experiments/scaffold_to_policy/reports/20260812T173000Z-bigcodebench-hard-expanded-calibration.md
```

A follow-up `contract_chat` prompt condition is implemented but currently
blocked by GPU memory contention, not by the coding harness:

```bash
RUN_ID=20260812T183000Z-bigcodebench-hard-contract-chat-blocked \
DATA_ROOT=experiments/scaffold_to_policy/data/bigcodebench_hard_public_vllm_expanded_clean \
RESULTS_ROOT=experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_blocked \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 DEV_OFFSET=0 OOD_OFFSET=72 \
NUM_ROLLOUTS=4 MAX_NEW_TOKENS=1024 TIMEOUT_SECONDS=60 \
PROMPT_VARIANT=contract_chat \
experiments/scaffold_to_policy/run_bigcodebench_hard_public_vllm_smoke.sh
```

The same dev/OOD canonical preflight passed 8/8, then the new
`preflight-vllm-gpu-memory` gate stopped before vLLM generation because cuda:0
had only 4.729 GiB free versus 160.516 GiB required. The blocked report is:

```text
experiments/scaffold_to_policy/reports/20260812T183000Z-bigcodebench-contract-chat-blocked.md
```

The runner now passes `GPU_MEMORY_UTILIZATION` through both the preflight and
`evaluate-coding-style-vllm`, matching the ARC runner behavior. A later
rootfs retry under shared-GPU contention confirmed the same 8/8 dev and 8/8
OOD canonical preflight for the `contract_chat` slice, but the model run still
did not complete because unrelated root-owned SGLang processes consumed nearly
all eight B200s. Under the most contended state, even CUDA memory discovery
failed; `preflight-vllm-gpu-memory` now records that as structured blocker
JSON with `reason: cuda memory query failed` instead of losing the diagnostic
to a Python traceback.

The current completion audit and blocker map is:

```text
experiments/scaffold_to_policy/reports/20260812T142500Z-final-completion-audit.md
experiments/scaffold_to_policy/reports/20260812T123000Z-completion-audit-and-next-steps.md
```

The newer audits include the completed MMLU-Pro and LiveCodeBench lanes,
metadata-backed MMLU-Pro and BigCodeBench-Hard refreshes, plus a fresh GPQA
blocker refresh. They mark the checkpoint blocked on authenticated GPQA access
or an authorized raw GPQA cache, not complete.

Exact-verifier scaffold report inputs share the common
`report_artifacts.build_report_input` shell. `arithmetic_words`, `gsm_style`,
`math_style`, `multiple_choice`, `arc_grid`, `coding_style`, and
`modular_sequences` use it for split-registry checks, summary count checks,
optional preflight checks, artifact hashes, mtimes, and conservative
fresh-vs-reused labeling. Reused artifacts are not treated as failed checks, but
the status is machine-readable for audit reports.

External-harness report inputs keep their own task-execution and reward
semantics, but reuse the same artifact provenance helpers for ingested Harbor,
Terminal-Bench, and tau2 artifacts.

Infrastructure blockers can also be promoted to report inputs without producing
benchmark scores:

```bash
python -m torchtitan.experiments.scaffold_to_policy.cli write-blocker-report-input \
  --results-root experiments/scaffold_to_policy/results/aime_gpu_blocker_report \
  --run-id 20260812T114923Z-aime-gpu-blocker-report \
  --task math_style \
  --lane reasoning \
  --blocker-type vllm_gpu_memory_preflight \
  --artifact gpu_memory=experiments/scaffold_to_policy/results/aime_gpu_blocker_report/eval/vllm_gpu_memory_preflight.json \
  --output experiments/scaffold_to_policy/results/aime_gpu_blocker_report/manifests/report_input_20260812T114923Z-aime-gpu-blocker-report.json
```

The AIME, ARC-AGI-2, and BigCodeBench-Hard runners now use this path for vLLM
GPU-memory preflight failures. A blocker report sets scaffold budget `0`,
records `benchmark_execution_completed=false`, hashes the blocker artifacts,
and explicitly states that no model score was produced.

Those hard-lane runners also source `run_common.sh`, which re-enters the bwrap
rootfs, sets the repo-local Python/Hugging Face/vLLM environment, and writes a
run-scoped JSONL command-stage manifest at
`$RESULTS_ROOT/manifests/$RUN_ID.jsonl`. Each row records the stage name,
command argv, rootfs state, stage work status, start/end time, duration, return
code, data root, and results root. Stage work status is `executed` for normal
runner stages, or a stage-provided value when a sidecar status file is present.
Artifact-level fresh/reused labeling remains in the report input freshness
section. If vLLM passes the memory-selection preflight but fails during model
initialization, the runner writes a JSON stage-failure marker and a
`vllm_runtime_failure` blocker report instead of leaving only a failed shell
exit. The stage-failure marker points back to the JSONL command trace.

Public dataset imports can be made offline-replayable by caching the exact raw
rows selected for a split:

```bash
python -m torchtitan.experiments.scaffold_to_policy.cli import-aime-split \
  --raw-cache experiments/scaffold_to_policy/data/aime_public_vllm_smoke/raw/dev.jsonl \
  --offline \
  --output experiments/scaffold_to_policy/data/aime_public_vllm_smoke/dev.jsonl \
  --provenance experiments/scaffold_to_policy/data/aime_public_vllm_smoke/dev_provenance.json \
  --revision main \
  --limit 8
```

`--raw-cache` without `--offline` reuses an existing cache if present, or writes
one after loading from Hugging Face. `--offline --raw-cache` requires the cache
and never calls `datasets.load_dataset`. Provenance records `row_source`,
`offline`, `raw_cache`, and a hash of the cache artifact. This is available for
GSM8K, MATH, AIME, GPQA, HumanEval, MBPP, and BigCodeBench-Hard imports.

To select the latest run-scoped report input without relying on a mutable
`current` pointer, write a latest-report index from any manifests directory:

```bash
python -m torchtitan.experiments.scaffold_to_policy.cli write-latest-report-index \
  --manifests-dir experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_rerun/manifests \
  --task coding_style \
  --output experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_rerun/manifests/latest_report_index.json
```

The BigCodeBench-Hard `contract_chat` condition has one completed small rerun
and one partial expanded rerun:

```bash
RUN_ID=20260812T235500Z-bigcodebench-hard-contract-chat-rerun \
DATA_ROOT=experiments/scaffold_to_policy/data/bigcodebench_hard_public_vllm_expanded_clean \
RESULTS_ROOT=experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_rerun \
DEV_PROBLEMS=2 OOD_PROBLEMS=2 DEV_OFFSET=0 OOD_OFFSET=32 \
NUM_ROLLOUTS=4 PROMPT_VARIANT=contract_chat GPU_MEMORY_UTILIZATION=0.24 \
SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN=8192 \
experiments/scaffold_to_policy/run_bigcodebench_hard_public_vllm_smoke.sh
```

That run completed under the bwrap rootfs and reached dev/OOD pass@1/pass@4
`0.000`; all sampled candidates failed released tests as assertion failures.
The later 8 dev / 8 OOD attempt with `TIMEOUT_SECONDS=30` completed the dev
half at pass@1/pass@4 `0.000`, but the OOD half remained blocked because GPU
memory changed between preflight and vLLM startup.

The refreshed hard coding blocker run
`20260812T114938Z-bigcodebench-hard-gpu-blocker-report` wrote a report input
instead of launching vLLM on a crowded GPU. Both selected canonical preflights
passed 1/1, but the GPU preflight saw only 1.72 GiB free versus 3.57 GiB
required at `GPU_MEMORY_UTILIZATION=0.02`, so
`benchmark_execution_completed=false` and no score was produced.

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

Run the tau2 upstream execution probe through the rootfs:

```bash
experiments/scaffold_to_policy/run_tau2_execution_probe.sh
```

This creates an isolated tau2 virtualenv, clones the pinned tau2-bench repo for
its `data/` directory, verifies `tau2 check-data`, launches one bounded
mock-domain task through upstream `tau2 run`, parses tau2's saved
`results.json`, and ingests the execution artifact. It is not a successful tau2
model benchmark result unless `task_execution_probes_succeeded=true` in the
report input, and even then the default deterministic probe is harness plumbing
evidence rather than model capability evidence.

The first completed probe reached tau2's batch runner and results writer for
`create_task_1`, but tau2 recorded one infrastructure error and zero evaluated
tasks:

```text
DummyUser.__init__() got an unexpected keyword argument 'tools'
```

The generated artifacts live under:

```text
experiments/scaffold_to_policy/results/tau2_execution_probe_20260812T132500Z/
```

The report is:

```text
experiments/scaffold_to_policy/reports/20260812T132500Z-tau2-execution-probe.md
```

The current runner default uses a benchmark-preserving deterministic probe path
instead of a fake LiteLLM provider. It invokes a small Python wrapper that
imports `torchtitan.experiments.scaffold_to_policy.tau2_probe_agent` before
dispatching to tau2's own CLI, then runs upstream `tau2 run` and ingests tau2's
official `results.json`. The registered probe agent is non-solo and is paired
with `torchtitan_static_user`, avoiding the pinned revision's solo
`DummyUser(tools=...)` constructor mismatch.

The completed deterministic execution probe is:

```text
RUN_ID=20260812T105500Z-tau2-deterministic-execution-probe
RESULTS_ROOT=experiments/scaffold_to_policy/results/tau2_deterministic_execution_probe
```

It recorded `returncode=0`, `num_simulations=1`, `num_evaluated=1`,
`num_infra_errors=0`, and `task_execution_probes_succeeded=true`. This clears
tau2 task-execution plumbing for the pinned mock-domain probe. It does not
evaluate Qwen3, a TorchTitan adapter, or a learned scaffold policy. The next
tau2 step is replacing the deterministic oracle behavior with a bounded model
or policy agent while preserving tau2's released task state, runner, and
evaluator semantics.

Run a non-oracle tau2 baseline by selecting the registered noop agent:

```bash
TAU2_AGENT=torchtitan_noop_agent \
TAU2_USER=torchtitan_static_user \
RESULTS_ROOT=experiments/scaffold_to_policy/results/tau2_noop_baseline \
experiments/scaffold_to_policy/run_tau2_execution_probe.sh
```

The completed baseline `20260812T113000Z-tau2-noop-baseline` ran one upstream
`create_task_1` simulation with `num_evaluated=1`, `num_infra_errors=0`,
`average_reward=0.0`, and `termination_reasons={"agent_stop": 1}`. The scaffold
report input records `task_execution_probes_completed=true` and
`task_execution_probes_succeeded=false`, separating clean tau2 execution from a
solved task. The official reward breakdown was DB `0.0` and COMMUNICATE `1.0`.
The report is:

```text
experiments/scaffold_to_policy/reports/20260812T113000Z-tau2-noop-baseline.md
```

Run the tau2 Qwen model-policy probe by selecting the registered Qwen agent:

```bash
RUN_ID=20260812Ttau2-qwen-model-policy \
RESULTS_ROOT=experiments/scaffold_to_policy/results/tau2_qwen_model_policy \
TAU2_AGENT=torchtitan_qwen_agent \
TAU2_AGENT_LLM=fake \
TAU2_USER=torchtitan_static_user \
TAU2_USER_LLM=fake \
TAU2_MAX_STEPS=3 \
PROBE_TIMEOUT_SECONDS=300 \
SCAFFOLD_TO_POLICY_TAU2_GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_tau2_execution_probe.sh
```

The completed run used the pinned upstream tau2 runner and Qwen3-1.7B/vLLM
inside the bwrap rootfs. It recorded `returncode=0`, `num_evaluated=1`,
`num_infra_errors=0`, and `average_reward=0.0`. The model emitted the correct
`create_task` tool call for `user_1` and `Important Meeting`, and tau2 executed
that tool. The simulation still terminated by `max_steps` before the agent sent
the required confirmation, so `task_execution_probes_succeeded=false`. This is
valid model-policy execution plumbing and a hard-negative task result, not a
successful tau2 benchmark score. The report is:

```text
experiments/scaffold_to_policy/reports/20260812Ttau2-qwen-model-policy.md
```

Run the Terminal-Bench / Harbor oracle execution probe through the rootfs:

```bash
experiments/scaffold_to_policy/run_terminal_bench_oracle_probe.sh
```

This creates an isolated Harbor/Terminal-Bench virtualenv, clones the pinned
Terminal-Bench 2.1 task repo, ingests a Terminal-Bench result-model smoke, and
attempts to run the `headless-terminal` task through Harbor's current CLI. The
entrypoint re-execs through the bwrap rootfs and opts into a narrow host Docker
passthrough (`TORCHTITAN_ROOTFS_BIND_DOCKER=1`) for `/usr/bin/docker`,
`/run/docker.sock`, and the Docker Compose CLI plugin directory. The fixed
ingestion path parses Harbor's own `result.json` and trial `exception_info`
instead of trusting the Harbor process return code.

The Docker/Compose-visible probe initially reached Docker Compose container
creation but did not evaluate a Terminal-Bench trial. Harbor returned process
code 0, while its result artifacts recorded `n_errors=1`, `n_trials=0`, and a
`RuntimeError`:

```text
Error response from daemon: invalid mount config for type "bind": bind source
path does not exist:
/workspace/torchtitan/experiments/scaffold_to_policy/results/terminal_bench_harbor_fixed_ingest/runs/20260812T160000Z-terminal-bench-harbor-fixed-ingest/headless-terminal__gfseUof/verifier
```

The report input now marks `task_execution_probes_succeeded=false` and
`score=0.0` for this failed execution probe. This is blocker evidence for the
Harbor/Docker mount setup, not a Terminal-Bench capability result. The pinned
Terminal-Bench 2.1 task repo uses Harbor's newer `task.toml` layout, while
`terminal-bench==0.2.18` expects the older `task.yaml`/`docker-compose.yaml`
layout. Use `harbor run`, not direct `tb runs create`, for this pinned task
corpus. The earlier report is:

```text
experiments/scaffold_to_policy/reports/20260812T084500Z-terminal-bench-harbor-probe.md
```

The latest probe fixes the bwrap/Docker path mismatch by binding the checkout at
its real host path when Docker passthrough is enabled and running Harbor from
that host-visible path. Harbor then completed one `headless-terminal` oracle
trial with `n_trials=1`, `n_errors=0`, mean metric `1.0`, and verifier reward
`1.0`. This clears the Harbor/Docker/Compose/verifier-mount infrastructure
smoke, but it is still an oracle probe rather than a model or agent capability
result. The current report is:

```text
experiments/scaffold_to_policy/reports/20260812T190000Z-terminal-bench-harbor-hostpath.md
```

Run a non-oracle Harbor baseline by selecting Harbor's built-in `nop` agent:

```bash
HARBOR_AGENT=nop \
RESULTS_ROOT=experiments/scaffold_to_policy/results/terminal_bench_harbor_nop_baseline \
experiments/scaffold_to_policy/run_terminal_bench_oracle_probe.sh
```

The runner name is historical; `HARBOR_AGENT` controls the actual Harbor agent
passed to `harbor run`. The completed baseline
`20260812T111500Z-terminal-bench-harbor-nop-baseline` ran one
`headless-terminal` trial with `agent_name=nop`, `n_trials=1`, `n_errors=0`,
no trial exceptions, and mean metric `0.0`. The scaffold report input records
`task_execution_probes_completed=true` and
`task_execution_probes_succeeded=false`, separating a clean harness execution
from a solved benchmark task. The report is:

```text
experiments/scaffold_to_policy/reports/20260812T111500Z-terminal-bench-harbor-nop-baseline.md
```

Run the bounded task-specific Harbor scripted agent:

```bash
RUN_ID=20260812Tscripted-headless-terminal-smoke \
RESULTS_ROOT=experiments/scaffold_to_policy/results/terminal_bench_harbor_scripted_headless \
HARBOR_AGENT=torchtitan.experiments.scaffold_to_policy.harbor_headless_terminal_agent:HeadlessTerminalScriptAgent \
TIMEOUT_SECONDS=900 \
experiments/scaffold_to_policy/run_terminal_bench_oracle_probe.sh
```

This uses Harbor's released Docker task and verifier for the pinned
`headless-terminal` task, but the policy is a repo-local script specialized to
that task. The completed run recorded one evaluated trial, zero Harbor errors,
and metric `1.0`. Treat it as a custom-agent harness smoke, not as a Qwen3,
learned-policy, or general Terminal-Bench capability result. The report is:

```text
experiments/scaffold_to_policy/reports/20260812Tscripted-headless-and-hard-reruns.md
```

Run the Qwen3/vLLM Harbor model agent:

```bash
RUN_ID=20260812Tqwen-headless-terminal-official-verifier \
RESULTS_ROOT=experiments/scaffold_to_policy/results/terminal_bench_harbor_qwen_headless_official_verifier \
HARBOR_AGENT=torchtitan.experiments.scaffold_to_policy.harbor_headless_terminal_agent:HeadlessTerminalQwenAgent \
TIMEOUT_SECONDS=1200 \
SCAFFOLD_TO_POLICY_HARBOR_GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_terminal_bench_oracle_probe.sh
```

The agent uses the Harbor venv only as the harness process and spawns rootfs
`/usr/bin/python` for local Qwen3-1.7B/vLLM generation. The final run reached
Harbor's official verifier with `n_trials=1`, `n_errors=0`, no trial
exceptions, and reward `0.0`. This clears model-policy execution plumbing for
one pinned Terminal-Bench task, but it is not a successful task solve. The
report is:

```text
experiments/scaffold_to_policy/reports/20260812Tqwen-harbor-model-agent.md
```

The first completed larger BigCodeBench-Hard `contract_chat` rerun used:

```bash
RUN_ID=20260812Tbigcode-hard-contract-chat-8x8-timeout30-rerun \
RESULTS_ROOT=experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_8x8_timeout30_rerun \
DATA_ROOT=experiments/scaffold_to_policy/data/bigcodebench_hard_contract_chat_8x8_timeout30_rerun \
DEV_PROBLEMS=8 \
OOD_PROBLEMS=8 \
DEV_OFFSET=0 \
OOD_OFFSET=32 \
NUM_ROLLOUTS=8 \
PROMPT_VARIANT=contract_chat \
GPU_MEMORY_UTILIZATION=0.05 \
TIMEOUT_SECONDS=30 \
experiments/scaffold_to_policy/run_bigcodebench_hard_public_vllm_smoke.sh
```

The run passed canonical preflights and completed shared-engine vLLM model
execution for both splits. The model score was a hard negative: dev and OOD
pass@1/pass@8/pass@32 were all `0.0`, with all 16 problems unreached.

After GPUs were free, the same hard coding condition was rerun with a fresh
rootfs stage manifest and 8 rollouts per problem:

```bash
RUN_ID=20260812T200000Z-bigcodebench-hard-contract-chat-8x8-freegpu \
DATA_ROOT=experiments/scaffold_to_policy/data/bigcodebench_hard_contract_chat_8x8_freegpu \
RESULTS_ROOT=experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_8x8_freegpu \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 DEV_OFFSET=0 OOD_OFFSET=32 \
NUM_ROLLOUTS=8 PROMPT_VARIANT=contract_chat \
GPU_MEMORY_UTILIZATION=0.05 TIMEOUT_SECONDS=30 MAX_NEW_TOKENS=1024 \
experiments/scaffold_to_policy/run_bigcodebench_hard_public_vllm_smoke.sh
```

The rerun again passed canonical preflight on all 16 selected tasks and reached
dev/OOD pass@1/pass@8/pass@32 `0.0`. Dev failures were 64/64 assertion
failures; OOD failures were 63 assertion failures and one syntax error. The
combined hard reasoning/coding report is:

```text
experiments/scaffold_to_policy/reports/20260812T204500Z-hard-reasoning-coding-freegpu.md
```

The current metadata-backed BigCodeBench-Hard refresh used the same 8/8
`contract_chat` slice, eight rollouts per problem, and released canonical
preflight gate, but writes rootfs, CUDA, package, model, and vLLM settings
directly into the report input:

```bash
RUN_ID=20260812T231500Z-bigcodebench-hard-8x8-runtime \
DATA_ROOT=experiments/scaffold_to_policy/data/bigcodebench_hard_contract_chat_8x8_runtime \
RESULTS_ROOT=experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_8x8_runtime \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 DEV_OFFSET=0 OOD_OFFSET=32 \
NUM_ROLLOUTS=8 PROMPT_VARIANT=contract_chat MAX_NEW_TOKENS=768 \
TIMEOUT_SECONDS=30 GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_bigcodebench_hard_public_vllm_smoke.sh
```

It remained a hard negative, with dev/OOD pass@1/pass@8/pass@32 `0.0`, and
records `runtime_metadata_present=true` in:

```text
experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_8x8_runtime/manifests/report_input_20260812T231500Z-bigcodebench-hard-8x8-runtime.json
experiments/scaffold_to_policy/reports/20260812T232000Z-hard-runtime-metadata-refresh.md
```

The current AIME low-memory hard-reasoning rerun used:

```bash
RUN_ID=20260812Taime-8x8-lowmem-rerun \
RESULTS_ROOT=experiments/scaffold_to_policy/results/aime_public_vllm_8x8_lowmem_rerun \
DATA_ROOT=experiments/scaffold_to_policy/data/aime_public_vllm_8x8_lowmem_rerun \
DEV_PROBLEMS=8 \
OOD_PROBLEMS=8 \
NUM_ROLLOUTS=4 \
GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_aime_public_vllm_smoke.sh
```

It completed both splits. Dev pass@1/pass@4/pass@32 was `0.125` with one easy
problem; OOD pass@1/pass@4/pass@32 remained `0.0`.

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
