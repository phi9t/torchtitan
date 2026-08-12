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
- GSM8K or a small GSM-style subset with parsed final-answer verification;
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
