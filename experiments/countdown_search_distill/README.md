# Countdown Scaffold-to-Policy Compression Pilot

This experiment tests whether Countdown problems that Qwen3-1.7B solves only
through best-of-N sampling can be compressed into one-shot policy behavior with
TorchTitan-native LoRA SFT.

The v1 pipeline is deliberately simple:

- generate exact-DP-solvable Countdown pools;
- collect vLLM best-of-N rollouts from a frozen base model;
- annotate every rollout with exact verifier metadata;
- build five ChatDataLoader-compatible SFT arms;
- train Qwen3 LoRA adapters with TorchTitan configs;
- evaluate pass@k, bucket metrics, validity, length, diversity, and bootstrap
  intervals.

## Scope

MCTS is out of scope. Best-of-N sampling is the scaffold.

Reusable Python lives under:

- `torchtitan/experiments/countdown_search_distill/`

Runnable experiment state lives under:

- `experiments/countdown_search_distill/data/`
- `experiments/countdown_search_distill/results/`
- `experiments/countdown_search_distill/reports/`

## Artifact Schemas

Problem rows include `problem_id`, `numbers`, `target`, `prompt`,
`generation_config`, `solver_metadata`, and `canonical_solution`.

Rollout rows include `problem_id`, `model_id`, `condition`, `sample_index`,
`prompt_variant`, `text`, and optional token/logprob fields.

Annotation rows include `problem_id`, `sample_index`, `success`, `bucket`,
`steps_consumed`, `states`, `first_invalid`, `first_unreachable`,
`final_value`, `distance_to_target`, and `error`.

Training rows are JSONL records with `question`, `answer`, `condition`,
`problem_id`, `source_rollout_ids`, and optional `hint` or
`curriculum_stage`. They are directly consumable by `ChatDataLoader`.

## CPU-Testable CLI

```bash
python -m torchtitan.experiments.countdown_search_distill.cli generate-pool \
  --output experiments/countdown_search_distill/data/dev/problems.jsonl \
  --num-problems 50 \
  --seed 42 \
  --target-min 100 \
  --target-max 499 \
  --min-solution-depth 3 \
  --require-all-numbers
```

Use `evaluate-fixture` for offline tests with precomputed rollout text. Use
`evaluate-vllm` for real model rollouts.

## Evaluate Frozen Qwen3 With vLLM

```bash
python -m torchtitan.experiments.countdown_search_distill.cli evaluate-vllm \
  --problems experiments/countdown_search_distill/data/dev/problems.jsonl \
  --model ./assets/hf/Qwen3-1.7B \
  --num-rollouts 32 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-new-tokens 192 \
  --output experiments/countdown_search_distill/data/dev/evaluations.jsonl \
  --rollouts-output experiments/countdown_search_distill/data/dev/rollouts.jsonl \
  --annotations experiments/countdown_search_distill/data/dev/annotations.jsonl \
  --summary experiments/countdown_search_distill/data/dev/summary.json
```

The target difficulty regime for the frozen model is approximately:

```text
pass@1  in [0.05, 0.20]
pass@32 in [0.35, 0.70]
```

If the gap is absent, adjust problem generation before training. Useful knobs:

- number range
- target range
- minimum solution depth
- all-number use
- division-required filtering
- distinct solution count

## Dataset Arms

```bash
python -m torchtitan.experiments.countdown_search_distill.cli build-datasets \
  --evaluations experiments/countdown_search_distill/data/train/evaluations.jsonl \
  --output-dir experiments/countdown_search_distill/data/train \
  --matched-only
```

The generated arms are:

- `raw.jsonl`: shortest verified base-model success
- `clean.jsonl`: deterministic cleaned rewrite of that success
- `formatting.jsonl`: parsed operation trace plus exact `FINAL: <target>`
- `hindsight.jsonl`: hint-conditioned question with verified answer
- `curriculum.jsonl`: hint-present, hint-dropout, and hint-absent stages

The primary comparison should use matched problems where every compared arm has
a verified target. Report arm coverage separately.

## Rootfs Runners

Real GPU workloads in this checkout must run through the TorchTitan-local rootfs
wrapper. These scripts re-exec through `scripts/rootfs/enter_rootfs.sh` unless
`TORCHTITAN_IN_ROOTFS=1` is already set.

```bash
# Verify rootfs runtime, imports, GPU visibility, model assets, and vLLM defaults
experiments/countdown_search_distill/run_preflight.sh

# 500-problem calibration by default
experiments/countdown_search_distill/run_calibration.sh

# Full collection: train 2000, dev 500, IID 1000, OOD 500
experiments/countdown_search_distill/run_collect.sh

# Smoke collection: train 100, dev 50, IID 100, OOD 50
MODE=smoke experiments/countdown_search_distill/run_collect.sh

# Reduced pilot: calibration 200, train 500, dev 200, IID 300, OOD 200
MODE=reduced experiments/countdown_search_distill/run_calibration.sh
MODE=reduced experiments/countdown_search_distill/run_collect.sh

# Train one LoRA arm
ARM=raw NGPU=8 experiments/countdown_search_distill/run_train.sh

# Evaluate a split with vLLM
SPLIT=iid_test experiments/countdown_search_distill/run_eval.sh

# Evaluate a vLLM-ready exported LoRA adapter with the same pass@k path
SPLIT=dev \
LORA_ADAPTER=experiments/countdown_search_distill/results/adapters/full/raw \
LORA_NAME=raw \
experiments/countdown_search_distill/run_eval.sh

# Export all mode-scoped TorchTitan LoRA checkpoints to PEFT/vLLM adapters
MODE=full experiments/countdown_search_distill/run_export_adapters.sh

# Evaluate all exported adapters and validate the split/arm matrix
MODE=full experiments/countdown_search_distill/run_eval_adapters.sh

# Calibration, collection, all arms, base eval, adapter export, adapter eval
experiments/countdown_search_distill/run_full_pilot.sh
```

## TorchTitan Training Configs

Configs are in `torchtitan/models/qwen3/config_registry.py`:

- `qwen3_1_7b_countdown_lora_raw`
- `qwen3_1_7b_countdown_lora_clean`
- `qwen3_1_7b_countdown_lora_formatting`
- `qwen3_1_7b_countdown_lora_hindsight`
- `qwen3_1_7b_countdown_lora_curriculum`
- `qwen3_debugmodel_countdown_lora_smoke`

They use `ChatDataLoader`, `LoRAConverter(rank=32, alpha=64)`, bf16,
assistant-token-only loss masking through ChatDataLoader, `seq_len=512`,
effective global batch size 64, and LR `1e-4`.

## Selection Rule

Select the primary checkpoint by maximizing dev pass@1 subject to dev pass@32 no
worse than base dev pass@32 minus 0.02. Smoke runs validate plumbing only; they
are not scientific evidence. The first evidence run is the reduced pilot with
raw, hindsight, and curriculum arms. Run the full pilot only after at least one
reduced-pilot arm improves dev pass@1 by 5 absolute points or more while keeping
dev pass@32 within 2 absolute points of the base model, measured on the
base-elicitable dev bucket and reported alongside all-dev metrics.

Abort the reduced pilot before adapter training if calibration misses the target
band or matched coverage falls below 300 train examples or 100 dev problems.
Close the reduced pilot with `experiments/countdown_search_distill/RESULTS.md`
plus an immutable report under `experiments/countdown_search_distill/reports/`.

The automated gate is:

```bash
python -m torchtitan.experiments.countdown_search_distill.cli preflight-reduced \
  --calibration-decision experiments/countdown_search_distill/data/calibration/decision.json \
  --train-evaluations experiments/countdown_search_distill/data/train/evaluations.jsonl \
  --dev-evaluations experiments/countdown_search_distill/data/dev/evaluations.jsonl \
  --decision experiments/countdown_search_distill/data/reduced_preflight.json
```

`LORA_ADAPTER` must point to a vLLM-ready exported adapter directory. Do not point
it directly at an internal TorchTitan checkpoint unless a later export step has
verified that checkpoint layout is vLLM-compatible.

`run_full_pilot.sh` now runs the runtime preflight first. In `MODE=reduced` and
`MODE=full`, it also exports adapters and evaluates the full adapter matrix
after base dev/IID/OOD evaluation. `run_eval_adapters.sh` writes adapter
summaries under `results/eval/adapters/${MODE}` by default, skips existing
summaries in that mode-scoped root unless `FORCE=1`, and validates the matrix
with:

```bash
python -m torchtitan.experiments.countdown_search_distill.cli validate-eval-matrix \
  --eval-root experiments/countdown_search_distill/results/eval/adapters/full \
  --decision experiments/countdown_search_distill/results/eval/adapters/full/adapter_matrix_full.json \
  --splits dev iid_test ood_test \
  --arms raw clean formatting hindsight curriculum \
  --expected-problems dev=500 iid_test=1000 ood_test=500 \
  --num-rollouts 32
```

Each `run_full_pilot.sh` invocation writes a JSONL stage manifest under
`experiments/countdown_search_distill/results/manifests/`. Override `RUN_ID` or
`TORCHTITAN_COUNTDOWN_MANIFEST` to choose the manifest path. Each record includes
the run ID, mode, rootfs state, stage name, command argv, start time, end time,
duration, and return code. Stages that can reuse artifacts, such as adapter
export and adapter evaluation, also record fresh/reused artifact status when
run through the pilot.

Reduced and full runs also write:

```text
experiments/countdown_search_distill/results/manifests/report_input_<run_id>.json
```

This structured report input collects the manifest, runtime preflight, split
registry, base summaries, adapter matrix, compact metrics, and small artifact
hashes. Use it as the source for future promotion gates and generated reports.

Summary files report both arithmetic-trace pass@k and
`strict_format_pass_at_k`. The strict-format metric requires a successful trace
and an exact `FINAL: <target>` line, so output-contract regressions are visible
separately from arithmetic success.

When evaluation JSONL files are present, `build-report-input` also adds an
`analysis` block with:

- `base_elicitable_subsets`: base and adapter metrics restricted to problems
  that the base model solved within 32 samples but not at sample 1;
- `representative_examples`: deterministic examples for wins, regressions,
  unchanged failures, and format failures, with compact problem metadata and
  bounded rollout excerpts.

## Replication And Sweep Runs

Use `run_replication_sweep.sh` for second-draw or champion-arm sweep runs. It
always re-enters the bwrap rootfs and writes isolated artifacts under:

```text
experiments/countdown_search_distill/sweeps/replication/<label>/
```

The default run is a second full-mode `clean` arm draw with seed 43, train size
2000, and LoRA rank 32:

```bash
experiments/countdown_search_distill/run_replication_sweep.sh
```

The script accepts whitespace-separated sweep axes:

```bash
SEEDS="43 44" \
TRAIN_SIZES="1000 2000" \
LORA_RANKS="16 32" \
ARMS=clean \
NGPU=8 \
experiments/countdown_search_distill/run_replication_sweep.sh
```

Each point sets `TORCHTITAN_COUNTDOWN_DATA_ROOT` and
`TORCHTITAN_COUNTDOWN_RESULTS_ROOT` before calling `run_full_pilot.sh`, so it
does not overwrite the canonical pilot data or results. The qwen3 Countdown
training configs read those roots and `TORCHTITAN_COUNTDOWN_LORA_RANK` for the
LoRA rank. Export uses the same rank before vLLM adapter evaluation.

`run_full_pilot.sh` also accepts `ARMS=<list>` for scoped reruns. Report inputs
record the scoped data/results roots and validate only the requested training
arms while still requiring calibration, split validation, base evaluations,
adapter export, and adapter evaluation.

