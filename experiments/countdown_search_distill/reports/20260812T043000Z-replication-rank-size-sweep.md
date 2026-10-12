# Countdown Replication And Rank-Size Sweep

Date: 2026-08-12

## Scope

This report records the completed clean-arm replication and rank-size sweep for
the Countdown scaffold-to-policy experiment. The run family used a fresh split
draw/training seed and isolated data/results roots for each cell:

```bash
RUN_ID_PREFIX=20260812T-sweep \
SEEDS=43 \
TRAIN_SIZES='1000 2000' \
LORA_RANKS='32 16' \
ARMS=clean \
NGPU=8 \
experiments/countdown_search_distill/run_replication_sweep.sh
```

The command was launched through:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc '<command>'
```

All real generation, training, export, vLLM evaluation, matrix validation, and
report-input generation stayed inside the bwrap rootfs. The final sweep exited
with code 0 and wrote:

```text
experiments/countdown_search_distill/sweeps/replication/replication_sweep_full.jsonl
```

## Runtime And Infrastructure

The sweep used the same hermetic runtime discipline as the pilot:

- TorchTitan training ran under the bwrap rootfs with `NGPU=8`.
- Base and adapter generation used vLLM against `assets/hf/Qwen3-1.7B`.
- TorchTitan checkpoints were exported to PEFT/vLLM LoRA adapter directories
  before adapter evaluation.
- Each sweep cell used isolated roots under
  `experiments/countdown_search_distill/sweeps/replication/<label>/`.
- The rootfs entrypoint managed Python imports, CUDA runtime, TorchTitan,
  vLLM, PEFT export, and report generation.

Per-cell artifacts:

| Label | Train size | LoRA rank | Data root | Results root |
| --- | ---: | ---: | --- | --- |
| `seed43_train1000_rank32` | 1000 | 32 | `sweeps/replication/seed43_train1000_rank32/data/` | `sweeps/replication/seed43_train1000_rank32/results/` |
| `seed43_train1000_rank16` | 1000 | 16 | `sweeps/replication/seed43_train1000_rank16/data/` | `sweeps/replication/seed43_train1000_rank16/results/` |
| `seed43_train2000_rank32` | 2000 | 32 | `sweeps/replication/seed43_train2000_rank32/data/` | `sweeps/replication/seed43_train2000_rank32/results/` |
| `seed43_train2000_rank16` | 2000 | 16 | `sweeps/replication/seed43_train2000_rank16/data/` | `sweeps/replication/seed43_train2000_rank16/results/` |

Every cell produced:

- `results/train/full/clean/checkpoint/step-94/`
- `results/adapters/full/clean/adapter_config.json`
- `results/adapters/full/clean/adapter_model.safetensors`
- `results/adapters/full/clean/export_summary.json`
- `results/eval/adapters/full/adapter_matrix_full.json`
- `results/manifests/report_input_20260812T-sweep-<label>.json`

Validation checked each adapter matrix, report input, PEFT adapter directory,
checkpoint directory, and sweep-summary row. All four cells passed. The
append-safe summary currently has five rows because the earlier
`seed43_train2000_rank32` replication row was preserved and the sweep appended
the refreshed/current row rather than truncating history.

## Aggregate Results

Each cell evaluates the `clean` adapter with 32 rollouts per problem:
500 dev problems, 1000 IID test problems, and 500 OOD test problems.

| Label | Dev pass@1 | Dev pass@32 | IID pass@1 | IID pass@32 | OOD pass@1 | OOD pass@32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `seed43_train1000_rank32` | 0.168 | 0.894 | 0.179 | 0.885 | 0.280 | 0.902 |
| `seed43_train1000_rank16` | 0.186 | 0.880 | 0.157 | 0.876 | 0.316 | 0.906 |
| `seed43_train2000_rank32` | 0.174 | 0.900 | 0.205 | 0.887 | 0.276 | 0.890 |
| `seed43_train2000_rank16` | 0.192 | 0.918 | 0.163 | 0.905 | 0.264 | 0.914 |

Strict-format metrics:

| Label | Dev strict pass@1 | Dev strict pass@32 | IID strict pass@1 | IID strict pass@32 | OOD strict pass@1 | OOD strict pass@32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `seed43_train1000_rank32` | 0.074 | 0.680 | 0.065 | 0.634 | 0.116 | 0.754 |
| `seed43_train1000_rank16` | 0.096 | 0.666 | 0.065 | 0.654 | 0.128 | 0.754 |
| `seed43_train2000_rank32` | 0.066 | 0.678 | 0.075 | 0.688 | 0.120 | 0.722 |
| `seed43_train2000_rank16` | 0.050 | 0.620 | 0.056 | 0.649 | 0.094 | 0.708 |

Bucket movement:

| Label | Split | Easy | Elicitable | Unreached |
| --- | --- | ---: | ---: | ---: |
| `seed43_train1000_rank32` | dev | 84 | 363 | 53 |
| `seed43_train1000_rank32` | iid_test | 179 | 706 | 115 |
| `seed43_train1000_rank32` | ood_test | 140 | 311 | 49 |
| `seed43_train1000_rank16` | dev | 93 | 347 | 60 |
| `seed43_train1000_rank16` | iid_test | 157 | 719 | 124 |
| `seed43_train1000_rank16` | ood_test | 158 | 295 | 47 |
| `seed43_train2000_rank32` | dev | 87 | 363 | 50 |
| `seed43_train2000_rank32` | iid_test | 205 | 682 | 113 |
| `seed43_train2000_rank32` | ood_test | 138 | 307 | 55 |
| `seed43_train2000_rank16` | dev | 96 | 363 | 41 |
| `seed43_train2000_rank16` | iid_test | 163 | 742 | 95 |
| `seed43_train2000_rank16` | ood_test | 132 | 325 | 43 |

## Base-Elicitable Subset

The strongest cell by dev pass@1/pass@32 and OOD pass@32 is
`seed43_train2000_rank16`. Its base-elicitable subset results are:

| Split | Arm | Problems | pass@1 | pass@32 | strict pass@1 | strict pass@32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dev | base | 226 | 0.000 | 1.000 | 0.000 | 0.279 |
| dev | clean | 226 | 0.221 | 0.991 | 0.053 | 0.774 |
| iid_test | base | 466 | 0.000 | 1.000 | 0.000 | 0.365 |
| iid_test | clean | 466 | 0.221 | 0.994 | 0.073 | 0.781 |
| ood_test | base | 279 | 0.000 | 1.000 | 0.000 | 0.552 |
| ood_test | clean | 279 | 0.351 | 1.000 | 0.151 | 0.889 |

This is the cleanest evidence that the scaffold-to-policy effect replicated
under a fresh seed/split draw: the clean adapter converts 35.1% of OOD
base-elicitable problems to pass@1 while preserving 100% pass@32 on that
subset.

## Examples

Examples below are artifact-selected from
`seed43_train2000_rank16/results/manifests/report_input_20260812T-sweep-seed43_train2000_rank16.json`.

### OOD Win

Problem:

```text
numbers: 100, 25, 54
target: 179
canonical solution:
54 + 100 = 154
154 + 25 = 179
FINAL: 179
```

Base sample 1 missed the strict final line and solved only at sample 16. The
clean adapter solved at sample 1 and also had strict success at sample 1. The
adapter response was verbose and prompt-like, but it contained a valid early
trace:

```text
100 + 25 = 125
125 + 54 = 179
FINAL: 179
```

### OOD Regression

Problem:

```text
numbers: 12, 65, 68
target: 145
canonical solution:
68 + 12 = 80
80 + 65 = 145
FINAL: 145
```

Base solved strictly at sample 1. The clean adapter's first sample reasoned
about the right additive structure but missed the strict final line and solved
only at sample 3. This is the recurring failure mode for the `clean` arm:
arithmetic competence improved, but strict output-contract reliability remains
weaker than the later `formatting` arm.

### OOD Unchanged Failure

Problem:

```text
numbers: 10, 65, 51
target: 140
canonical solution:
65 - 51 = 14
14 * 10 = 140
FINAL: 140
```

Both base and clean stayed unreached. The adapter attempted traces that reused
unavailable intermediate values, for example:

```text
10 + 65 = 75
75 + 51 = 126
126 + 65 = 191
```

The verifier rejected this as `operation consumes unavailable numbers`.

### OOD Format Failure

Problem:

```text
numbers: 100, 85, 50
target: 135
canonical solution:
85 + 100 = 185
185 - 50 = 135
FINAL: 135
```

The clean adapter solved arithmetically at sample 1 but did not emit a strict
`FINAL: 135` line before drifting into explanation text. This is why strict
format metrics remain materially lower than arithmetic pass@k for the clean
arm.

## Interpretation

The replication sweep supports the pilot's core claim: Countdown best-of-32
behavior can be compressed into materially better first-sample policy behavior
with TorchTitan LoRA SFT, and the effect persists under a fresh seed/split draw.

The rank-size sweep does not show a monotonic "more data and higher rank is
better" trend. Rank 16 with 2000 training examples is strongest on dev pass@1,
dev pass@32, IID pass@32, OOD pass@32, and OOD base-elicitable pass@1. Rank 32
with 2000 examples is strongest on IID pass@1. Rank 16 with 1000 examples is
strongest on OOD pass@1 but not on OOD pass@32.

The strict-format results remain a separate issue for the clean arm. The later
formatting replication in
`reports/20260812T054500Z-formatting-replication.md` replicated the stronger
strict-output strategy under the same seed-43, train-2000, rank-16 sweep
layout.

## Infrastructure Notes

What worked well:

- The bwrap rootfs path was sufficient for TorchTitan training, vLLM base
  generation, PEFT export, and vLLM LoRA evaluation.
- Isolated data/results roots prevented the sweep from overwriting the earlier
  full-pilot and formatting artifacts.
- `run_replication_sweep.sh` reused `run_full_pilot.sh`, so calibration,
  collection, split validation, training, export, evaluation, and report-input
  generation stayed on one orchestration path.
- The PEFT export boundary remained explicit; vLLM never received a TorchTitan
  internal checkpoint as `LORA_ADAPTER`.
- The append-safe sweep summary preserved earlier rows instead of truncating
  prior evidence.

What needs improvement:

- The sweep summary is append-only and can contain multiple rows per label. The
  report layer should choose a canonical latest row by run ID or timestamp.
- Dataset loading still emitted network HEAD requests from the `datasets` JSON
  loader path during earlier runs. The next infra hardening should force a
  local/offline dataset loader implementation for generated JSONL files.
- vLLM LoRA evaluation reloads the engine for each split. This is robust, but
  expensive; a future runner could evaluate all splits in one process while
  preserving separate summaries.
- Clean-arm generations still overproduce prompt-like text and often miss
  strict `FINAL:` despite arithmetic success. Formatting/stop-quality shaping
  should be replicated next.
- Manifest provenance records the key artifacts, but reused-versus-fresh stage
  status should be surfaced directly in the report tables.

## Decision

Countdown cleared the replication gate for the clean arm in this run, and the
follow-up formatting replication has since cleared the strict-output champion
replication gate. The broader program can now start the reasoning-first
expansion:

1. Add one local exact-verifier arithmetic word-problem task.
2. Add one local symbolic or constraint-puzzle task with deterministic checking.
3. Add a small public no-tool numeric reasoning smoke only after local task
   registry/reporting works.
4. Keep Terminal-Bench/Harbor and tau2-bench as later rootfs-managed harness
   smokes until the reasoning lane has replicated transfer evidence.
