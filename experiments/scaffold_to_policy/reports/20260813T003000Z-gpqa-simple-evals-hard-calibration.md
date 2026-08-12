# GPQA Simple-Evals Hard Calibration

Run id: `20260813T003000Z-gpqa-simple-evals-16x16-labeled`

This run closes the previous GPQA access blocker for the public-cache path. It
uses the OpenAI simple-evals GPQA Diamond CSV as an offline raw cache, then
executes the existing no-tool multiple-choice GPQA runner inside the bwrap
rootfs with Qwen3-1.7B and vLLM.

This is not an official GPQA leaderboard score. It is a small, fixed-slice
hard-reasoning calibration over public GPQA Diamond rows, with exact
`FINAL: <A|B|C|D>` verification and no tools.

## Commands

Cache command:

```bash
scripts/rootfs/enter_rootfs.sh -- python -m torchtitan.experiments.scaffold_to_policy.cli \
  cache-gpqa-simple-evals-csv \
  --output experiments/scaffold_to_policy/data/gpqa_simple_evals/raw/gpqa_diamond.jsonl \
  --provenance experiments/scaffold_to_policy/data/gpqa_simple_evals/raw/gpqa_diamond.provenance.json
```

Calibration command:

```bash
RUN_ID=20260813T003000Z-gpqa-simple-evals-16x16-labeled \
DATA_ROOT=experiments/scaffold_to_policy/data/gpqa_simple_evals_16x16_labeled \
RESULTS_ROOT=experiments/scaffold_to_policy/results/gpqa_simple_evals_16x16_labeled \
DATASET=openai/simple-evals-gpqa \
DATASET_SUBSET=gpqa_diamond \
DATASET_REVISION=main \
SOURCE_SPLIT=gpqa_diamond \
OFFLINE=1 \
DEV_RAW_CACHE=experiments/scaffold_to_policy/data/gpqa_simple_evals/raw/gpqa_diamond.jsonl \
OOD_RAW_CACHE=experiments/scaffold_to_policy/data/gpqa_simple_evals/raw/gpqa_diamond.jsonl \
DEV_OFFSET=0 OOD_OFFSET=64 \
DEV_PROBLEMS=16 OOD_PROBLEMS=16 \
NUM_ROLLOUTS=4 MAX_NEW_TOKENS=512 GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_gpqa_public_vllm_smoke.sh
```

The shell runner re-entered `scripts/rootfs/enter_rootfs.sh` automatically and
all recorded stages have `rootfs_active=true`.

## Source And Provenance

The cache source is:

```text
https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv
```

Cache provenance:

| Field | Value |
| --- | --- |
| Rows | `198` |
| Source label | `openai-simple-evals:gpqa_diamond` |
| Cache hash | `312b80837f84194c3d18675a3f4cd3cabea6d767ef2d1cebacb62d8b0df49d7e` |
| Required fields | `Question`, `Correct Answer`, `Incorrect Answer 1`, `Incorrect Answer 2`, `Incorrect Answer 3` |

The imported split registry selected 16 dev rows at offset 0 and 16 OOD rows at
offset 64. It recorded no overlap and verifier
`multiple_choice_final_letter_v1`.

## Runtime

Runtime metadata records:

| Field | Value |
| --- | --- |
| Rootfs active | `true` |
| Model | `./assets/hf/Qwen3-1.7B` |
| Torch | `2.13.0+cu132` |
| CUDA | `13.2` |
| GPUs visible | `8 x NVIDIA B200` |
| vLLM | `0.27.0` |
| Max model length | `2048` |
| Prompt variant | `chat` |
| Rollouts per problem | `4` |
| Temperature / top-p | `0.2 / 0.95` |
| GPU memory utilization | `0.05` |
| Attention backend | `TRITON_ATTN` |

vLLM loaded Qwen3-1.7B from local safetensors, captured CUDA graphs, and
generated both splits successfully. The runner still starts one vLLM engine per
split, which is operationally correct but inefficient for larger calibrations.

## Metrics

| Split | Problems | Rollouts | pass@1 | pass@2 | pass@4 | pass@32 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 16 | 64 | 0.4375 | 0.5625 | 0.5625 | 0.5625 | easy 7, elicitable 2, unreached 7 |
| OOD | 16 | 64 | 0.3125 | 0.3750 | 0.4375 | 0.4375 | easy 5, elicitable 2, unreached 9 |

Strict-format pass@k matched pass@k for all reported cutoffs. The model mostly
failed by choosing a wrong final letter rather than by format drift, though
four dev and four OOD rollouts missed the final-answer contract.

Failure breakdown:

| Split | Successes | Wrong B | Wrong C | Wrong D | Missing final |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev | 23 | 12 | 17 | 8 | 4 |
| OOD | 21 | 8 | 13 | 18 | 4 |

## Examples

### Elicitable Dev Example

Problem `mc-d1018788c9968c11` asks for the resolvable energy difference between
two quantum states with lifetimes `10^-9 sec` and `10^-8 sec`. Choices are:

```text
A. 10^-4 eV
B. 10^-11 eV
C. 10^-8 eV
D. 10^-9 eV
```

Answer: `A`

The four rollouts produced `B`, `A`, `A`, `D`, so the problem is elicitable:
sample 1 was wrong, then samples 2 and 3 selected the correct final letter.

Representative correct final line:

```text
FINAL: A
```

### Unreached Dev Example

Problem `mc-f9300e7e6555e809` asks for the number of carbon atoms in the final
product of a reaction sequence beginning with trans-cinnamaldehyde. Choices are:

```text
A. 11
B. 10
C. 12
D. 14
```

Answer: `A`

All four rollouts selected `D` or `C`. The model repeatedly guessed from
reaction complexity instead of preserving the carbon count through the sequence.

### Easy OOD Example

Problem `mc-be11de1013a1bea5` asks for the joint probability of measuring both
orbital angular momentum components as `-hbar` in the coupled state
`|1,1,2,-1>`. Choices are:

```text
A. 0
B. 1/2
C. 1
D. 2/3
```

Answer: `A`

All four rollouts ended with `FINAL: A`, making this an easy OOD item under the
current bucket definition.

## Infra Notes

- The previous live Hugging Face GPQA path remains gated without `HF_TOKEN`, but
  the public simple-evals CSV provides a benchmark-preserving GPQA Diamond cache
  path for this experiment.
- `_load_public_rows` now applies `limit` and `offset` to offline/raw-cache
  imports, matching the live dataset slicing contract.
- `cache-gpqa-simple-evals-csv` writes raw JSONL plus provenance rather than
  directly writing normalized problem rows. The existing `import-gpqa-split`
  command remains the single normalization boundary.
- The four-choice prompt bug was fixed in `multiple_choice.prompt_for_problem`;
  it now zips choices with the problem's actual letter set instead of all ten
  possible letters.
- Generated cache and result trees remain untracked. The durable artifacts are
  the code, runner docs, and this report.

## Interpretation

The run adds a second hard multiple-choice reasoning calibration next to the
existing MMLU-Pro 16/16 run. Compared with MMLU-Pro, GPQA is lower on both dev
and OOD pass@1, and has more elicitable headroom than a pure hard-negative lane.
The next useful reasoning experiment is therefore not more plumbing; it is a
matched prompt/scaffold comparison on the same cached GPQA split, preserving the
same verifier and offsets.
