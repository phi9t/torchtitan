# Hard Reasoning And Coding Runtime-Metadata Refresh

Run IDs:

- `20260812T230000Z-mmlu-pro-16x16-runtime`
- `20260812T231500Z-bigcodebench-hard-8x8-runtime`
- `20260812T223000Z-gpqa-runtime-metadata`

This refresh reran the latest hard reasoning and hard coding slices after the
report contract gained explicit runtime metadata. The goal was not to change
benchmark scope; it was to make the current report inputs auditable against the
rootfs, CUDA, package, model, and vLLM settings required by
`spec.md`.

All commands entered through `scripts/rootfs/enter_rootfs.sh`. The generated
data and result trees are runtime artifacts and remain out of git.

## Commands

MMLU-Pro:

```bash
RUN_ID=20260812T230000Z-mmlu-pro-16x16-runtime \
DATA_ROOT=experiments/scaffold_to_policy/data/mmlu_pro_public_vllm_16x16_runtime \
RESULTS_ROOT=experiments/scaffold_to_policy/results/mmlu_pro_public_vllm_16x16_runtime \
DEV_PROBLEMS=16 OOD_PROBLEMS=16 DEV_OFFSET=0 OOD_OFFSET=32 \
NUM_ROLLOUTS=4 GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_mmlu_pro_public_vllm_smoke.sh
```

BigCodeBench-Hard:

```bash
RUN_ID=20260812T231500Z-bigcodebench-hard-8x8-runtime \
DATA_ROOT=experiments/scaffold_to_policy/data/bigcodebench_hard_contract_chat_8x8_runtime \
RESULTS_ROOT=experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_8x8_runtime \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 DEV_OFFSET=0 OOD_OFFSET=32 \
NUM_ROLLOUTS=8 PROMPT_VARIANT=contract_chat MAX_NEW_TOKENS=768 \
TIMEOUT_SECONDS=30 GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_bigcodebench_hard_public_vllm_smoke.sh
```

GPQA blocker refresh:

```bash
RUN_ID=20260812T223000Z-gpqa-runtime-metadata \
DATA_ROOT=experiments/scaffold_to_policy/data/gpqa_public_vllm_runtime_metadata \
RESULTS_ROOT=experiments/scaffold_to_policy/results/gpqa_public_vllm_runtime_metadata \
DEV_PROBLEMS=1 OOD_PROBLEMS=1 GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_gpqa_public_vllm_smoke.sh
```

## Runtime Metadata

Each refreshed report input now includes top-level `runtime` metadata and
artifact provenance for the runtime JSON file.

Shared runtime facts recorded by the new report inputs:

| Field | Value |
| --- | --- |
| Rootfs active | `true` |
| Model path | `./assets/hf/Qwen3-1.7B` |
| Model files | `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json` present |
| Safetensors shards | `2` |
| CUDA available | `true` |
| CUDA device count | `8` |
| Torch | `2.13.0+cu132` |
| vLLM | `0.27.0` |
| datasets | `4.7.0` |
| transformers | `5.15.0` |
| spmd_types | `0.2.1` |
| vLLM attention backend | `TRITON_ATTN` |
| FlashInfer sampler | `0` |
| FlashInfer autotune | `false` |
| GPU memory utilization | `0.05` |
| Max model length | `2048` |

Report inputs:

```text
experiments/scaffold_to_policy/results/mmlu_pro_public_vllm_16x16_runtime/manifests/report_input_20260812T230000Z-mmlu-pro-16x16-runtime.json
experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_8x8_runtime/manifests/report_input_20260812T231500Z-bigcodebench-hard-8x8-runtime.json
experiments/scaffold_to_policy/results/gpqa_public_vllm_runtime_metadata/manifests/report_input_20260812T223000Z-gpqa-runtime-metadata.json
```

All three report inputs set `runtime_metadata_present=true` and
`artifact_provenance_labeled=true`.

## MMLU-Pro Results

The MMLU-Pro run used the same no-tool exact-final-letter condition as the
prior 16/16 calibration: `TIGER-Lab/MMLU-Pro`, validation split, revision
`main`, dev offset `0`, OOD offset `32`, four rollouts per problem, and Qwen3
1.7B through vLLM.

| Split | Problems | Rollouts | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 16 | 64 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | easy 8, elicitable 0, unreached 8 |
| OOD | 16 | 64 | 0.5625 | 0.6250 | 0.6875 | 0.6875 | 0.6875 | 0.6875 | easy 9, elicitable 2, unreached 5 |

Strict-format pass@k matched answer pass@k on both splits. Failure aggregation
still separates missing final answers from verifier-readable wrong final
letters.

## BigCodeBench-Hard Results

The BigCodeBench-Hard run used the no-tool `contract_chat` condition over
`bigcode/bigcodebench-hard`, split `v0.1.4`, revision `main`, dev offset `0`,
OOD offset `32`, and eight rollouts per problem. Released canonical solutions
passed preflight on all 16 selected tasks before model scoring.

| Split | Problems | Rollouts | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 8 | 64 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 8 |
| OOD | 8 | 64 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 8 |

Failure breakdown:

| Split | Assertion failures | Syntax errors |
| --- | ---: | ---: |
| dev | 64 | 0 |
| OOD | 63 | 1 |

This remains a hard-negative coding result, not a leaderboard score.

## GPQA Status

Update: this section is superseded for the public-cache GPQA path by
`experiments/scaffold_to_policy/reports/20260813T003000Z-gpqa-simple-evals-hard-calibration.md`.
The live Hugging Face loader path described below remains gated without
credentials, but the OpenAI simple-evals GPQA Diamond CSV was later cached and
run through the existing rootfs/vLLM GPQA runner.

The refreshed rootfs run in this report attempted `Idavidrein/gpqa`, subset
`gpqa_diamond`, and stopped at the benchmark-preserving Hugging Face access
boundary:

```text
datasets.exceptions.DatasetNotFoundError: Dataset 'Idavidrein/gpqa' is a gated
dataset on the Hub. You must be authenticated to access it.
```

Additional inspection found no `HF_TOKEN` in the host environment, no token in
the rootfs-visible environment, no rootfs Hugging Face token file, and no
authorized raw GPQA JSONL rows. The local Hugging Face cache contains only a
small metadata blob for `datasets--Idavidrein--gpqa`, not usable benchmark rows.

The blocker report records `benchmark_execution_completed=false`,
`blocker_artifacts_present=true`, `runtime_metadata_present=true`, and
`artifact_provenance_labeled=true`.

## Interpretation

The current hard-lane artifacts in this report satisfy the runtime-metadata
requirements for the executed public reasoning and coding slices. MMLU-Pro
remains useful positive no-tool reasoning calibration. BigCodeBench-Hard remains
a clean hard-negative no-tool coding calibration with canonical verifier
preflight selected. For GPQA, use the later public simple-evals calibration
report for executed benchmark-task evidence, and use this report only as the
gated Hugging Face access diagnostic.
